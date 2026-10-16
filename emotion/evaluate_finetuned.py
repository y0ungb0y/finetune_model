#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""使用微调后的 LoRA 模型在测试集上评估情感分类准确率。"""

import os
import json
import time
import random
from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

BASE_MODEL_PATH = "/data/xinwang/finetune_model"
DATA_FILE = "/data/xinwang/finetune_model/emotion/平衡后的客户对话数据.json"
ADAPTER_OUTPUT_DIR = "/data/xinwang/finetune_model/emotion/lora_adapter"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42

BATCH_SIZE = 16
MAX_LENGTH = 512

# 输出文件路径
OUTPUT_RESULTS_FILE = "/data/xinwang/finetune_model/emotion/evaluate_results.json"
OUTPUT_ERRORS_FILE = "/data/xinwang/finetune_model/emotion/evaluate_errors.json"


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_prompt(text: str) -> str:
    system_message = (
        "你是一个情感分析专家，专门识别客户对话中的负面情绪。"
        "只有当对话中包含明显的生气、不耐烦、愤怒等负面语气词时才判断为负面，否则默认为正面。"
    )
    user_message = (
        "请判断下面这句话是否包含明显的生气或不耐烦的语气。\n\n"
        "判断标准：\n"
        "- 如果句子中有明显的生气、不耐烦、愤怒、抱怨等负面语气词（如：烦死了、气死了、什么破、太差了、垃圾、投诉、退款等），则判断为\"负面\"\n"
        "- 否则默认判断为\"正面\"\n\n"
        f"句子：{text}\n\n"
        "请只回答\"正面\"或\"负面\"，不要有其他解释。\n\n"
        "情感："
    )
    return f"System: {system_message}\n\nUser: {user_message}"


def load_test_split() -> List[Dict[str, str]]:
    """按与训练相同的方式从全集中构造测试集。"""
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 与 finetune.load_dialogue_data 一致的清洗逻辑
    cleaned = [
        item
        for item in data
        if item.get("语句内容") and item.get("情绪积极") in {"正面", "负面"}
    ]

    positive_samples = [item for item in cleaned if item["情绪积极"] == "正面"]
    negative_samples = [item for item in cleaned if item["情绪积极"] == "负面"]

    # 使用相同随机种子和打乱方式
    random.shuffle(positive_samples)
    random.shuffle(negative_samples)

    total_positive = len(positive_samples)
    total_negative = len(negative_samples)

    train_pos = int(total_positive * TRAIN_RATIO)
    val_pos = int(total_positive * VAL_RATIO)
    test_pos = total_positive - train_pos - val_pos

    train_neg = int(total_negative * TRAIN_RATIO)
    val_neg = int(total_negative * VAL_RATIO)
    test_neg = total_negative - train_neg - val_neg

    test_data = (
        positive_samples[train_pos + val_pos :]
        + negative_samples[train_neg + val_neg :]
    )

    random.shuffle(test_data)

    print(f"测试集大小: {len(test_data)} 条 (正面:{test_pos}, 负面:{test_neg})")
    return test_data


def load_finetuned_model():
    """加载基础模型并应用 LoRA 适配器。"""
    print("\n加载微调后的 LoRA 模型...")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(model, ADAPTER_OUTPUT_DIR)

    if not torch.cuda.is_available():
        model = model.to("cpu")

    model.eval()
    print("模型加载完成！\n")
    return tokenizer, model


def predict_batch(tokenizer, model, texts: List[str]) -> List[str]:
    """对一个批次的文本进行预测，返回"正面"/"负面"标签。"""
    device = next(model.parameters()).device

    # 与训练时一致的 prompt 形式
    prompts = [create_prompt(text) for text in texts]

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    predictions: List[str] = []
    for response in responses:
        if "情感：" in response:
            answer = response.split("情感：")[-1].strip()
        else:
            answer = response.strip()

        if "负面" in answer:
            predictions.append("负面")
        elif "正面" in answer:
            predictions.append("正面")
        else:
            if "负" in answer or "消极" in answer or "不好" in answer:
                predictions.append("负面")
            else:
                predictions.append("正面")

    return predictions


def evaluate_on_test_set():
    """在测试集上评估微调模型。"""
    set_seed(SEED)

    test_data = load_test_split()
    texts = [item["语句内容"] for item in test_data]
    true_labels = [item["情绪积极"] for item in test_data]

    tokenizer, model = load_finetuned_model()

    print(f"开始在测试集上预测，共 {len(texts)} 条...")
    predictions: List[str] = []

    num_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    start_time = time.time()

    pbar = tqdm(total=num_batches, desc="测试集预测进度", ncols=100)
    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i : i + BATCH_SIZE]
        batch_preds = predict_batch(tokenizer, model, batch_texts)
        predictions.extend(batch_preds)
        pbar.update(1)
    pbar.close()

    total_time = time.time() - start_time
    print(f"\n预测完成，总耗时 {total_time:.2f} 秒，平均每条 {total_time/len(texts):.3f} 秒")

    # 计算指标
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average="weighted", zero_division=0
    )
    cm = confusion_matrix(true_labels, predictions, labels=["正面", "负面"])

    print("\n================ 测试集评估结果 ================")
    print(f"测试样本数: {len(texts)}")
    print(f"准确率 (Accuracy):  {accuracy*100:.4f}%")

    print("\n混淆矩阵 (真实 / 预测):")
    print(f"                 预测正面    预测负面")
    print(f"  真实正面:      {cm[0][0]:6d}      {cm[0][1]:6d}")
    print(f"  真实负面:      {cm[1][0]:6d}      {cm[1][1]:6d}")
    print("================================================\n")

    # 保存详细结果
    results = {
        "总体统计": {
            "总样本数": len(texts),
            "真实正面数": true_labels.count('正面'),
            "真实负面数": true_labels.count('负面'),
            "预测正面数": predictions.count('正面'),
            "预测负面数": predictions.count('负面')
        },
        "性能指标": {
            "准确率": float(accuracy),
            "精确率": float(precision),
            "召回率": float(recall),
            "F1分数": float(f1)
        },
        "混淆矩阵": {
            "真实正面_预测正面": int(cm[0][0]),
            "真实正面_预测负面": int(cm[0][1]),
            "真实负面_预测正面": int(cm[1][0]),
            "真实负面_预测负面": int(cm[1][1])
        },
        "预测详情": [
            {
                "文本": text,
                "真实标签": true_label,
                "预测标签": pred_label,
                "正确": true_label == pred_label
            }
            for text, true_label, pred_label in zip(texts, true_labels, predictions)
        ]
    }

    # 保存结果到JSON文件
    with open(OUTPUT_RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"详细结果已保存至: {OUTPUT_RESULTS_FILE}")

    # 分析并保存错误案例
    errors = [
        {
            "文本": text,
            "真实标签": true_label,
            "预测标签": pred_label
        }
        for text, true_label, pred_label in zip(texts, true_labels, predictions)
        if true_label != pred_label
    ]

    print(f"\n错误总数: {len(errors)} ({len(errors)/len(texts)*100:.2f}%)")

    # 统计错误类型
    false_positive = sum(1 for e in errors if e["真实标签"] == "正面" and e["预测标签"] == "负面")
    false_negative = sum(1 for e in errors if e["真实标签"] == "负面" and e["预测标签"] == "正面")

    print(f"\n错误类型:")
    print(f"  假阳性（正面被误判为负面）: {false_positive} ({false_positive/len(errors)*100:.2f}%)" if len(errors) > 0 else "  假阳性: 0")
    print(f"  假阴性（负面被误判为正面）: {false_negative} ({false_negative/len(errors)*100:.2f}%)" if len(errors) > 0 else "  假阴性: 0")

    # 显示部分错误案例
    if len(errors) > 0:
        print(f"\n前10个错误案例:")
        for i, error in enumerate(errors[:10], 1):
            print(f"\n  [{i}] 文本: {error['文本'][:50]}...")
            print(f"      真实: {error['真实标签']} | 预测: {error['预测标签']}")

    # 保存错误案例到JSON文件
    with open(OUTPUT_ERRORS_FILE, 'w', encoding='utf-8') as f:
        json.dump(errors, f, ensure_ascii=False, indent=2)
    print(f"\n错误案例已保存至: {OUTPUT_ERRORS_FILE}")
    print("\n评估完成！")


if __name__ == "__main__":
    evaluate_on_test_set()
