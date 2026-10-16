#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用微调后的LoRA模型评估同理心分类准确率
"""

import os
import json
import random
import re
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import warnings
import logging

warnings.filterwarnings('ignore')
logging.getLogger('transformers').setLevel(logging.ERROR)
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# ==================== 配置参数 ====================
BASE_MODEL_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADAPTER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lora_adapter")
DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "所有会话_同理心数据.json")

BATCH_SIZE = 4
MAX_LENGTH = 1024
CONTEXT_TURNS = 5
SAMPLE_SESSIONS = 500  # 抽取的会话数量

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_RESULTS_FILE = os.path.join(OUTPUT_DIR, "finetuned_predict_results.json")
OUTPUT_ERRORS_FILE = os.path.join(OUTPUT_DIR, "finetuned_predict_errors.json")
# ==================================================


def parse_dialogue_line(line):
    """解析对话行"""
    match = re.match(r'^(客户|客服): (.+?) (合规|不合规)$', line)
    if match:
        return match.group(1), match.group(2), match.group(3)
    return None, None, None


def extract_samples_from_data(data):
    """从对话数据中提取样本（对每一句客服对话进行分类）"""
    samples = []
    
    for session in data:
        session_id = session.get("会话编号", "")
        dialogue = session.get("对话内容", [])
        
        parsed_dialogue = []
        for line in dialogue:
            role, content, label = parse_dialogue_line(line)
            if role and content:
                parsed_dialogue.append({
                    "role": role,
                    "content": content,
                    "label": label
                })
        
        # 对每一句客服对话进行分类
        for i, turn in enumerate(parsed_dialogue):
            if turn["role"] == "客服":
                start_idx = max(0, i - CONTEXT_TURNS)
                context = parsed_dialogue[start_idx:i]
                
                samples.append({
                    "session_id": session_id,
                    "context": context,
                    "utterance": turn["content"],
                    "true_label": turn["label"],
                    "turn_index": i
                })
    
    return samples


def create_empathy_prompt(context, utterance):
    """创建同理心分类的prompt"""
    system_message = """你是一个客服质量评估专家，专门评估客服回复是否具有同理心。
同理心是指客服能够理解客户的感受和需求，并给予恰当的情感回应。

判断标准：
- 合规（有同理心）：客服回复体现了对客户的理解、关心，语气友好，有适当的安慰或道歉
- 不合规（缺乏同理心）：客服回复生硬、冷漠，只是机械地给出解决方案，没有考虑客户感受"""

    context_str = ""
    if context:
        context_str = "对话上下文：\n"
        for turn in context:
            role_name = "客户" if turn["role"] == "客户" else "客服"
            context_str += f"{role_name}: {turn['content']}\n"
        context_str += "\n"
    
    user_message = f"""{context_str}请判断以下客服回复是否具有同理心：

客服回复：{utterance}

判断要点：
1. 客服是否理解了客户的问题和情绪？
2. 客服的语气是否友好、有耐心？
3. 客服是否给予了适当的情感回应（如道歉、安慰）？
4. 客服是否只是机械地给出解决方案而忽视客户感受？

请只回答"合规"或"不合规"，不要有其他解释。

判断结果："""

    return system_message, user_message


def load_finetuned_model():
    """加载微调后的模型"""
    print("正在加载微调后的模型...")
    print(f"基础模型路径: {BASE_MODEL_PATH}")
    print(f"LoRA适配器路径: {ADAPTER_PATH}")

    from contextlib import redirect_stderr
    import io

    with redirect_stderr(io.StringIO()):
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    print("加载基础模型...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )

    print("加载LoRA适配器...")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)

    if not torch.cuda.is_available():
        model = model.to('cpu')

    model.eval()
    print("模型加载完成！\n")

    return tokenizer, model


def predict_batch(tokenizer, model, batch_samples):
    """对一个批次的样本进行预测"""
    if len(batch_samples) == 0:
        return []

    device = next(model.parameters()).device
    batch_inputs = []

    for sample in batch_samples:
        system_message, user_message = create_empathy_prompt(
            sample["context"], 
            sample["utterance"]
        )
        
        if hasattr(tokenizer, 'apply_chat_template'):
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
            input_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            input_text = f"System: {system_message}\n\nUser: {user_message}\n\nAssistant: "
        
        batch_inputs.append(input_text)

    inputs = tokenizer(
        batch_inputs,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True
        )

    responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    predictions = []
    for response in responses:
        if "判断结果：" in response:
            answer = response.split("判断结果：")[-1].strip()
        elif "Assistant:" in response:
            answer = response.split("Assistant:")[-1].strip()
        else:
            answer = response.strip()

        if "不合规" in answer:
            predictions.append("不合规")
        elif "合规" in answer:
            predictions.append("合规")
        else:
            predictions.append("合规")

    return predictions


def main():
    """主函数"""
    try:
        print("=" * 60)
        print("使用微调后的模型评估同理心分类")
        print("=" * 60)

        if torch.cuda.is_available():
            print(f"\n使用GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("\n使用CPU")

        # 检查LoRA适配器是否存在
        if not os.path.exists(ADAPTER_PATH):
            print(f"\n错误: LoRA适配器不存在: {ADAPTER_PATH}")
            print("请先运行 finetune_empathy.py 进行微调训练")
            return

        # 加载数据
        print(f"\n正在加载数据: {DATA_FILE}")
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"加载了 {len(data)} 个会话")

        # 抽取完整会话
        print(f"\n正在抽取 {SAMPLE_SESSIONS} 个完整会话...")
        random.seed(42)
        
        if len(data) > SAMPLE_SESSIONS:
            sampled_sessions = random.sample(data, SAMPLE_SESSIONS)
        else:
            sampled_sessions = data
        print(f"抽取了 {len(sampled_sessions)} 个会话")

        # 从抽取的会话中提取每一句客服回复
        print("\n正在提取客服回复样本...")
        sampled_data = extract_samples_from_data(sampled_sessions)
        print(f"共提取 {len(sampled_data)} 条客服回复")

        # 统计标签分布
        compliant_count = sum(1 for s in sampled_data if s["true_label"] == "合规")
        non_compliant_count = sum(1 for s in sampled_data if s["true_label"] == "不合规")
        print(f"  合规样本: {compliant_count}")
        print(f"  不合规样本: {non_compliant_count}")

        # 加载模型
        tokenizer, model = load_finetuned_model()

        # 批量预测
        print(f"\n开始预测（批处理大小: {BATCH_SIZE}）...")
        predictions = []
        true_labels = [s["true_label"] for s in sampled_data]
        num_batches = (len(sampled_data) + BATCH_SIZE - 1) // BATCH_SIZE

        start_time = time.time()

        pbar = tqdm(total=num_batches, desc="批量预测进度", ncols=100)
        for i in range(0, len(sampled_data), BATCH_SIZE):
            batch_samples = sampled_data[i:i + BATCH_SIZE]
            batch_preds = predict_batch(tokenizer, model, batch_samples)
            predictions.extend(batch_preds)
            pbar.update(1)
        pbar.close()

        end_time = time.time()
        total_time = end_time - start_time

        print(f"\n预测完成！")
        print(f"  总耗时: {total_time:.2f} 秒")
        print(f"  处理速度: {len(sampled_data)/total_time:.2f} 条/秒")

        # 计算评估指标
        print("\n" + "=" * 60)
        print("评估结果（微调后模型）")
        print("=" * 60)

        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted', zero_division=0
        )

        cm = confusion_matrix(true_labels, predictions, labels=["合规", "不合规"])

        print(f"\n数据统计:")
        print(f"  总样本数: {len(sampled_data)}")
        print(f"  真实合规: {true_labels.count('合规')}")
        print(f"  真实不合规: {true_labels.count('不合规')}")

        print(f"\n性能指标:")
        print(f"  准确率 (Accuracy):  {accuracy*100:.2f}%")
        print(f"  精确率 (Precision): {precision*100:.2f}%")
        print(f"  召回率 (Recall):    {recall*100:.2f}%")
        print(f"  F1分数 (F1-Score):  {f1*100:.2f}%")

        print(f"\n混淆矩阵:")
        print(f"                 预测合规    预测不合规")
        print(f"  真实合规:      {cm[0][0]:6d}      {cm[0][1]:6d}")
        print(f"  真实不合规:    {cm[1][0]:6d}      {cm[1][1]:6d}")

        # 各类别性能
        precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
            true_labels, predictions, labels=["合规", "不合规"], zero_division=0
        )

        print(f"\n各类别性能:")
        for i, label in enumerate(["合规", "不合规"]):
            print(f"\n  {label}:")
            print(f"    精确率: {precision_per_class[i]*100:.2f}%")
            print(f"    召回率: {recall_per_class[i]*100:.2f}%")
            print(f"    F1分数: {f1_per_class[i]*100:.2f}%")

        # 保存结果
        results = {
            "模型类型": "微调后LoRA模型",
            "总体统计": {
                "总样本数": len(sampled_data),
                "真实合规数": true_labels.count('合规'),
                "真实不合规数": true_labels.count('不合规'),
                "预测合规数": predictions.count('合规'),
                "预测不合规数": predictions.count('不合规')
            },
            "性能指标": {
                "准确率": float(accuracy),
                "精确率": float(precision),
                "召回率": float(recall),
                "F1分数": float(f1)
            },
            "混淆矩阵": {
                "真实合规_预测合规": int(cm[0][0]),
                "真实合规_预测不合规": int(cm[0][1]),
                "真实不合规_预测合规": int(cm[1][0]),
                "真实不合规_预测不合规": int(cm[1][1])
            },
            "预测详情": [
                {
                    "会话ID": sample["session_id"],
                    "上下文": [f"{t['role']}: {t['content']}" for t in sample["context"]],
                    "客服回复": sample["utterance"],
                    "真实标签": sample["true_label"],
                    "预测标签": pred,
                    "正确": sample["true_label"] == pred
                }
                for sample, pred in zip(sampled_data, predictions)
            ]
        }

        with open(OUTPUT_RESULTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n详细结果已保存至: {OUTPUT_RESULTS_FILE}")

        # 错误分析
        errors = [
            {
                "会话ID": sample["session_id"],
                "上下文": [f"{t['role']}: {t['content']}" for t in sample["context"]],
                "客服回复": sample["utterance"],
                "真实标签": sample["true_label"],
                "预测标签": pred
            }
            for sample, pred in zip(sampled_data, predictions)
            if sample["true_label"] != pred
        ]

        print(f"\n错误总数: {len(errors)} ({len(errors)/len(sampled_data)*100:.2f}%)")

        with open(OUTPUT_ERRORS_FILE, 'w', encoding='utf-8') as f:
            json.dump(errors, f, ensure_ascii=False, indent=2)
        print(f"错误案例已保存至: {OUTPUT_ERRORS_FILE}")

        print("\n" + "=" * 60)
        print("评估完成！")
        print("=" * 60)

    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
