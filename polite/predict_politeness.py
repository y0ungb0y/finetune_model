#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用基础模型对对话数据进行礼貌性分类，并与标签对比计算准确率
"""
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# 配置
BASE_MODEL_PATH = "/data/xinwang/finetune_model"  # 基础模型路径
DATA_PATH = "/data/xinwang/finetune_model/polite/对话内容礼貌性.json"  # 数据文件路径
OUTPUT_PATH = "/data/xinwang/finetune_model/polite/预测结果.json"  # 输出文件路径

BATCH_SIZE = 8
MAX_LENGTH = 512
SAMPLE_SIZE = 1000  # 设置为 None 则使用全部数据，设置数字则采样


def create_prompt(text: str) -> str:
    """构造礼貌性分类的 Prompt"""
    system_message = (
        "你是一个客服对话质量评估专家，专门判断客服语句的礼貌性。"
        "合规的语句应该使用礼貌用语、语气友好、表达清晰。"
        "不合规的语句可能语气生硬、缺乏礼貌用语、表达不够友好。"
    )
    user_message = (
        "请判断下面这句客服语句的礼貌性是否合规。\n\n"
        "判断标准：\n"
        "- 如果语句使用了礼貌用语（如：您好、请、麻烦、感谢、抱歉、亲等），语气友好，则判断为\"合规\"\n"
        "- 如果语句语气生硬、缺乏礼貌用语、表达不够友好，则判断为\"不合规\"\n\n"
        f"语句：{text}\n\n"
        "请只回答\"合规\"或\"不合规\"，不要有其他解释。\n\n"
        "礼貌性："
    )
    return f"System: {system_message}\n\nUser: {user_message}"


def load_model():
    """加载基础模型"""
    print("\n加载基础模型...")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    model.eval()
    print("模型加载完成！\n")
    return tokenizer, model


def predict_batch(tokenizer, model, texts):
    """对一个批次的文本进行预测，返回"合规"/"不合规"标签列表"""
    if len(texts) == 0:
        return []

    device = next(model.parameters()).device

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
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    predictions = []
    for response in responses:
        if "礼貌性：" in response:
            answer = response.split("礼貌性：")[-1].strip()
        else:
            answer = response.strip()

        if "不合规" in answer:
            predictions.append("不合规")
        elif "合规" in answer:
            predictions.append("合规")
        else:
            # 默认判断
            predictions.append("合规")

    return predictions


def main():
    """主函数"""
    print("=" * 60)
    print("礼貌性分类预测与准确率评估")
    print("=" * 60)

    # 加载数据
    print(f"\n加载数据: {DATA_PATH}")
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"总样本数: {len(data)}")

    # 采样（如果设置了 SAMPLE_SIZE）
    if SAMPLE_SIZE and SAMPLE_SIZE < len(data):
        import random
        random.seed(42)
        data = random.sample(data, SAMPLE_SIZE)
        print(f"采样后样本数: {len(data)}")

    # 加载模型
    tokenizer, model = load_model()

    # 提取文本和标签
    texts = [item['语句内容'] for item in data]
    labels = [item['礼貌性'] for item in data]
    sources = [item.get('样本来源', '未知') for item in data]

    # 批量预测
    print(f"\n开始预测 (batch_size={BATCH_SIZE})...")
    predictions = []
    num_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in tqdm(range(0, len(texts), BATCH_SIZE), total=num_batches, desc="预测进度"):
        batch_texts = texts[i:i + BATCH_SIZE]
        batch_preds = predict_batch(tokenizer, model, batch_texts)
        predictions.extend(batch_preds)

    # 计算准确率
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)

    correct = sum(1 for p, l in zip(predictions, labels) if p == l)
    accuracy = correct / len(labels) * 100

    print(f"\n总样本数: {len(labels)}")
    print(f"正确预测数: {correct}")
    print(f"准确率: {accuracy:.2f}%")

    # 分类别统计
    label_stats = {"合规": {"total": 0, "correct": 0}, "不合规": {"total": 0, "correct": 0}}
    source_stats = {"原始": {"total": 0, "correct": 0}, "生成": {"total": 0, "correct": 0}}

    for pred, label, source in zip(predictions, labels, sources):
        if label in label_stats:
            label_stats[label]["total"] += 1
            if pred == label:
                label_stats[label]["correct"] += 1
        if source in source_stats:
            source_stats[source]["total"] += 1
            if pred == label:
                source_stats[source]["correct"] += 1

    print("\n按标签分类准确率:")
    for label, stats in label_stats.items():
        if stats["total"] > 0:
            acc = stats["correct"] / stats["total"] * 100
            print(f"  {label}: {stats['correct']}/{stats['total']} = {acc:.2f}%")

    print("\n按来源分类准确率:")
    for source, stats in source_stats.items():
        if stats["total"] > 0:
            acc = stats["correct"] / stats["total"] * 100
            print(f"  {source}: {stats['correct']}/{stats['total']} = {acc:.2f}%")

    # 收集错误样本
    errors = []
    for i, (pred, label) in enumerate(zip(predictions, labels)):
        if pred != label:
            errors.append({
                "语句内容": texts[i],
                "真实标签": label,
                "预测标签": pred,
                "样本来源": sources[i]
            })

    print(f"\n错误样本数: {len(errors)}")

    # 保存结果
    results = {
        "统计": {
            "总样本数": len(labels),
            "正确预测数": correct,
            "准确率": f"{accuracy:.2f}%",
            "标签分类准确率": {
                label: {
                    "总数": stats["total"],
                    "正确数": stats["correct"],
                    "准确率": f"{stats['correct'] / stats['total'] * 100:.2f}%" if stats["total"] > 0 else "N/A"
                }
                for label, stats in label_stats.items()
            },
            "来源分类准确率": {
                source: {
                    "总数": stats["total"],
                    "正确数": stats["correct"],
                    "准确率": f"{stats['correct'] / stats['total'] * 100:.2f}%" if stats["total"] > 0 else "N/A"
                }
                for source, stats in source_stats.items()
            }
        },
        "预测详情": [
            {
                "语句内容": texts[i],
                "真实标签": labels[i],
                "预测标签": predictions[i],
                "样本来源": sources[i],
                "是否正确": predictions[i] == labels[i]
            }
            for i in range(len(texts))
        ],
        "错误样本": errors[:100]  # 只保存前100个错误样本
    }

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存到: {OUTPUT_PATH}")
    print("\n🎉 礼貌性分类预测完成！")


if __name__ == '__main__':
    main()
