#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用基础模型对客服对话进行同理心分类预测
结合上下文进行判断，并计算模型预测准确率
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import time
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import random
import warnings
import logging
import os
import re

# 禁用警告信息
warnings.filterwarnings('ignore')
logging.getLogger('transformers').setLevel(logging.ERROR)
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# ==================== 配置参数 ====================
# 模型路径 - 使用当前目录的基础模型
BASE_MODEL_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据文件
DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "所有会话_同理心数据.json")

# 预测参数
BATCH_SIZE = 8  # 批处理大小，因为包含上下文所以适当减小
MAX_LENGTH = 1024  # 最大输入长度，增加以容纳上下文
CONTEXT_TURNS = 5  # 使用前N轮对话作为上下文
SAMPLES_PER_CLASS = 300  # 每个类别抽取的样本数

# 输出文件
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_RESULTS_FILE = os.path.join(OUTPUT_DIR, "empathy_predict_results.json")
OUTPUT_ERRORS_FILE = os.path.join(OUTPUT_DIR, "empathy_predict_errors.json")
# ==================================================


def load_base_model():
    """加载基础模型"""
    print("正在加载基础模型...")
    print(f"基础模型路径: {BASE_MODEL_PATH}")

    from contextlib import redirect_stderr
    import io

    with redirect_stderr(io.StringIO()):
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    print("加载模型权重...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )

    if not torch.cuda.is_available():
        model = model.to('cpu')

    model.eval()
    print("模型加载完成！\n")

    return tokenizer, model


def parse_dialogue_line(line):
    """
    解析对话行，提取角色、内容和标签
    格式: "角色: 内容 标签"
    """
    # 匹配格式: "客户: xxx 合规" 或 "客服: xxx 不合规"
    match = re.match(r'^(客户|客服): (.+?) (合规|不合规)$', line)
    if match:
        role = match.group(1)
        content = match.group(2)
        label = match.group(3)
        return role, content, label
    return None, None, None


def extract_samples_from_data(data):
    """
    从对话数据中提取需要预测的样本
    只提取客服的回复，并保留上下文
    """
    samples = []
    
    for session in data:
        session_id = session.get("会话编号", "")
        dialogue = session.get("对话内容", [])
        
        # 解析所有对话
        parsed_dialogue = []
        for line in dialogue:
            role, content, label = parse_dialogue_line(line)
            if role and content:
                parsed_dialogue.append({
                    "role": role,
                    "content": content,
                    "label": label
                })
        
        # 提取客服回复及其上下文
        for i, turn in enumerate(parsed_dialogue):
            if turn["role"] == "客服":
                # 获取上下文（前N轮对话）
                start_idx = max(0, i - CONTEXT_TURNS)
                context = parsed_dialogue[start_idx:i]
                
                samples.append({
                    "session_id": session_id,
                    "context": context,
                    "agent_response": turn["content"],
                    "true_label": turn["label"],
                    "turn_index": i
                })
    
    return samples


def create_empathy_prompt(context, agent_response):
    """创建同理心分类的prompt，结合上下文"""
    system_message = """你是一个客服质量评估专家，专门评估客服回复是否具有同理心。
同理心是指客服能够理解客户的感受和需求，并给予恰当的情感回应。

判断标准：
- 合规（有同理心）：客服回复体现了对客户的理解、关心，语气友好，有适当的安慰或道歉
- 不合规（缺乏同理心）：客服回复生硬、冷漠，只是机械地给出解决方案，没有考虑客户感受"""

    # 构建上下文字符串
    context_str = ""
    if context:
        context_str = "对话上下文：\n"
        for turn in context:
            role_name = "客户" if turn["role"] == "客户" else "客服"
            context_str += f"{role_name}: {turn['content']}\n"
        context_str += "\n"
    
    user_message = f"""{context_str}请判断以下客服回复是否具有同理心：

客服回复：{agent_response}

判断要点：
1. 客服是否理解了客户的问题和情绪？
2. 客服的语气是否友好、有耐心？
3. 客服是否给予了适当的情感回应（如道歉、安慰）？
4. 客服是否只是机械地给出解决方案而忽视客户感受？

请只回答"合规"或"不合规"，不要有其他解释。

判断结果："""

    return system_message, user_message


def predict_batch(tokenizer, model, batch_samples):
    """对一个批次的样本进行预测"""
    if len(batch_samples) == 0:
        return []

    device = next(model.parameters()).device
    batch_inputs = []

    for sample in batch_samples:
        system_message, user_message = create_empathy_prompt(
            sample["context"], 
            sample["agent_response"]
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

    # 批量编码
    inputs = tokenizer(
        batch_inputs,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 批量生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True
        )

    # 批量解码
    responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # 提取答案
    predictions = []
    for response in responses:
        if "判断结果：" in response:
            answer = response.split("判断结果：")[-1].strip()
        elif "Assistant:" in response:
            answer = response.split("Assistant:")[-1].strip()
        else:
            answer = response.strip()

        # 判断
        if "不合规" in answer:
            predictions.append("不合规")
        elif "合规" in answer:
            predictions.append("合规")
        else:
            # 默认判断为合规
            predictions.append("合规")

    return predictions


def main():
    """主函数"""
    try:
        print("=" * 60)
        print("使用基础模型预测客服同理心")
        print("=" * 60)

        # 检查CUDA
        if torch.cuda.is_available():
            print(f"\n使用GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            print("\n使用CPU（速度较慢）")

        # 加载数据
        print(f"\n正在加载数据: {DATA_FILE}")
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"加载了 {len(data)} 个会话")

        # 提取样本
        print("\n正在提取客服回复样本...")
        all_samples = extract_samples_from_data(data)
        print(f"共提取 {len(all_samples)} 条客服回复")

        # 统计标签分布
        label_counts = {"合规": 0, "不合规": 0}
        for sample in all_samples:
            label_counts[sample["true_label"]] = label_counts.get(sample["true_label"], 0) + 1
        print(f"标签分布: 合规={label_counts.get('合规', 0)}, 不合规={label_counts.get('不合规', 0)}")

        # 按类别分组并抽样
        print(f"\n正在从合规和不合规各抽取 {SAMPLES_PER_CLASS} 个样例...")
        random.seed(42)

        compliant_samples = [s for s in all_samples if s["true_label"] == "合规"]
        non_compliant_samples = [s for s in all_samples if s["true_label"] == "不合规"]

        print(f"  合规样本总数: {len(compliant_samples)}")
        print(f"  不合规样本总数: {len(non_compliant_samples)}")

        # 抽样
        if len(compliant_samples) > SAMPLES_PER_CLASS:
            compliant_samples = random.sample(compliant_samples, SAMPLES_PER_CLASS)
        if len(non_compliant_samples) > SAMPLES_PER_CLASS:
            non_compliant_samples = random.sample(non_compliant_samples, SAMPLES_PER_CLASS)

        # 合并抽样后的数据
        sampled_data = compliant_samples + non_compliant_samples
        random.shuffle(sampled_data)

        print(f"抽样完成: 合规 {len(compliant_samples)} 个，不合规 {len(non_compliant_samples)} 个，总计 {len(sampled_data)} 条")

        # 加载模型
        tokenizer, model = load_base_model()

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
        print(f"  平均每条: {total_time/len(sampled_data):.3f} 秒")
        print(f"  处理速度: {len(sampled_data)/total_time:.2f} 条/秒")

        # 计算评估指标
        print("\n" + "=" * 60)
        print("评估结果")
        print("=" * 60)

        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted', zero_division=0
        )

        # 混淆矩阵
        cm = confusion_matrix(true_labels, predictions, labels=["合规", "不合规"])

        print(f"\n数据统计:")
        print(f"  总样本数: {len(sampled_data)}")
        print(f"  真实合规: {true_labels.count('合规')} ({true_labels.count('合规')/len(true_labels)*100:.2f}%)")
        print(f"  真实不合规: {true_labels.count('不合规')} ({true_labels.count('不合规')/len(true_labels)*100:.2f}%)")

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
        print(f"\n各类别性能:")
        precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
            true_labels, predictions, labels=["合规", "不合规"], zero_division=0
        )

        for i, label in enumerate(["合规", "不合规"]):
            print(f"\n  {label}:")
            print(f"    精确率: {precision_per_class[i]*100:.2f}%")
            print(f"    召回率: {recall_per_class[i]*100:.2f}%")
            print(f"    F1分数: {f1_per_class[i]*100:.2f}%")
            print(f"    样本数: {support_per_class[i]}")

        # 保存详细结果
        results = {
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
            "各类别性能": {
                "合规": {
                    "精确率": float(precision_per_class[0]),
                    "召回率": float(recall_per_class[0]),
                    "F1分数": float(f1_per_class[0]),
                    "样本数": int(support_per_class[0])
                },
                "不合规": {
                    "精确率": float(precision_per_class[1]),
                    "召回率": float(recall_per_class[1]),
                    "F1分数": float(f1_per_class[1]),
                    "样本数": int(support_per_class[1])
                }
            },
            "预测详情": [
                {
                    "会话ID": sample["session_id"],
                    "上下文": [f"{t['role']}: {t['content']}" for t in sample["context"]],
                    "客服回复": sample["agent_response"],
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

        # 错误案例分析
        print("\n" + "=" * 60)
        print("错误案例分析")
        print("=" * 60)

        errors = [
            {
                "会话ID": sample["session_id"],
                "上下文": [f"{t['role']}: {t['content']}" for t in sample["context"]],
                "客服回复": sample["agent_response"],
                "真实标签": sample["true_label"],
                "预测标签": pred
            }
            for sample, pred in zip(sampled_data, predictions)
            if sample["true_label"] != pred
        ]

        print(f"\n错误总数: {len(errors)} ({len(errors)/len(sampled_data)*100:.2f}%)")

        # 统计错误类型
        false_positive = sum(1 for e in errors if e["真实标签"] == "合规" and e["预测标签"] == "不合规")
        false_negative = sum(1 for e in errors if e["真实标签"] == "不合规" and e["预测标签"] == "合规")

        print(f"\n错误类型:")
        if len(errors) > 0:
            print(f"  假阳性（合规被误判为不合规）: {false_positive} ({false_positive/len(errors)*100:.2f}%)")
            print(f"  假阴性（不合规被误判为合规）: {false_negative} ({false_negative/len(errors)*100:.2f}%)")
        else:
            print("  无错误")

        # 显示部分错误案例
        if len(errors) > 0:
            print(f"\n前5个错误案例:")
            for i, error in enumerate(errors[:5], 1):
                print(f"\n  [{i}] 客服回复: {error['客服回复'][:60]}...")
                print(f"      真实: {error['真实标签']} | 预测: {error['预测标签']}")
                if error['上下文']:
                    print(f"      上下文: {error['上下文'][-1][:50]}...")

        with open(OUTPUT_ERRORS_FILE, 'w', encoding='utf-8') as f:
            json.dump(errors, f, ensure_ascii=False, indent=2)
        print(f"\n错误案例已保存至: {OUTPUT_ERRORS_FILE}")

        print("\n" + "=" * 60)
        print("评估完成！")
        print("=" * 60)

    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
