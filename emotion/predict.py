#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用基础模型预测对话数据并计算总准确率
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import time
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import warnings
import logging
import sys
import os
import random

# 禁用警告信息
warnings.filterwarnings('ignore')
logging.getLogger('transformers').setLevel(logging.ERROR)

# 禁用transformers的进度条
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# ==================== 配置参数 ====================
# 模型路径，此处需要修改为本地模型路径
BASE_MODEL_PATH = "/data/xinwang/finetune_model"

# 数据文件，此处需要修改为本地数据文件路径
ALL_DATA_FILES = [
    "/data/xinwang/finetune_model/emotion/平衡后的客户对话数据.json",
]

# 预测参数
BATCH_SIZE = 16  # 批处理大小，根据GPU内存调整
MAX_LENGTH = 512  # 最大输入长度
SAMPLES_PER_CLASS = 500  # 每个类别抽取的样本数（正面/负面）

# 输出文件，此处需要修改为本地输出文件路径
OUTPUT_RESULTS_FILE = "/data/xinwang/finetune_model/emotion/predict_results.json"
OUTPUT_ERRORS_FILE = "/data/xinwang/finetune_model/emotion/predict_errors.json"
# ==================================================

def load_base_model():
    """加载基础模型"""
    print("正在加载基础模型...")
    print(f"基础模型路径: {BASE_MODEL_PATH}")

    # 临时重定向stderr以隐藏加载进度条
    from contextlib import redirect_stderr
    import io

    # 加载tokenizer（静默）
    with redirect_stderr(io.StringIO()):
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)

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

def create_sentiment_prompt(text):
    """创建情感分类的prompt（与训练时一致）"""
    system_message = "你是一个情感分析专家，专门识别客户对话中的负面情绪。只有当对话中包含明显的生气、不耐烦、愤怒等负面语气词时才判断为负面，否则默认为正面。"
    
    user_message = f"""请判断下面这句话是否包含明显的生气或不耐烦的语气。

判断标准：
- 如果句子中有明显的生气、不耐烦、愤怒、抱怨等负面语气词（如：烦死了、气死了、什么破、太差了、垃圾、投诉、退款等），则判断为"负面"
- 否则默认判断为"正面"

句子：{text}

请只回答"正面"或"负面"，不要有其他解释。

情感："""
    
    return system_message, user_message

def predict_sentiment_single_batch(tokenizer, model, batch_texts):
    """预测单个批次的情感（不包含循环和进度条）"""
    device = next(model.parameters()).device
    batch_inputs = []
    
    for text in batch_texts:
        system_message, user_message = create_sentiment_prompt(text)
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        if hasattr(tokenizer, 'apply_chat_template'):
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
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True
        )
    
    # 批量解码
    responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    # 提取答案
    predictions = []
    for i, response in enumerate(responses):
        # 特殊规则：问候语判定为正面
        text = batch_texts[i]
        greeting_keywords = ["在吗", "在不在", "在么", "在嘛", "有人吗", "有人么", "有人在吗", "有人在么"]
        if any(keyword in text for keyword in greeting_keywords):
            predictions.append("正面")
            continue
        
        if "情感：" in response:
            answer = response.split("情感：")[-1].strip()
        elif "Assistant:" in response:
            answer = response.split("Assistant:")[-1].strip()
        else:
            answer = response.strip()
        
        # 判断
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

def load_all_data():
    """加载所有对话数据"""
    print("\n正在加载所有对话数据...")
    all_data = []
    file_stats = []
    
    for data_file in ALL_DATA_FILES:
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            file_name = data_file.split('/')[-1]
            print(f"  - {file_name}: {len(data)} 条")
            
            all_data.extend(data)
            file_stats.append({
                "file": file_name,
                "count": len(data)
            })
            
        except Exception as e:
            print(f"  - 加载 {data_file} 失败: {str(e)}")
            continue
    
    print(f"\n总共加载: {len(all_data)} 条数据")
    
    return all_data, file_stats

def predict_all_data(tokenizer, model):
    """预测所有数据并计算准确率"""
    # 加载所有数据
    all_data, file_stats = load_all_data()
    
    if len(all_data) == 0:
        print("错误: 没有加载到任何数据！")
        return
    
    # 提取文本和标签
    texts = []
    true_labels = []
    
    print("\n正在提取文本和标签...")
    for item in all_data:
        text = item["语句内容"]
        true_label = item["情绪积极"]
        texts.append(text)
        true_labels.append(true_label)
    
    print(f"提取完成: {len(texts)} 条数据")
    
    # 按类别分组并抽样
    print(f"\n正在从正面和负面各抽取 {SAMPLES_PER_CLASS} 个样例...")
    random.seed(42)  # 设置随机种子以保证可重复性
    
    # 按标签分组
    pos_samples = []
    neg_samples = []
    for text, label in zip(texts, true_labels):
        if label == "正面":
            pos_samples.append((text, label))
        elif label == "负面":
            neg_samples.append((text, label))
    
    print(f"  正面样本总数: {len(pos_samples)}")
    print(f"  负面样本总数: {len(neg_samples)}")
    
    # 抽样
    if len(pos_samples) > SAMPLES_PER_CLASS:
        pos_samples = random.sample(pos_samples, SAMPLES_PER_CLASS)
    if len(neg_samples) > SAMPLES_PER_CLASS:
        neg_samples = random.sample(neg_samples, SAMPLES_PER_CLASS)
    
    # 合并抽样后的数据
    sampled_data = pos_samples + neg_samples
    random.shuffle(sampled_data)  # 打乱顺序
    
    texts = [item[0] for item in sampled_data]
    true_labels = [item[1] for item in sampled_data]
    
    print(f"抽样完成: 正面 {len(pos_samples)} 个，负面 {len(neg_samples)} 个，总计 {len(texts)} 条数据")
    
    # 批量预测
    print(f"\n开始预测（批处理大小: {BATCH_SIZE}）...")
    predictions = []
    num_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    
    start_time = time.time()
    
    # 使用一个进度条，类似evaluate_finetuned_model的实现
    pbar = tqdm(total=num_batches, desc="批量预测进度", ncols=100, position=0, leave=True)
    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i:i+BATCH_SIZE]
        batch_preds = predict_sentiment_single_batch(tokenizer, model, batch_texts)
        predictions.extend(batch_preds)
        pbar.update(1)  
    pbar.close()
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / len(texts)
    
    print(f"\n预测完成！")
    print(f"  总耗时: {total_time:.2f} 秒")
    print(f"  平均每条: {avg_time:.3f} 秒")
    print(f"  处理速度: {len(texts)/total_time:.2f} 条/秒")
    
    # 计算总体准确率
    print("\n" + "="*60)
    print("总体评估结果")
    print("="*60)
    
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='weighted', zero_division=0
    )
    
    # 混淆矩阵
    cm = confusion_matrix(true_labels, predictions, labels=["正面", "负面"])
    
    print(f"\n数据统计:")
    print(f"  总样本数: {len(texts)}")
    print(f"  真实正面: {true_labels.count('正面')} ({true_labels.count('正面')/len(true_labels)*100:.2f}%)")
    print(f"  真实负面: {true_labels.count('负面')} ({true_labels.count('负面')/len(true_labels)*100:.2f}%)")
    
    print(f"\n性能指标:")
    print(f"  准确率 (Accuracy):  {accuracy*100:.2f}%")
    print(f"  精确率 (Precision): {precision*100:.2f}%")
    print(f"  召回率 (Recall):    {recall*100:.2f}%")
    print(f"  F1分数 (F1-Score):  {f1*100:.2f}%")
    
    print(f"\n混淆矩阵:")
    print(f"                 预测正面    预测负面")
    print(f"  真实正面:      {cm[0][0]:6d}      {cm[0][1]:6d}")
    print(f"  真实负面:      {cm[1][0]:6d}      {cm[1][1]:6d}")
    
    # 计算各类别的准确率
    print(f"\n各类别性能:")
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        true_labels, predictions, labels=["正面", "负面"], zero_division=0
    )
    
    for i, label in enumerate(["正面", "负面"]):
        print(f"\n  {label}:")
        print(f"    精确率: {precision_per_class[i]*100:.2f}%")
        print(f"    召回率: {recall_per_class[i]*100:.2f}%")
        print(f"    F1分数: {f1_per_class[i]*100:.2f}%")
        print(f"    样本数: {support_per_class[i]}")
    
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
        "各类别性能": {
            "正面": {
                "精确率": float(precision_per_class[0]),
                "召回率": float(recall_per_class[0]),
                "F1分数": float(f1_per_class[0]),
                "样本数": int(support_per_class[0])
            },
            "负面": {
                "精确率": float(precision_per_class[1]),
                "召回率": float(recall_per_class[1]),
                "F1分数": float(f1_per_class[1]),
                "样本数": int(support_per_class[1])
            }
        },
        "文件统计": file_stats,
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
    
    # 保存结果
    with open(OUTPUT_RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n详细结果已保存至: {OUTPUT_RESULTS_FILE}")
    
    # 分析错误案例
    print("\n" + "="*60)
    print("错误案例分析")
    print("="*60)
    
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
    
    # 保存错误案例
    with open(OUTPUT_ERRORS_FILE, 'w', encoding='utf-8') as f:
        json.dump(errors, f, ensure_ascii=False, indent=2)
    
    print(f"\n错误案例已保存至: {OUTPUT_ERRORS_FILE}")
    
    print("\n" + "="*60)
    print("评估完成！")
    print("="*60)
    
    return results

def main():
    """主函数"""
    try:
        print("="*60)
        print("使用基础模型预测所有对话数据")
        print("="*60)
        
        # 检查CUDA
        if torch.cuda.is_available():
            print(f"\n使用GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            print("\n使用CPU（速度较慢）")
        
        # 加载模型
        tokenizer, model = load_base_model()
        
        # 预测所有数据
        results = predict_all_data(tokenizer, model)
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

