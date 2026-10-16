#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对预测结果进行分层抽样，计算模型准确率
"""

import json
import random
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# 配置
RESULTS_FILE = "/data/xinwang/finetune_model/empathy/finetuned_predict_results.json"
SAMPLE_SIZE_PER_CLASS =300   # 每个类别抽取的样本数
RANDOM_SEED = 42


def stratified_sampling(predictions, sample_size_per_class):
    """
    分层抽样：按真实标签分层，每层抽取相同数量的样本
    """
    # 按真实标签分组
    groups = defaultdict(list)
    for item in predictions:
        groups[item["真实标签"]].append(item)
    
    print(f"\n原始数据分布:")
    for label, items in groups.items():
        print(f"  {label}: {len(items)} 条")
    
    # 分层抽样
    sampled = []
    random.seed(RANDOM_SEED)
    
    for label, items in groups.items():
        if len(items) >= sample_size_per_class:
            sampled_items = random.sample(items, sample_size_per_class)
        else:
            # 如果某类样本不足，全部取出
            sampled_items = items
            print(f"  警告: {label} 类样本不足 {sample_size_per_class}，仅取 {len(items)} 条")
        sampled.extend(sampled_items)
    
    return sampled


def calculate_metrics(sampled_data):
    """计算评估指标"""
    true_labels = [item["真实标签"] for item in sampled_data]
    pred_labels = [item["预测标签"] for item in sampled_data]
    
    accuracy = accuracy_score(true_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average='weighted', zero_division=0
    )
    
    cm = confusion_matrix(true_labels, pred_labels, labels=["合规", "不合规"])
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "true_labels": true_labels,
        "pred_labels": pred_labels
    }


def main():
    print("=" * 60)
    print("分层抽样计算模型准确率")
    print("=" * 60)
    
    # 加载预测结果
    print(f"\n加载预测结果: {RESULTS_FILE}")
    with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    predictions = results["预测详情"]
    print(f"总样本数: {len(predictions)}")
    
    # 原始准确率
    print(f"\n原始整体准确率: {results['性能指标']['准确率']*100:.2f}%")
    
    # 分层抽样
    print(f"\n进行分层抽样 (每类 {SAMPLE_SIZE_PER_CLASS} 条)...")
    sampled_data = stratified_sampling(predictions, SAMPLE_SIZE_PER_CLASS)
    
    print(f"\n抽样后数据分布:")
    sampled_compliant = sum(1 for item in sampled_data if item["真实标签"] == "合规")
    sampled_non_compliant = sum(1 for item in sampled_data if item["真实标签"] == "不合规")
    print(f"  合规: {sampled_compliant} 条")
    print(f"  不合规: {sampled_non_compliant} 条")
    print(f"  总计: {len(sampled_data)} 条")
    
    # 计算指标
    metrics = calculate_metrics(sampled_data)
    
    print("\n" + "=" * 60)
    print("分层抽样后的评估结果")
    print("=" * 60)
    
    print(f"\n性能指标:")
    print(f"  准确率 (Accuracy):  {metrics['accuracy']*100:.2f}%")
    print(f"  精确率 (Precision): {metrics['precision']*100:.2f}%")
    print(f"  召回率 (Recall):    {metrics['recall']*100:.2f}%")
    print(f"  F1分数 (F1-Score):  {metrics['f1']*100:.2f}%")
    
    cm = metrics['confusion_matrix']
    print(f"\n混淆矩阵:")
    print(f"                 预测合规    预测不合规")
    print(f"  真实合规:      {cm[0][0]:6d}      {cm[0][1]:6d}")
    print(f"  真实不合规:    {cm[1][0]:6d}      {cm[1][1]:6d}")
    
    # 各类别准确率
    print(f"\n各类别准确率:")
    for label in ["合规", "不合规"]:
        label_samples = [item for item in sampled_data if item["真实标签"] == label]
        correct = sum(1 for item in label_samples if item["正确"])
        acc = correct / len(label_samples) if label_samples else 0
        print(f"  {label}: {correct}/{len(label_samples)} = {acc*100:.2f}%")
    
    # 保存分层抽样结果
    output_file = RESULTS_FILE.replace(".json", "_stratified.json")
    stratified_results = {
        "抽样方法": "分层抽样",
        "每类抽样数": SAMPLE_SIZE_PER_CLASS,
        "随机种子": RANDOM_SEED,
        "抽样后统计": {
            "总样本数": len(sampled_data),
            "合规样本数": sampled_compliant,
            "不合规样本数": sampled_non_compliant
        },
        "性能指标": {
            "准确率": float(metrics['accuracy']),
            "精确率": float(metrics['precision']),
            "召回率": float(metrics['recall']),
            "F1分数": float(metrics['f1'])
        },
        "混淆矩阵": {
            "真实合规_预测合规": int(cm[0][0]),
            "真实合规_预测不合规": int(cm[0][1]),
            "真实不合规_预测合规": int(cm[1][0]),
            "真实不合规_预测不合规": int(cm[1][1])
        },
        "各类别准确率": {
            label: sum(1 for item in sampled_data if item["真实标签"] == label and item["正确"]) / 
                   sum(1 for item in sampled_data if item["真实标签"] == label)
            for label in ["合规", "不合规"]
        },
        "抽样详情": sampled_data
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(stratified_results, f, ensure_ascii=False, indent=2)
    print(f"\n分层抽样结果已保存至: {output_file}")
    
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
