#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""利用现有对话数据对基础模型进行LoRA微调，以提升情感分类准确率。"""

import os
import json
import random
from dataclasses import dataclass
from typing import List, Dict

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from transformers.data.data_collator import DataCollatorMixin

try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)

# ==================== 基础配置 ====================
BASE_MODEL_PATH = "/data/xinwang/finetune_model"
DATA_FILE = "/data/xinwang/finetune_model/平衡后的客户对话数据.json"
ADAPTER_OUTPUT_DIR = "/data/xinwang/finetune_model/emotion/lora_adapter"
MERGED_MODEL_DIR = "/data/xinwang/finetune_model/outputs/merged_model"

TRAIN_RATIO = 0.70  # 训练集比例
VAL_RATIO = 0.15   # 验证集比例
TEST_RATIO = 0.15  # 测试集比例
MAX_SEQ_LENGTH = 512
SEED = 42

# 训练超参（可根据显存情况调整）
MICRO_BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 4
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01
SAVE_STEPS = 1000  # 减少保存频率
LOG_STEPS = 100
EVAL_STEPS = 500   # 减少评估频率

# LoRA 配置
LORA_R = 64
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# 是否使用4bit量化加载（QLoRA）。在24GB显存上建议保持开启。
# 禁用bitsandbytes后默认走标准LoRA；如需QLoRA可改为True且安装对应该版本torch/bitsandbytes
USE_QLORA = False
SAVE_MERGED_MODEL = False  # 训练完成后将LoRA权重合并成可直接推理的完整模型
# ==================================================\n

def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_dialogue_data(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 仅保留包含标签的条目
    cleaned = [item for item in data if item.get("语句内容") and item.get("情绪积极") in {"正面", "负面"}]
    print(f"加载数据: {len(cleaned)} 条（原始 {len(data)} 条）")
    return cleaned


def create_prompt(text: str) -> str:
    """与预测阶段保持一致的提示语。"""
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


def tokenize_samples(dataset: Dataset, tokenizer: AutoTokenizer) -> Dataset:
    """将原始数据集转换为语言模型可训练的格式。"""

    def preprocess_function(examples):
        input_ids_list = []
        attention_masks = []
        labels_list = []

        for text, sentiment in zip(examples["语句内容"], examples["情绪积极"]):
            prompt = create_prompt(text)
            full_text = prompt + sentiment

            tokenized_full = tokenizer(
                full_text,
                max_length=MAX_SEQ_LENGTH,
                truncation=True,
                padding=False,
            )
            tokenized_prompt = tokenizer(
                prompt,
                max_length=MAX_SEQ_LENGTH,
                truncation=True,
                padding=False,
            )

            input_ids = tokenized_full["input_ids"]
            attention_mask = tokenized_full["attention_mask"]
            
            # 确保 labels 与 input_ids 长度完全一致
            labels = [-100] * len(input_ids)
            prompt_len = min(len(tokenized_prompt["input_ids"]), len(input_ids))
            
            # 只对答案部分计算损失
            for idx in range(prompt_len, len(input_ids)):
                labels[idx] = input_ids[idx]
            
            # 确保所有序列都被截断到相同的最大长度
            if len(input_ids) > MAX_SEQ_LENGTH:
                input_ids = input_ids[:MAX_SEQ_LENGTH]
                attention_mask = attention_mask[:MAX_SEQ_LENGTH]
                labels = labels[:MAX_SEQ_LENGTH]

            input_ids_list.append(input_ids)
            attention_masks.append(attention_mask)
            labels_list.append(labels)

        return {
            "input_ids": input_ids_list,
            "attention_mask": attention_masks,
            "labels": labels_list,
        }

    processed = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset",
    )
    return processed


def prepare_datasets(tokenizer: AutoTokenizer):
    data = load_dialogue_data(DATA_FILE)
    
    # 按标签分组
    positive_samples = [item for item in data if item["情绪积极"] == "正面"]
    negative_samples = [item for item in data if item["情绪积极"] == "负面"]
    
    # 打乱每个类别的数据
    random.shuffle(positive_samples)
    random.shuffle(negative_samples)
    
    total_positive = len(positive_samples)
    total_negative = len(negative_samples)
    total_size = total_positive + total_negative
    
    # 计算每个集合中每个类别的样本数
    train_pos = int(total_positive * TRAIN_RATIO)
    val_pos = int(total_positive * VAL_RATIO)
    test_pos = total_positive - train_pos - val_pos
    
    train_neg = int(total_negative * TRAIN_RATIO)
    val_neg = int(total_negative * VAL_RATIO)
    test_neg = total_negative - train_neg - val_neg
    
    # 分层抽样
    train_data = positive_samples[:train_pos] + negative_samples[:train_neg]
    val_data = positive_samples[train_pos:train_pos + val_pos] + negative_samples[train_neg:train_neg + val_neg]
    test_data = positive_samples[train_pos + val_pos:] + negative_samples[train_neg + val_neg:]
    
    # 打乱每个数据集
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)
    
    # 转换为Dataset并tokenize
    train_dataset = tokenize_samples(Dataset.from_list(train_data), tokenizer)
    eval_dataset = tokenize_samples(Dataset.from_list(val_data), tokenizer)
    test_dataset = tokenize_samples(Dataset.from_list(test_data), tokenizer)
    
    print(f"原始数据分布:")
    print(f"  正面样本: {total_positive} 条 ({total_positive/total_size:.1%})")
    print(f"  负面样本: {total_negative} 条 ({total_negative/total_size:.1%})")
    print(f"  总计: {total_size} 条")
    
    print(f"\n数据集划分:")
    print(f"  训练集: {len(train_dataset)} 条 (正面:{train_pos}, 负面:{train_neg})")
    print(f"  验证集: {len(eval_dataset)} 条 (正面:{val_pos}, 负面:{val_neg})")
    print(f"  测试集: {len(test_dataset)} 条 (正面:{test_pos}, 负面:{test_neg})")
    
    print(f"\n各集合正负比例:")
    print(f"  训练集: 正面 {train_pos/(train_pos+train_neg):.1%}, 负面 {train_neg/(train_pos+train_neg):.1%}")
    print(f"  验证集: 正面 {val_pos/(val_pos+val_neg):.1%}, 负面 {val_neg/(val_pos+val_neg):.1%}")
    print(f"  测试集: 正面 {test_pos/(test_pos+test_neg):.1%}, 负面 {test_neg/(test_pos+test_neg):.1%}")
    
    print(f"\n平均序列长度 ~{len(train_dataset[0]['input_ids'])}")
    
    return train_dataset, eval_dataset, test_dataset


@dataclass
class DataCollatorForSupervisedDataset:
    """自定义数据整理器，处理不同长度的序列"""
    tokenizer: AutoTokenizer
    
    def __call__(self, instances):
        input_ids = [instance["input_ids"] for instance in instances]
        labels = [instance["labels"] for instance in instances]
        attention_mask = [instance["attention_mask"] for instance in instances]
        
        # 动态 padding 到批次内最大长度
        max_len = max(len(seq) for seq in input_ids)
        
        padded_input_ids = []
        padded_labels = []
        padded_attention_mask = []
        
        for i in range(len(input_ids)):
            seq_len = len(input_ids[i])
            pad_len = max_len - seq_len
            
            padded_input_ids.append(input_ids[i] + [self.tokenizer.pad_token_id] * pad_len)
            padded_labels.append(labels[i] + [-100] * pad_len)
            padded_attention_mask.append(attention_mask[i] + [0] * pad_len)
        
        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
        }


def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    if USE_QLORA:
        if BitsAndBytesConfig is None:
            raise ImportError("需要 transformers>=4.30 与 bitsandbytes 才能启用QLoRA，请先安装相关依赖。")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )

    # 为LoRA训练准备模型（无论是否使用量化）
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    
    if USE_QLORA:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return tokenizer, model


def train():
    set_seed(SEED)
    ensure_dir(ADAPTER_OUTPUT_DIR)
    if SAVE_MERGED_MODEL:
        ensure_dir(MERGED_MODEL_DIR)

    tokenizer, model = load_model_and_tokenizer()
    train_dataset, eval_dataset, test_dataset = prepare_datasets(tokenizer)

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=ADAPTER_OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        per_device_eval_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        evaluation_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_steps=SAVE_STEPS,
        logging_steps=LOG_STEPS,
        save_total_limit=1,  # 只保留最新的检查点
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type="cosine",
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=False,  # 已在模型层面启用
        optim="adamw_torch",
        max_grad_norm=1.0,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # 从最新检查点恢复训练
    resume_from_checkpoint = "/data/xinwang/finetune_model/outputs/lora_adapter/checkpoint-3400"
    print(f"从检查点恢复训练: {resume_from_checkpoint}")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(ADAPTER_OUTPUT_DIR)
    tokenizer.save_pretrained(ADAPTER_OUTPUT_DIR)
    print(f"LoRA 适配器已保存到: {ADAPTER_OUTPUT_DIR}")

    if SAVE_MERGED_MODEL:
        merge_lora_weights(ADAPTER_OUTPUT_DIR, MERGED_MODEL_DIR)


def merge_lora_weights(adapter_dir: str, merged_dir: str) -> None:
    """将LoRA权重合并进基础模型，生成可直接用于predict.py的完整权重。"""
    print("开始合并LoRA权重...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=None,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    peft_model = PeftModel.from_pretrained(base_model, adapter_dir)
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(merged_dir)
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    tokenizer.save_pretrained(merged_dir)
    print(f"合并后的模型已保存到: {merged_dir}\n可在predict.py中将BASE_MODEL_PATH指向该目录进行评估。")


if __name__ == "__main__":
    train()
