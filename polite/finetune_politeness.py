#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""利用现有对话数据对基础模型进行LoRA微调，以提升礼貌性分类准确率。"""

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
    TrainingArguments,
    Trainer,
)

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
DATA_FILE = "/data/xinwang/finetune_model/polite/对话内容礼貌性_清洗后.json"
ADAPTER_OUTPUT_DIR = "/data/xinwang/finetune_model/polite/lora_adapter_v2"

TRAIN_RATIO = 0.80  # 训练集比例
VAL_RATIO = 0.10    # 验证集比例
TEST_RATIO = 0.10   # 测试集比例
MAX_SEQ_LENGTH = 512
SEED = 42

# 训练超参
MICRO_BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 4
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01
SAVE_STEPS = 400
LOG_STEPS = 50
EVAL_STEPS = 200

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

USE_QLORA = False
# ==================================================


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
    cleaned = [item for item in data if item.get("语句内容") and item.get("礼貌性") in {"合规", "不合规"}]
    print(f"加载数据: {len(cleaned)} 条（原始 {len(data)} 条）")
    return cleaned


def create_prompt(text: str) -> str:
    """与预测阶段保持一致的提示语。"""
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


def tokenize_samples(dataset: Dataset, tokenizer: AutoTokenizer) -> Dataset:
    """将原始数据集转换为语言模型可训练的格式。"""

    def preprocess_function(examples):
        input_ids_list = []
        attention_masks = []
        labels_list = []

        for text, label in zip(examples["语句内容"], examples["礼貌性"]):
            prompt = create_prompt(text)
            full_text = prompt + label

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
    positive_samples = [item for item in data if item["礼貌性"] == "合规"]
    negative_samples = [item for item in data if item["礼貌性"] == "不合规"]
    
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
    
    print(f"\n原始数据分布:")
    print(f"  合规样本: {total_positive} 条 ({total_positive/total_size:.1%})")
    print(f"  不合规样本: {total_negative} 条 ({total_negative/total_size:.1%})")
    print(f"  总计: {total_size} 条")
    
    print(f"\n数据集划分:")
    print(f"  训练集: {len(train_dataset)} 条 (合规:{train_pos}, 不合规:{train_neg})")
    print(f"  验证集: {len(eval_dataset)} 条 (合规:{val_pos}, 不合规:{val_neg})")
    print(f"  测试集: {len(test_dataset)} 条 (合规:{test_pos}, 不合规:{test_neg})")
    
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
    print("\n加载基础模型...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    if USE_QLORA:
        if BitsAndBytesConfig is None:
            raise ImportError("需要 transformers>=4.30 与 bitsandbytes 才能启用QLoRA")
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

    # 为LoRA训练准备模型
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
    print("模型加载完成！")
    return tokenizer, model


def train():
    print("=" * 60)
    print("礼貌性分类模型 LoRA 微调")
    print("=" * 60)
    
    set_seed(SEED)
    ensure_dir(ADAPTER_OUTPUT_DIR)

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
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_steps=SAVE_STEPS,
        logging_steps=LOG_STEPS,
        save_total_limit=2,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type="cosine",
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=False,
        optim="adamw_torch",
        max_grad_norm=1.0,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    print("\n开始训练...")
    trainer.train()
    
    # 保存最终模型
    trainer.save_model(ADAPTER_OUTPUT_DIR)
    tokenizer.save_pretrained(ADAPTER_OUTPUT_DIR)
    print(f"\nLoRA 适配器已保存到: {ADAPTER_OUTPUT_DIR}")
    
    # 保存训练信息
    train_info = {
        "base_model": BASE_MODEL_PATH,
        "data_file": DATA_FILE,
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset),
        "test_samples": len(test_dataset),
        "epochs": NUM_EPOCHS,
        "batch_size": MICRO_BATCH_SIZE * GRADIENT_ACCUMULATION,
        "learning_rate": LEARNING_RATE,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
    }
    with open(os.path.join(ADAPTER_OUTPUT_DIR, "train_info.json"), "w", encoding="utf-8") as f:
        json.dump(train_info, f, ensure_ascii=False, indent=2)
    
    print("\n🎉 微调完成！")


if __name__ == "__main__":
    train()
