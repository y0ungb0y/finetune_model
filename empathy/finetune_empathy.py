#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
利用现有对话数据对基础模型进行LoRA微调，以提升同理心分类准确率。
结合上下文进行训练，提高模型对客服同理心的判断能力。
"""

import os
os.environ["TRANSFORMERS_NO_TORCH_LOAD_CHECK"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "0"

import json
import random
import re
from dataclasses import dataclass
from typing import List, Dict, Any

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
BASE_MODEL_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "所有会话_同理心数据.json")
ADAPTER_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lora_adapter")

TRAIN_RATIO = 0.70  # 训练集比例
VAL_RATIO = 0.15    # 验证集比例
TEST_RATIO = 0.15   # 测试集比例
MAX_SEQ_LENGTH = 1024  # 增加长度以容纳上下文
CONTEXT_TURNS = 5   # 使用前N轮对话作为上下文
SEED = 42

# 训练超参
MICRO_BATCH_SIZE = 2  # 因为序列更长，减小batch size
GRADIENT_ACCUMULATION = 8
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01
SAVE_STEPS = 200
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
SAVE_MERGED_MODEL = False
RESUME_FROM_CHECKPOINT = True  # 是否从检查点恢复训练
# ==================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_dialogue_line(line: str):
    """解析对话行，提取角色、内容和标签"""
    match = re.match(r'^(客户|客服): (.+?) (合规|不合规)$', line)
    if match:
        return match.group(1), match.group(2), match.group(3)
    return None, None, None


def extract_training_samples(data: List[Dict]) -> List[Dict[str, Any]]:
    """从对话数据中提取训练样本"""
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
                    "label": turn["label"]
                })
    
    return samples


def create_empathy_prompt(context: List[Dict], agent_response: str) -> str:
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

    return f"System: {system_message}\n\nUser: {user_message}"


def load_dialogue_data() -> List[Dict[str, Any]]:
    """加载并处理对话数据"""
    print(f"加载数据文件: {DATA_FILE}")
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"原始会话数: {len(data)}")
    
    # 提取训练样本
    samples = extract_training_samples(data)
    
    # 过滤有效样本
    valid_samples = [s for s in samples if s["label"] in {"合规", "不合规"}]
    
    print(f"提取的客服回复样本数: {len(valid_samples)}")
    
    # 统计标签分布
    compliant = sum(1 for s in valid_samples if s["label"] == "合规")
    non_compliant = sum(1 for s in valid_samples if s["label"] == "不合规")
    print(f"标签分布: 合规={compliant}, 不合规={non_compliant}")
    
    return valid_samples


def tokenize_samples(dataset: Dataset, tokenizer: AutoTokenizer) -> Dataset:
    """将原始数据集转换为语言模型可训练的格式"""

    def preprocess_function(examples):
        input_ids_list = []
        attention_masks = []
        labels_list = []

        for context, response, label in zip(
            examples["context"], 
            examples["agent_response"], 
            examples["label"]
        ):
            prompt = create_empathy_prompt(context, response)
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
    """准备训练、验证和测试数据集"""
    data = load_dialogue_data()
    
    # 按标签分组
    compliant_samples = [item for item in data if item["label"] == "合规"]
    non_compliant_samples = [item for item in data if item["label"] == "不合规"]
    
    # 打乱每个类别的数据
    random.shuffle(compliant_samples)
    random.shuffle(non_compliant_samples)
    
    total_compliant = len(compliant_samples)
    total_non_compliant = len(non_compliant_samples)
    total_size = total_compliant + total_non_compliant
    
    # 计算每个集合中每个类别的样本数
    train_comp = int(total_compliant * TRAIN_RATIO)
    val_comp = int(total_compliant * VAL_RATIO)
    test_comp = total_compliant - train_comp - val_comp
    
    train_non = int(total_non_compliant * TRAIN_RATIO)
    val_non = int(total_non_compliant * VAL_RATIO)
    test_non = total_non_compliant - train_non - val_non
    
    # 分层抽样
    train_data = compliant_samples[:train_comp] + non_compliant_samples[:train_non]
    val_data = compliant_samples[train_comp:train_comp + val_comp] + non_compliant_samples[train_non:train_non + val_non]
    test_data = compliant_samples[train_comp + val_comp:] + non_compliant_samples[train_non + val_non:]
    
    # 打乱每个数据集
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)
    
    # 转换为Dataset格式
    def convert_to_dataset_format(samples):
        return {
            "context": [s["context"] for s in samples],
            "agent_response": [s["agent_response"] for s in samples],
            "label": [s["label"] for s in samples]
        }
    
    train_dataset = tokenize_samples(Dataset.from_dict(convert_to_dataset_format(train_data)), tokenizer)
    eval_dataset = tokenize_samples(Dataset.from_dict(convert_to_dataset_format(val_data)), tokenizer)
    test_dataset = tokenize_samples(Dataset.from_dict(convert_to_dataset_format(test_data)), tokenizer)
    
    print(f"\n原始数据分布:")
    print(f"  合规样本: {total_compliant} 条 ({total_compliant/total_size:.1%})")
    print(f"  不合规样本: {total_non_compliant} 条 ({total_non_compliant/total_size:.1%})")
    print(f"  总计: {total_size} 条")
    
    print(f"\n数据集划分:")
    print(f"  训练集: {len(train_dataset)} 条 (合规:{train_comp}, 不合规:{train_non})")
    print(f"  验证集: {len(eval_dataset)} 条 (合规:{val_comp}, 不合规:{val_non})")
    print(f"  测试集: {len(test_dataset)} 条 (合规:{test_comp}, 不合规:{test_non})")
    
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
    """加载模型和tokenizer"""
    print(f"\n加载基础模型: {BASE_MODEL_PATH}")
    
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
    return tokenizer, model


def get_latest_checkpoint(output_dir: str) -> str:
    """获取最新的检查点目录"""
    if not os.path.exists(output_dir):
        return None
    
    checkpoints = []
    for name in os.listdir(output_dir):
        if name.startswith("checkpoint-"):
            checkpoint_path = os.path.join(output_dir, name)
            if os.path.isdir(checkpoint_path):
                try:
                    step = int(name.split("-")[1])
                    checkpoints.append((step, checkpoint_path))
                except ValueError:
                    continue
    
    if not checkpoints:
        return None
    
    # 返回步数最大的检查点
    checkpoints.sort(key=lambda x: x[0], reverse=True)
    return checkpoints[0][1]


def train():
    """主训练函数"""
    print("=" * 60)
    print("同理心分类模型 LoRA 微调")
    print("=" * 60)
    
    set_seed(SEED)
    ensure_dir(ADAPTER_OUTPUT_DIR)
    
    # 检查是否有可用的检查点
    resume_checkpoint = None
    if RESUME_FROM_CHECKPOINT:
        resume_checkpoint = get_latest_checkpoint(ADAPTER_OUTPUT_DIR)
        if resume_checkpoint:
            print(f"\n发现检查点: {resume_checkpoint}")
            print("将从该检查点继续训练...")
        else:
            print("\n未发现检查点，将从头开始训练...")

    tokenizer, model = load_model_and_tokenizer()
    train_dataset, eval_dataset, test_dataset = prepare_datasets(tokenizer)

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=ADAPTER_OUTPUT_DIR,
        overwrite_output_dir=False if resume_checkpoint else True,  # 恢复训练时不覆盖
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        per_device_eval_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        eval_strategy="steps",  # 新版本transformers使用eval_strategy
        eval_steps=EVAL_STEPS,
        save_steps=SAVE_STEPS,
        logging_steps=LOG_STEPS,
        save_total_limit=2,  # 保留最近3个检查点，便于恢复
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type="cosine",
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=False,
        optim="adamw_torch",
        max_grad_norm=1.0,
        report_to="none",
        load_best_model_at_end=True,  # 训练结束时加载最佳模型
        metric_for_best_model="eval_loss",  # 使用验证集损失作为最佳模型的指标
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    print("\n开始训练...")
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    
    # 保存最终模型
    trainer.save_model(ADAPTER_OUTPUT_DIR)
    tokenizer.save_pretrained(ADAPTER_OUTPUT_DIR)
    print(f"\nLoRA 适配器已保存到: {ADAPTER_OUTPUT_DIR}")

    if SAVE_MERGED_MODEL:
        merged_dir = os.path.join(os.path.dirname(ADAPTER_OUTPUT_DIR), "merged_model")
        ensure_dir(merged_dir)
        merge_lora_weights(ADAPTER_OUTPUT_DIR, merged_dir)
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)


def merge_lora_weights(adapter_dir: str, merged_dir: str) -> None:
    """将LoRA权重合并进基础模型"""
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
    print(f"合并后的模型已保存到: {merged_dir}")


if __name__ == "__main__":
    train()
