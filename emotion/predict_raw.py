#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import json
import re
import glob
import os
from collections import Counter, defaultdict
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
#修改此处和268行处文件路径
BASE_MODEL_PATH = "/data/xinwang/finetune_model"#放基础模型的路径
ADAPTER_OUTPUT_DIR = "/data/xinwang/finetune_model/emotion/lora_adapter"#放LoRA适配器的路径

BATCH_SIZE = 16
MAX_LENGTH = 512


def has_garbled_text(text):
    """
    检测文本是否包含乱码
    
    Args:
        text: 待检测的文本
        
    Returns:
        bool: 如果包含乱码返回True，否则返回False
    """
    if not isinstance(text, str):
        return False
    
    # 检测特殊符号序列（如 **$%...^&**）
    if re.search(r'\*\*[\$%^&]+.*?\^&\*\*', text):
        return True
    
    # 检测特殊字符占比是否过高
    special_chars = re.findall(r'[^\u4e00-\u9fa5a-zA-Z0-9\s，。！？、：；""''（）《》\-]', text)
    if len(special_chars) > len(text) * 0.3:  # 超过30%是特殊字符
        return True
    
    return False


def process_online_service_file(file_path):
    """
    处理在线客服通话记录文件
    
    Args:
        file_path: Excel文件路径
        
    Returns:
        dict: 包含客户和坐席对话会话的字典
    """
    print(f'\n处理文件: {os.path.basename(file_path)}')
    
    try:
        # 读取"会话"sheet
        df = pd.read_excel(file_path, sheet_name='会话')
        print(f'  总记录数: {len(df)}')
        print(f'  列名: {list(df.columns)}')
        
        # 按会话ID分组
        sessions = defaultdict(list)
        
        for _, row in df.iterrows():
            session_id = row['会话ID']
            role = row['角色']
            content = row['语句内容']
            
            # 跳过系统消息
            if role == '系统':
                continue
            
            # 跳过NaN值
            if pd.isna(content) or content == '' or str(content).strip() == '' or str(content).upper() == 'NAN':
                continue
            
            # 检测并过滤乱码
            if has_garbled_text(str(content)):
                continue
            
            sessions[session_id].append({
                'role': role,
                'content': content,
                'date': row.get('统计日期', ''),
                'agent_id': row.get('坐席工号', '')
            })
        
        # 分离客户和坐席对话会话
        customer_sessions = []
        agent_sessions = []
        
        for session_id, turns in sessions.items():
            if not turns:  # 空会话（全是乱码或系统消息）
                continue
            
            # 分别构建客户和坐席的对话内容
            customer_dialogue = []
            agent_dialogue = []
            
            for turn in turns:
                # 处理客户对话
                if turn['role'] == '客户':
                    customer_turn = {
                        '内容': turn['content']
                    }
                    customer_dialogue.append(customer_turn)
                
                # 处理坐席对话（相当于客服）
                elif turn['role'] == '坐席':
                    agent_turn = {
                        '内容': turn['content']
                    }
                    agent_dialogue.append(agent_turn)
            
            # 添加到对应的会话列表
            if customer_dialogue:
                customer_sessions.append({
                    '会话ID': session_id,
                    '对话内容': customer_dialogue
                })
            
            if agent_dialogue:
                agent_sessions.append({
                    '会话ID': session_id,
                    '对话内容': agent_dialogue
                })
        
        # 统计信息
        unique_sessions = df['会话ID'].nunique()
        customer_count = len([t for turns in sessions.values() for t in turns if t['role'] == '客户'])
        agent_count = len([t for turns in sessions.values() for t in turns if t['role'] == '坐席'])
        
        print(f'  总会话数: {unique_sessions} 个')
        print(f'  客户对话条数: {customer_count} 条')
        print(f'  坐席对话条数: {agent_count} 条')
        print(f'  客户会话数: {len(customer_sessions)} 个')
        print(f'  坐席会话数: {len(agent_sessions)} 个')
        
        return {
            'customer_sessions': customer_sessions,
            'agent_sessions': agent_sessions,
            'stats': {
                'total_sessions': unique_sessions,
                'customer_turns': customer_count,
                'agent_turns': agent_count,
                'customer_sessions_count': len(customer_sessions),
                'agent_sessions_count': len(agent_sessions)
            }
        }
        
    except Exception as e:
        print(f'  ❌ 错误: {str(e)}')
        import traceback
        traceback.print_exc()
        return None


def create_prompt(text: str) -> str:
    """构造与微调时一致的情感分析 Prompt。"""
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


def load_finetuned_model():
    """加载基础模型并应用 LoRA 适配器。"""
    print("\n加载微调后的 LoRA 模型用于原始 Excel 预测...")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # 为 decoder-only 结构使用 left padding，避免 transformers 的警告
    tokenizer.padding_side = "left"

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


def predict_batch(tokenizer, model, texts):
    """对一个批次的文本进行预测，返回"正面"/"负面"标签列表。"""
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
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    predictions = []
    for i, response in enumerate(responses):
        # 特殊规则：问候语判定为正面
        text = texts[i]
        greeting_keywords = ["在吗", "在不在", "在么", "在嘛", "有人吗", "有人么", "有人在吗", "有人在么"]
        if any(keyword in text for keyword in greeting_keywords):
            predictions.append("正面")
            continue
        
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


def main():
    """主函数"""
    # 修改要预测的原始文件路径参数
    base_dir = '/data/xinwang/finetune_model/emotion'#所有要预测的原始文件的路径

    # 查找所有符合格式的文件（当前只会有一个江西在线客服通话记录文件）
    pattern = os.path.join(base_dir, '*在线客服通话记录*.xls*')
    excel_files = glob.glob(pattern)

    if not excel_files:
        print(f'未找到符合格式的文件: {pattern}')
        return

    print(f'找到 {len(excel_files)} 个在线客服通话记录文件:')
    for file in excel_files:
        print(f'  - {os.path.basename(file)}')

    # 加载微调后的模型
    tokenizer, model = load_finetuned_model()

    for file_path in excel_files:
        print(f"\n开始对文件进行情绪预测: {os.path.basename(file_path)}")

        # 读取 Excel 中的会话 sheet
        df = pd.read_excel(file_path, sheet_name='会话')
        print(f"  会话记录总数: {len(df)}")

        # 仅对角色为客户的语句进行预测
        customer_mask = df['角色'] == '客户'
        customer_texts = df.loc[customer_mask, '语句内容']

        # 过滤空值
        valid_mask = customer_mask & customer_texts.notna() & (customer_texts.astype(str).str.strip() != '')
        texts_to_predict = df.loc[valid_mask, '语句内容'].astype(str).tolist()

        print(f"  需要预测的客户语句条数: {len(texts_to_predict)}")

        predictions = []
        if len(texts_to_predict) > 0:
            num_batches = (len(texts_to_predict) + BATCH_SIZE - 1) // BATCH_SIZE
            pbar = tqdm(total=num_batches, desc="情绪预测进度", ncols=100)

            for i in range(0, len(texts_to_predict), BATCH_SIZE):
                batch_texts = texts_to_predict[i:i + BATCH_SIZE]
                batch_preds = predict_batch(tokenizer, model, batch_texts)
                predictions.extend(batch_preds)
                pbar.update(1)

            pbar.close()

        # 将预测结果对齐写回 DataFrame 新列
        df['模型情绪预测'] = ''
        df.loc[valid_mask, '模型情绪预测'] = predictions

        # 特殊规则：对于系统生成的固定入口描述，强制判定为正面
        # 文本示例："人工**$%changjingzhi^&**和家亲-投诉场景客户咨询（直进人工客服）-FW-固定型入口"
        special_mask = df['语句内容'].astype(str).str.contains('**$%changjingzhi^&**', regex=False)
        df.loc[special_mask, '模型情绪预测'] = '正面'

        # 生成总览统计 DataFrame
        total_customer = int(valid_mask.sum())
        pos_count = int((df.loc[valid_mask, '模型情绪预测'] == '正面').sum())
        neg_count = int((df.loc[valid_mask, '模型情绪预测'] == '负面').sum())

        overview_data = {
            '指标': ['总客户语句数', '预测为正面', '预测为负面', '正面占比(%)', '负面占比(%)'],
            '数值': [
                total_customer,
                pos_count,
                neg_count,
                round(pos_count / total_customer * 100, 4) if total_customer > 0 else 0.0,
                round(neg_count / total_customer * 100, 4) if total_customer > 0 else 0.0,
            ],
        }
        overview_df = pd.DataFrame(overview_data)

        # 读取原有所有 sheet
        xls = pd.ExcelFile(file_path)
        # 将结果写入新的 .xlsx 文件，避免对 .xls 写入引擎的兼容性问题
        base_name, _ = os.path.splitext(file_path)
        backup_path = base_name + "_带情绪预测.xlsx"

        # 对于 .xlsx 文件，使用 openpyxl 引擎
        # 如环境中缺少 openpyxl，请先安装: pip install openpyxl
        with pd.ExcelWriter(backup_path, engine="openpyxl") as writer:
            for sheet_name in xls.sheet_names:
                if sheet_name == '会话':
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                else:
                    sheet_df = pd.read_excel(xls, sheet_name=sheet_name)
                    sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)

            overview_df.to_excel(writer, sheet_name='情绪预测总览', index=False)

        print(f"  已将预测结果写回源文件，并生成'情绪预测总览' sheet: {backup_path}")

    print('\n🎉 原始在线客服通话记录情绪预测完成！')


if __name__ == '__main__':
    main()

