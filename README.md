# 客服对话分类与 LoRA 微调

基于本地 Llama 架构语言模型，对客服对话执行情绪、同理心与礼貌性分类。项目提供 Python 训练、批量推理和评估脚本。

| 目录 | 任务 | 输出标签 | 微调适配器目录 |
| --- | --- | --- | --- |
| `emotion/` | 客户语句情绪分类 | 正面 / 负面 | `emotion/lora_adapter/` |
| `empathy/` | 结合上下文判断客服同理心 | 合规 / 不合规 | `empathy/lora_adapter/` |
| `polite/` | 客服语句礼貌性分类 | 合规 / 不合规 | `polite/lora_adapter_v2/` |

## 部署

请先阅读 [部署文档](部署文档.md)，其中包含环境安装、依赖版本、单条推理示例、批量任务、路径配置和已知问题。

推理需要根目录基础模型的四个 `model-*.safetensors` 分片，以及所选任务的 `adapter_model.safetensors`。基础权重与 LoRA 适配器必须配套使用。获取仓库后应先检查这些权重是否已另行提供；源码与配置本身不能替代模型权重。

部分原脚本使用固定 Linux 绝对路径，礼貌性训练与评估脚本还引用了当前目录中缺失的数据文件。运行前需按照部署文档完成配置，不能直接假设所有脚本开箱即用。

## 主要入口

| 用途 | 脚本 |
| --- | --- |
| 情绪 LoRA 评估 | `emotion/evaluate_finetuned.py` |
| Excel 情绪预测 | `emotion/predict_raw.py` |
| 同理心 LoRA 评估 | `empathy/evaluate_finetuned.py` |
| 礼貌性 LoRA 评估 | `polite/evaluate_finetuned.py` |
| 情绪微调 | `emotion/finetune.py` |
| 同理心微调 | `empathy/finetune_empathy.py` |
| 礼貌性微调 | `polite/finetune_politeness.py` |

项目目前通过命令行运行，没有 HTTP API 或 Web 界面。训练为可选操作，使用已有适配器推理无需重新训练。

## 模型与数据

本仓库包含代码、模型和 tokenizer 配置、适配器配置、项目说明与部署文档。模型权重、训练检查点、业务数据、历史预测结果和日志通过 `.gitignore` 排除，需在部署机器上另行配置。

完整本地交付包约 20 GiB，保留这些模型与数据。部署文档中的 ZIP 内容与校验说明针对该完整交付包；GitHub 下载的源码包不包含这些被排除的文件。

## 许可

项目保留原有 [LICENSE](LICENSE)。
