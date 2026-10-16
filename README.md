# 客服对话分类与 LoRA 微调

本项目使用本地 Llama 架构语言模型，对客服对话执行情绪、同理心和礼貌性分类，提供 LoRA 微调、批量推理和评估脚本。

| 任务 | 输入 | 输出标签 | 默认适配器目录 |
| --- | --- | --- | --- |
| 情绪分类 `emotion` | 客户单句 | 正面 / 负面 | `emotion/lora_adapter/` |
| 同理心分类 `empathy` | 客服回复及前 5 条对话 | 合规 / 不合规 | `empathy/lora_adapter/` |
| 礼貌性分类 `polite` | 客服单句 | 合规 / 不合规 | `polite/lora_adapter_v2/`，另保留旧版配置 |

**部署顺序：克隆源码 → 补齐模型权重 → 创建环境并安装依赖 → 验证单条推理 → 按需配置数据并运行批量任务。**

项目通过 Python 命令行运行，目前没有 HTTP API、Web 界面或 Docker 部署入口。使用已有 LoRA 权重推理时，无需重新训练。下文相对路径均以项目根目录为起点。

## 1. 仓库内容与模型准备

GitHub 仓库包含代码、模型配置、tokenizer、适配器配置及本 README。`.gitignore` 排除了模型权重、训练检查点、业务数据、历史预测结果、Excel 文件和训练日志。**仅执行 git clone 不能直接进行模型推理，必须另行补齐权重。**

```text
finetune_model/
├── README.md
├── LICENSE
├── config.json
├── generation_config.json
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
├── model.safetensors.index.json
├── model-00001-of-00004.safetensors  # 需另行准备
├── model-00002-of-00004.safetensors  # 需另行准备
├── model-00003-of-00004.safetensors  # 需另行准备
├── model-00004-of-00004.safetensors  # 需另行准备
├── emotion/
│   ├── requirements.txt
│   ├── finetune.py
│   ├── predict.py
│   ├── predict_raw.py
│   ├── evaluate_finetuned.py
│   └── lora_adapter/
│       ├── adapter_config.json
│       └── adapter_model.safetensors  # 需另行准备
├── empathy/
│   ├── finetune_empathy.py
│   ├── predict_empathy.py
│   ├── evaluate_finetuned.py
│   ├── stratified_sampling_accuracy.py
│   └── lora_adapter/
│       ├── adapter_config.json
│       └── adapter_model.safetensors  # 需另行准备
└── polite/
    ├── finetune_politeness.py
    ├── predict_politeness.py
    ├── evaluate_finetuned.py
    ├── lora_adapter/                 # 旧版适配器配置
    └── lora_adapter_v2/
        ├── adapter_config.json
        └── adapter_model.safetensors  # 需另行准备
```

### ModelScope 模型下载

可在 ModelScope（魔搭）平台获取与本项目配置接近的 **Meta-Llama-3-8B-Instruct** 模型：

- [模型主页与下载入口：LLM-Research/Meta-Llama-3-8B-Instruct](https://modelscope.cn/models/LLM-Research/Meta-Llama-3-8B-Instruct)
- [模型文件列表](https://modelscope.cn/models/LLM-Research/Meta-Llama-3-8B-Instruct/files)
- [ModelScope 官方下载说明](https://modelscope.cn/docs/models/download)

进入模型页面的文件列表，下载四个 `model-00001-of-00004.safetensors` 至 `model-00004-of-00004.safetensors` 分片。下载文件应放在上方目录树标示的项目根目录，不能放入某个任务的 `lora_adapter/` 目录。用于本项目 Transformers/PEFT 加载流程的文件格式为 safetensors。

**对应关系说明：** 该 ModelScope 条目的模型架构、层数、隐藏维度、词表和配置中的结束 token 与本项目相符，但四个分片的 SHA-256 与原始本地交付文件不同。因此，这里提供的是同架构指令模型的下载入口，尚未确认它就是现有 LoRA 训练时使用的原始基座。需要复现已有微调结果时，应向维护者确认并获取原训练基座；不能仅依据相同架构就保证适配器效果一致。

ModelScope 上述链接提供基础模型，不包含本项目训练得到的情绪、同理心或礼貌性 LoRA 权重。各任务的 `adapter_model.safetensors` 仍需向项目维护者单独获取，放入对应适配器目录。仓库中的配置与 tokenizer 也可能与镜像当前版本不同，部署已有适配器时应保留与原训练配套的文件。

根目录配置为 `LlamaForCausalLM`，32 层、隐藏维度 4096、词表大小 128256、最大位置长度 8192。原始基础权重为 BF16，四个分片合计约 14.96 GiB；每套 LoRA 权重约 320 MiB，r=64、alpha=16。原始本地基座的具体发行版本和来源仍待确认。

只运行一个任务时，准备基础模型和该任务适配器即可。`checkpoint-*`、`optimizer.pt`、`scheduler.pt`、`training_args.bin` 不是推理必需文件。LoRA 不是独立完整模型，不能只复制适配器权重就运行。业务数据仅在批量任务或评估时需要，第 5 节单条示例无需业务数据。

原始完整本地项目约 20 GiB，包含多套检查点与业务数据；这不是 GitHub 源码仓库的体积。GitHub 自动下载的源码 ZIP 同样不含被排除的模型和数据。

## 2. 环境与容量规划

以下为按权重大小估算的部署起点，未在本机进行显存或速度实测：

| 项目 | 部署起点 |
| --- | --- |
| 操作系统 | Linux 或 Windows；本文分别给出虚拟环境命令 |
| Python | 优先使用 3.10 或 3.11，便于安装下文固定版本依赖 |
| GPU 推理 | NVIDIA GPU，建议至少 24 GiB 显存；从 batch_size=1 开始 |
| 系统内存 | GPU 推理建议至少 32 GiB；CPU 推理建议 64 GiB 或更多 |
| 磁盘 | 仅推理建议至少 25 GiB；同时保存完整交付 ZIP 并解压建议至少 50 GiB；训练另留空间 |
| CPU 推理 | 脚本有 FP32 回退路径，单基础模型权重约 30 GiB 内存，执行速度需实测 |

各脚本会独立加载一份基础模型。首次验收请串行运行三个任务，避免多个 Python 进程同时占用显存。`device_map="auto"` 可能把部分层放到 CPU；能启动不代表吞吐满足要求。其行为可参考 [Accelerate 大模型推理文档](https://huggingface.co/docs/accelerate/usage_guides/big_modeling)。

现有训练代码默认 `USE_QLORA=False`，CUDA 下启用 BF16；训练内存需求明显高于推理。24 GiB 不能视为默认训练配置可运行的保证。训练前应检查 `torch.cuda.is_bf16_supported()`，不支持时需要调整模型 dtype 与训练精度参数。

## 3. 获取代码并创建虚拟环境

先安装 Git 和 Python 3.11，克隆源码：

```bash
git clone https://github.com/y0ungb0y/finetune_model.git
cd finetune_model
```

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

如果执行策略阻止激活，可直接用 `.\.venv\Scripts\python.exe` 替代下文的 `python`，无需更改系统策略。

### Linux Bash

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

如果使用维护者另行提供的完整 ZIP，先通过支持 ZIP64 的工具解压，再进入其中的项目根目录创建环境。基础模型分片大于 4 GiB，应使用 NTFS、ext4 等支持大文件的文件系统。下文安装与运行命令应使用同一个虚拟环境。

## 4. 安装依赖

现有依赖清单为 `emotion/requirements.txt`，且遗漏了源码实际用到的 `accelerate`、`scikit-learn`、`numpy`、训练用 `datasets` 及读取 `.xls` 用的 `xlrd`。其宽泛下限也不足以约束整套脚本兼容性。

以下是一组供首次部署验证的固定版本起点，并非从原训练环境导出的锁定文件，也未经本次端到端运行验证。选择 Transformers 4.44.2 是因为源码同时使用了 `evaluation_strategy` 与 `eval_strategy`；该版本文档同时列出了两者。详见 [Transformers 4.44.2 Trainer 参数](https://huggingface.co/docs/transformers/v4.44.2/en/main_classes/trainer)。适配器自带模型卡记录 PEFT 0.10.0，本文使用更新的 0.12.0 作为待验证起点。

先安装 PyTorch。以下 CUDA 12.4 构建命令来自 [PyTorch 历史版本安装页](https://pytorch.org/get-started/previous-versions/)，须与目标机器驱动、GPU 架构兼容；其他硬件请按 [官方安装选择器](https://pytorch.org/get-started/locally/) 选择匹配构建。较新的显卡可能需要更新 PyTorch，此时须重新验收依赖组合。

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

纯 CPU 环境将上面命令替换为：

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
```

然后安装共同依赖（Windows 和 Linux 命令相同）：

```bash
python -m pip install transformers==4.44.2 peft==0.12.0 accelerate==0.33.0 safetensors==0.4.5 numpy==1.26.4 scikit-learn==1.5.2 pandas==2.2.3 openpyxl==3.1.5 xlrd==2.0.1 tqdm==4.66.5
python -m pip check
```

只有需要训练时再安装：

```bash
python -m pip install datasets==2.21.0
```

默认部署不需要 bitsandbytes。只有自行启用 QLoRA 时才需另行匹配其环境；本文不提供未经验证的量化运行配置。

检查环境：

```bash
python -c "import torch,transformers,peft,accelerate,pandas,sklearn,xlrd; print('torch',torch.__version__,'transformers',transformers.__version__,'peft',peft.__version__); print('CUDA',torch.cuda.is_available(),'runtime',torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

在 NVIDIA GPU 机器上还可运行 `nvidia-smi`。如果预期使用 GPU 而 `CUDA` 输出为 `False`，应先解决驱动或 PyTorch 构建问题。

## 5. 首次单条推理验收：无需修改原脚本

将以下代码保存为项目根目录的 `smoke_inference.py`。这是文档提供的操作示例，仓库中没有预先创建该文件。它复用现有提示词函数，显式指定本地模型和适配器，绕开原脚本的绝对路径及评估数据要求。三个任务应分别运行，每次只加载一个模型。

```python
import argparse
import importlib.util
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("task", choices=["emotion", "empathy", "polite"])
args = parser.parse_args()
root = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "task_eval", root / args.task / "evaluate_finetuned.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

if args.task == "emotion":
    prompt = module.create_prompt("等了这么久还没处理好，太差了！")
    labels = ("负面", "正面")
elif args.task == "polite":
    prompt = module.create_prompt("您好，请稍等，我马上为您查询，谢谢您的耐心。")
    labels = ("不合规", "合规")
else:
    context = [{"role": "客户", "content": "我等了很久，真的很着急。"}]
    system, user = module.create_empathy_prompt(
        context, "抱歉让您久等了，我理解您的着急，我马上为您核实。"
    )
    # 复用训练脚本使用的文本格式。
    prompt = f"System: {system}\n\nUser: {user}"
    labels = ("不合规", "合规")

adapter = root / args.task / (
    "lora_adapter_v2" if args.task == "polite" else "lora_adapter"
)
index = json.loads((root / "model.safetensors.index.json").read_text("utf-8"))
required = [root / name for name in set(index["weight_map"].values())]
required += [adapter / "adapter_config.json", adapter / "adapter_model.safetensors"]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit("缺少模型文件，请先补齐：\n" + "\n".join(missing))
tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
use_cuda = torch.cuda.is_available()
model = AutoModelForCausalLM.from_pretrained(
    root,
    torch_dtype=torch.float16 if use_cuda else torch.float32,
    device_map="auto" if use_cuda else None,
    low_cpu_mem_usage=True,
    local_files_only=True,
)
model = PeftModel.from_pretrained(model, str(adapter), local_files_only=True)
model.eval()
inputs = tokenizer(
    prompt, return_tensors="pt", truncation=True,
    max_length=1024 if args.task == "empathy" else 512,
)
inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
with torch.inference_mode():
    outputs = model.generate(
        **inputs, max_new_tokens=10, do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
answer = tokenizer.decode(
    outputs[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
).strip()
label = next((value for value in labels if value in answer), "未识别")
print({"task": args.task, "raw_answer": answer, "label": label})
if label == "未识别":
    raise SystemExit("模型未生成预期标签，请检查提示词、tokenizer 与适配器。")
```

运行：

```bash
python smoke_inference.py emotion
python smoke_inference.py empathy
python smoke_inference.py polite
```

验收标准：基础分片和所选适配器成功加载、没有缺失文件或 CUDA 错误、生成内容可解析为该任务的合法标签。示例语句的语义预期不能替代独立测试集质量验收。本示例只解码新增 token，避免把提示词里的候选标签误认为模型输出。

## 6. 原有批量脚本的路径配置

`empathy` 的训练、推理和评估入口使用脚本所在位置计算路径，迁移后一般无需改模型和数据路径；其 `stratified_sampling_accuracy.py` 仍写死 Linux 路径。`emotion`、`polite` 多数路径固定为 `/data/xinwang/finetune_model`，需要修改后再运行。

推荐在要运行的原脚本顶部配置区使用以下模式，而不是只更改工作目录：

```python
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL_PATH = str(ROOT)
```

随后逐项调整该脚本已定义的变量。原脚本不提供统一命令行参数，也不会自动读取 `BASE_MODEL_PATH` 等同名环境变量。

| 入口 | 必须检查的配置与注意事项 |
| --- | --- |
| `emotion/evaluate_finetuned.py` | `BASE_MODEL_PATH`；`DATA_FILE` 指向 `ROOT / "emotion" / "平衡后的客户对话数据.json"`；`ADAPTER_OUTPUT_DIR` 指向 `ROOT / "emotion" / "lora_adapter"`；调整两个输出路径 |
| `emotion/predict.py` | `BASE_MODEL_PATH`、`ALL_DATA_FILES` 和两个输出路径；仅基础模型，不加载 LoRA |
| `emotion/predict_raw.py` | `BASE_MODEL_PATH`、`ADAPTER_OUTPUT_DIR`，还要修改 `main()` 中的 `base_dir`，它不受顶部模型路径控制 |
| `polite/predict_politeness.py` | `BASE_MODEL_PATH`、`DATA_PATH` 和 `OUTPUT_PATH`；实际数据为 `polite/对话内容礼貌性.json`；仅基础模型 |
| `polite/evaluate_finetuned.py` | `BASE_MODEL_PATH`、`ADAPTER_PATH`、`DATA_PATH`、`OUTPUT_PATH`；当前引用的去重数据文件未交付，见第 8 节 |
| `empathy/stratified_sampling_accuracy.py` | `RESULTS_FILE` 改为实际的 `empathy/finetuned_predict_results.json` 路径 |

例如情绪评估的完整路径配置可写成：

```python
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL_PATH = str(ROOT)
DATA_FILE = str(ROOT / "emotion" / "平衡后的客户对话数据.json")
ADAPTER_OUTPUT_DIR = str(ROOT / "emotion" / "lora_adapter")
OUTPUT_RESULTS_FILE = str(ROOT / "emotion" / "evaluate_results_new.json")
OUTPUT_ERRORS_FILE = str(ROOT / "emotion" / "evaluate_errors_new.json")
```

适配器配置 JSON 中也记录了旧的 `base_model_name_or_path`。本文和现有评估脚本都先加载显式指定的基础模型，再调用 `PeftModel.from_pretrained(base_model, adapter_path)`，因此无需仅为迁移而修改适配器 JSON。若改用自动读取基础路径的加载方式，则需要另外处理该字段。参考 [PEFT 模型加载接口](https://huggingface.co/docs/peft/v0.12.0/en/package_reference/peft_model)。

## 7. 批量推理与结果文件

完成第 6 节配置后，根据需求选择入口：

| 目的 | 命令 | 默认结果位置 |
| --- | --- | --- |
| 情绪 LoRA 测试集评估 | `python emotion/evaluate_finetuned.py` | `emotion/evaluate_results.json`、`evaluate_errors.json`，或自设路径 |
| 情绪基础模型对照 | `python emotion/predict.py` | `emotion/predict_results.json`、`predict_errors.json` |
| Excel 情绪批量预测 | `python emotion/predict_raw.py` | 输入文件旁生成 `原文件名_带情绪预测.xlsx` |
| 同理心 LoRA 评估 | `python empathy/evaluate_finetuned.py` | `empathy/finetuned_predict_results.json`、`finetuned_predict_errors.json` |
| 同理心基础模型对照 | `python empathy/predict_empathy.py` | `empathy/empathy_predict_results.json`、`empathy_predict_errors.json` |
| 礼貌性基础模型对照 | `python polite/predict_politeness.py` | `polite/预测结果.json` |
| 礼貌性 LoRA 评估 | `python polite/evaluate_finetuned.py` | `polite/微调后预测结果v3.json`，必须先补齐或重设数据路径 |

结果 JSON 通常包含统计、指标和预测详情；原脚本会覆盖同名输出文件，正式运行前建议改成新文件名。默认 batch_size 分别为情绪 16、同理心 LoRA 4、礼貌性 8；首次运行应改为 1，再根据显存逐步调大。

Excel 入口要求输入表包含 `会话` 工作表，至少有 `会话ID`、`角色`、`语句内容` 列，使用角色 `客户` 筛选待预测语句。读取 `.xls` 需要 xlrd，写出 `.xlsx` 需要 openpyxl。它会扫描 `*在线客服通话记录*.xls*`，带预测结果的 XLSX 也可能被再次匹配。请将 `base_dir` 指向只含待处理原始文件的新目录，避免重复处理并生成多重 `_带情绪预测` 后缀。

同理心 JSON 的顶层是会话列表，每项含 `会话编号`、`对话内容`、`语句数`；`对话内容` 为字符串列表。解析器严格匹配类似 `客户: 我很着急 合规`、`客服: 抱歉让您久等了 合规` 的格式，使用英文冒号和空格，末尾必须为 `合规` 或 `不合规`。格式不匹配的行会被跳过。

### 数据文件格式

以下仅展示格式，示例不是可用于报告性能的测试集。实际业务数据需另行获取或自行准备，并按脚本配置放置。

情绪任务默认使用 `emotion/平衡后的客户对话数据.json`：

```json
[
  {"语句内容": "您好，我想查询处理进度。", "情绪积极": "正面"},
  {"语句内容": "等了这么久还没处理好，太差了！", "情绪积极": "负面"}
]
```

礼貌性任务可使用 `polite/对话内容礼貌性.json`，但训练和 LoRA 评估的默认文件名不同，必须按第 8 节处理：

```json
[
  {"语句内容": "您好，请稍等，我帮您查询。", "礼貌性": "合规", "样本来源": "原始"},
  {"语句内容": "自己看说明。", "礼貌性": "不合规", "样本来源": "原始"}
]
```

同理心任务默认使用 `empathy/所有会话_同理心数据.json`：

```json
[
  {
    "会话编号": "demo-001",
    "对话内容": [
      "客户: 我等了很久，真的很着急。 合规",
      "客服: 抱歉让您久等了，我理解您的着急，马上为您核实。 合规"
    ],
    "语句数": 2
  }
]
```

## 8. 已确认的部署障碍与结果解释

以下问题来自当前源码与实际文件清单的对照，编写文档时未改动原业务脚本：

1. **情绪训练数据路径不正确。** `emotion/finetune.py` 的 `DATA_FILE` 指向根目录，实际数据位于 `emotion/平衡后的客户对话数据.json`。
2. **情绪训练强制恢复到缺失检查点。** 同一脚本在 `train()` 中写死 `outputs/lora_adapter/checkpoint-3400`，原始本地交付中没有该检查点。开始新训练时需改成 `resume_from_checkpoint=None`，恢复训练则需提供真实兼容检查点；仅改模型路径不能解决此问题。
3. **礼貌性训练和评估数据未完整交付。** `finetune_politeness.py` 引用 `对话内容礼貌性_清洗后.json`，`evaluate_finetuned.py` 引用 `对话内容_礼貌性标签_去重.json`，两者未包含在原始本地交付文件中。维护者的原始本地数据中已有 `对话内容礼貌性.json` 共 13721 条，可以明确修改数据路径后用于新的试运行，但不能据此声称复现原清洗、去重流程或历史指标。
4. **同理心训练和评估的提示词包装不同。** 训练使用文本 `System: ... User: ...`，现有评估使用 tokenizer 的 `apply_chat_template`。第 5 节采用训练的文本形式；评估方式改变可能影响结果，需要在固定数据集上分别验证。
5. **情绪 LoRA 评估默认右侧 padding。** 对生成式批量推理，迁移时建议把 `tokenizer.padding_side` 调整为 `"left"` 后重新评估；第 5 节使用左侧 padding。
6. **部分脚本解码整个输入和输出，并对异常答案回退为默认标签。** 这可能掩盖输出格式错误。应结合原始生成文本、错误样本和每类指标判断结果，不能仅以脚本完成或退出码为依据。部分入口捕获异常后仅打印，不一定返回失败退出码。
7. **历史结果的采样方法不同。** 情绪 LoRA 评估构造测试拆分；同理心 LoRA 评估默认抽取 500 个完整会话，不保证与训练集隔离；礼貌性评估进行类别平衡抽样。现有结果文件属于历史产物，不能作为新机器已部署成功或独立泛化能力的证明。

训练是可选操作，推理部署无需重新训练。若要重新训练，先完成上述数据和检查点处理，并将 `ADAPTER_OUTPUT_DIR` 指向新的输出目录：

```bash
python emotion/finetune.py
python empathy/finetune_empathy.py
python polite/finetune_politeness.py
```

同理心训练默认 `RESUME_FROM_CHECKPOINT=True`，会自动选择已有检查点。要开始独立新实验，设置为 `False` 并使用新的输出目录。不要把现有交付适配器目录直接当作新实验输出目录。

## 9. 离线部署

离线机器需要另行准备源码、基础模型、适配器、所需数据、Python 解释器、依赖 wheel 和匹配的 GPU 驱动。先在与目标机器操作系统、CPU 架构和 Python 版本一致的联网环境验证第 4 节依赖，再准备离线 wheel：

```bash
python -m pip freeze > requirements.lock.txt
python -m pip download -r requirements.lock.txt --dest wheelhouse --extra-index-url https://download.pytorch.org/whl/cu124
```

CPU 环境将额外索引改为 `/whl/cpu`。若固定构建或间接依赖找不到，应在联网准备阶段解决，不要把其他平台下载的 wheel 直接复制使用。

把源码、权重、所需数据、`requirements.lock.txt` 和 `wheelhouse/` 传入目标机，在新建虚拟环境中安装：

```bash
python -m pip install --no-index --find-links=wheelhouse -r requirements.lock.txt
python -m pip check
```

Windows PowerShell 设置离线模式：

```powershell
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
```

Linux Bash 设置离线模式：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

第 5 节另外使用 `local_files_only=True`。如果仍出现联网请求，应检查路径是否存在，避免将不存在的本地路径解释为远程模型标识。

## 10. 常见故障

| 现象 | 处理 |
| --- | --- |
| 找不到 `/data/xinwang/...` | 按第 6 节更改脚本内路径，包含函数内的路径 |
| `No module named sklearn/accelerate/datasets/xlrd` | 在运行脚本的同一个虚拟环境补齐第 4 节依赖 |
| CUDA out of memory | 先串行运行并减小 batch_size 至 1；再检查上下文长度、其他 GPU 进程和自动 CPU 分配情况 |
| BF16 不支持 | 推理使用脚本既有 FP16 路径；训练需同时修改模型 dtype 与 bf16/fp16 参数 |
| `unexpected keyword argument eval_strategy` | 检查 Transformers 版本；不要直接采用 requirements 的过低下限 |
| tokenizer 没有 pad token | 设置 `tokenizer.pad_token = tokenizer.eos_token`，批量生成采用左侧 padding |
| 找不到 adapter 或基础模型分片 | 同时检查适配器目录和根目录四个分片；不要只复制单个 LoRA 文件 |
| Excel 提示不支持 xls | 确认安装 xlrd；不要仅凭文件后缀改变读取引擎 |
| 同理心抽取样本为空 | 检查对话字符串格式、末尾标签和英文冒号，见第 7 节 |
| 训练提示没有 checkpoint-3400 | 修正情绪训练函数中硬编码的恢复参数，见第 8 节 |

## 11. 验收范围与许可

部署验收至少应完成以下步骤：

1. 基础模型的四个分片与目标适配器文件齐全，路径与本 README 一致。
2. 同一虚拟环境中 `python -m pip check` 通过；预期使用 GPU 时 CUDA 可用。
3. 第 5 节单条推理成功加载模型，输出合法标签，无显存或内存错误。
4. 需要批量任务时，补齐数据并修改输入、输出路径，先用 batch_size=1 执行，再检查预测详情和错误样本。
5. 需要评估质量时，固定独立测试集、提示词及采样方式，分别检查各类别指标。

本次文档更新核对了源码、路径、配置及示例语法。当前维护环境没有安装完整模型运行依赖，本文版本组合和示例未经过本次真实模型加载、GPU 推理或训练验收，需在目标环境验证。

仓库保留原有 [LICENSE](LICENSE)，内容为 Meta Llama 3 Community License Agreement。基础模型与适配器应与其来源和许可配套提供。
