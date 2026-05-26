# PaperVLM-Agent

PaperVLM-Agent 是一个面向本科保研项目的科研论文图表理解智能体原型。项目目标是结合 PDF 解析、图表抽取、RAG 检索和多模态大模型，让用户能够上传论文 PDF 或图表图片，并围绕论文内容、图表含义和实验结果进行问答。

## 项目背景

科研论文中大量关键信息分布在正文、表格、图像、坐标图和页面布局中。传统文本检索方法难以完整理解图表和上下文之间的关系。本项目希望构建一个多模态智能体流程，将论文文本解析、图表抽取、向量检索和视觉语言模型推理结合起来，为论文阅读、图表解释和问答提供一个可运行 Demo。

## 功能目标

1. 支持上传论文 PDF 或单独的图表图片。
2. 解析 PDF 文本，保存页面文本、页面截图和候选图表区域。
3. 对论文文本进行分块，并构建 FAISS 向量索引。
4. 根据用户问题检索相关论文片段，形成 RAG 上下文。
5. 调用多模态大模型理解图表图片并回答问题。
6. 使用 Gradio 构建可视化交互 Demo。
7. 后续支持 ChartQA、ScienceQA、AI2D 等数据集评测。

## 目录结构

```text
PaperVLM-Agent/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── config.yaml
├── data/
│   ├── raw_papers/
│   ├── extracted/
│   │   ├── text/
│   │   ├── figures/
│   │   └── pages/
│   ├── datasets/
│   │   ├── chartqa/
│   │   ├── scienceqa/
│   │   └── ai2d/
│   └── eval/
├── src/
│   ├── pdf/
│   ├── rag/
│   ├── vlm/
│   ├── agent/
│   ├── eval/
│   └── app/
├── scripts/
├── notebooks/
└── assets/
```

## 安装方法

建议使用 Python 3.10 或 3.11。

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

如果需要使用 GPU 推理，请根据本机 CUDA 版本单独安装匹配的 PyTorch 版本。

## 运行方式

当前版本是项目基础结构和占位代码，核心算法会在后续步骤逐步实现。

解析 PDF：

```bash
python scripts/run_parse_pdf.py --pdf data/raw_papers/example.pdf
```

构建 RAG 索引：

```bash
python scripts/run_build_index.py
```

命令行问答：

```bash
python scripts/run_ask.py --question "What does Figure 1 show?"
```

启动 Gradio Demo：

```bash
python scripts/run_demo.py
```

## 后续计划

- 实现 PDF 文本解析和页面截图保存。
- 实现基于 PyMuPDF 的候选图表抽取。
- 实现文本分块、embedding 生成和 FAISS 索引构建。
- 接入 Qwen-VL 或其他多模态大模型。
- 设计 Agent 规划模块，自动选择文本检索或图表理解工具。
- 增加 ChartQA、ScienceQA、AI2D 数据集评测脚本。
- 完善 Gradio 可视化 Demo 和错误分析 Notebook。

## Troubleshooting: BGE embedding model download

If you see:

```text
Failed to load embedding model: BAAI/bge-small-en-v1.5
```

it is usually caused by a Hugging Face model download failure.

On Windows PowerShell, try configuring a Hugging Face mirror for the current terminal session:

```powershell
.\scripts\setup_hf_mirror.ps1
python scripts\test_bge_download.py
```

After the model download test succeeds, rebuild the FAISS index:

```powershell
python scripts\run_build_index.py --chunks data\extracted\chunks\example_chunks.json --model-name BAAI/bge-small-en-v1.5
```

If the script reports that direct `SentenceTransformer("BAAI/bge-small-en-v1.5")` loading failed but the local fallback model is ready, use the local model path instead:

```powershell
python scripts\run_build_index.py --chunks data\extracted\chunks\example_chunks.json --model-name models\bge-small-en-v1.5-hf-mirror
```

In the Gradio Demo, set the `embedding_model` textbox to:

```text
models\bge-small-en-v1.5-hf-mirror
```

If it still fails, you can temporarily use a qwen-api embedding backend or `text-embedding-v4` in a later implementation.

## Evaluation

PaperVLM-Agent includes a small text RAG evaluation set at:

```text
data/eval/example_text_qa.jsonl
```

Each JSONL record contains:

```json
{
  "id": "q001",
  "paper_id": "example",
  "question": "What is the Transformer architecture?",
  "expected_pages": [2, 3],
  "reference_answer": "A short reference answer.",
  "question_type": "method"
}
```

Run text RAG evaluation with the real LLM backend:

```powershell
python scripts\run_eval_text_rag.py --eval-file data\eval\example_text_qa.jsonl --paper-id example
```

If `DASHSCOPE_API_KEY` is not available, run the pipeline with the mock backend:

```powershell
python scripts\run_eval_text_rag.py --eval-file data\eval\example_text_qa.jsonl --paper-id example --llm-backend mock
```

The evaluator saves full results to:

```text
data/eval/results/example_text_qa_results.json
```

Current metrics include:

- `success` / `failed`
- `page_hit_rate`
- `by_question_type`

The evaluation output is designed for later manual scoring. Future fields can include:

- `answer_correctness`
- `evidence_support`
- `hallucination`
- `reasoning_quality`

## Evaluation and Error Analysis

Run the text RAG evaluation with the qwen-vl backend:

```powershell
python scripts\run_eval_text_rag.py --eval-file data\eval\example_text_qa.jsonl --paper-id example --llm-backend qwen-vl
```

Generate error analysis tables and a Markdown report from an existing result file:

```powershell
python scripts\run_error_analysis.py --eval-result data\eval\results\example_text_qa_results.json
```

The analysis command does not call the LLM API and does not modify the original result file. It writes derived artifacts to `data/eval/analysis/`:

- `example_text_qa_summary.md`
- `example_text_qa_error_cases.jsonl`
- `example_text_qa_table.csv`
- `example_text_qa_by_type.csv`

Current error labels include:

- `retrieval_miss`
- `empty_answer`
- `short_answer`
- `low_top_score`
- `runtime_error`
- `possible_success`

## Netlify Frontend

The project now includes a Netlify-ready Vite + React frontend in `frontend/`.
This frontend is a deployable UI shell for PaperVLM-Agent. It can run in a static
demo mode, or call a separately deployed Python backend through an API base URL.

Run locally:

```powershell
cd frontend
npm install
npm run dev
```

Build for Netlify:

```powershell
cd frontend
npm run build
```

Root `netlify.toml` is configured with:

```toml
[build]
base = "frontend"
command = "npm run build"
publish = "dist"
```

To connect the Netlify frontend to a Python backend, set this Netlify environment variable:

```text
VITE_PAPERVLM_API_BASE_URL=https://your-backend.example.com
```

Keep `DASHSCOPE_API_KEY` only on the backend. Do not expose it in the Netlify frontend.
