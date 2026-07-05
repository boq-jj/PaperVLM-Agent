# PaperVLM-Agent

PaperVLM-Agent 是一个面向科研论文图表理解的多模态智能体原型。项目目标是结合 PDF 解析、图表抽取、RAG 检索和多模态大模型，让用户能够上传论文 PDF 或图表图片，并围绕论文内容、图表含义和实验结果进行问答。

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

当前版本提供 PDF 解析、文本分块、FAISS 索引、命令行问答、Gradio Demo 和可供前端调用的 FastAPI 后端。

解析 PDF：

```bash
python scripts/run_parse_pdf.py --pdf data/raw_papers/example.pdf
```

构建 RAG 索引：

```bash
python scripts/run_chunk_text.py --input data/extracted/text/example.json
python scripts/run_build_index.py --chunks data/extracted/chunks/example_chunks.json
```

命令行问答：

```bash
python scripts/run_ask.py --paper-id example --query "What does Figure 1 show?"
```

启动 Gradio Demo：

```bash
python scripts/run_demo.py
```

启动 REST API 后端：

```bash
python scripts/run_api.py --host 127.0.0.1 --port 8000
```

前端连接本地后端时，将 API 地址设为：

```text
http://127.0.0.1:8000
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

PaperVLM-Agent includes two lightweight evaluation paths: a self-built paper QA set for text RAG and a small ChartQA-compatible template for chart question answering. These evaluation outputs can be used in README summaries, reports, and experiment tables.

### 自建论文 QA 评测

`data/eval/example_text_qa.jsonl` contains 20 English QA examples for `example.pdf` (`Attention Is All You Need`). Each record includes `paper_id`, `question`, `expected_pages`, `reference_answer`, and `question_type`.

Run the text RAG evaluation with the qwen-vl backend:

```powershell
python scripts\run_eval_text_rag.py --eval-file data\eval\example_text_qa.jsonl --paper-id example --llm-backend qwen-vl
```

If `DASHSCOPE_API_KEY` is not available, run the same pipeline with the mock backend:

```powershell
python scripts\run_eval_text_rag.py --eval-file data\eval\example_text_qa.jsonl --paper-id example --llm-backend mock
```

Output:

```text
data/eval/results/example_text_qa_results.json
```

Current text RAG metrics include:

- `page_hit_rate`
- `success`
- `failed`
- `by_question_type`

### ChartQA 图表问答评测

#### Synthetic ChartQA-style sample

`data/eval/chartqa_sample.jsonl` is a small synthetic ChartQA-style template. It is generated locally and is used to verify that the image QA pipeline, matching logic, and error analysis work correctly. It is not an official ChartQA benchmark result.

Put images under the paths referenced by `image_path`, for example:

```text
data/datasets/chartqa/images/example_001.png
```

For a reproducible local smoke test, generate five ChartQA-style sample charts:

```powershell
python scripts\create_chartqa_sample_images.py
```

Run ChartQA evaluation:

```powershell
python scripts\run_eval_chartqa.py --eval-file data\eval\chartqa_sample.jsonl
```

Output:

```text
data/eval/results/chartqa_sample_results.json
```

#### Official ChartQA Hugging Face subset

For a more realistic small-scale chart QA evaluation, export a subset from `HuggingFaceM4/ChartQA`:

```powershell
python scripts/prepare_chartqa_hf_sample.py --num-samples 50
```

This creates:

```text
data/eval/chartqa_hf_sample.jsonl
data/datasets/chartqa/images/chartqa_hf_000001.png
```

Run qwen-vl evaluation on the exported subset:

```powershell
python scripts/run_eval_chartqa.py --eval-file data/eval/chartqa_hf_sample.jsonl
```

Generate error analysis:

```powershell
python scripts/run_error_analysis.py --eval-result data/eval/results/chartqa_hf_sample_results.json --task-type chartqa
```

If Hugging Face download fails, check the network connection or set a mirror endpoint such as:

```powershell
$env:HF_ENDPOINT="https://hf-mirror.com"
```

You can still use the synthetic ChartQA-style sample first to validate the local pipeline.

Current ChartQA metrics include:

- `exact_match_rate`
- `relaxed_match_rate`
- `success`
- `failed`
- `by_question_type`

### 错误分析

Generate text RAG error analysis:

```powershell
python scripts\run_error_analysis.py --eval-result data\eval\results\example_text_qa_results.json --task-type text_rag
```

Generate ChartQA error analysis:

```powershell
python scripts\run_error_analysis.py --eval-result data\eval\results\chartqa_sample_results.json --task-type chartqa
```

The analysis command does not call the LLM API and does not modify the original result file. It writes derived artifacts to `data/eval/analysis/`:

- `{stem}_summary.md`
- `{stem}_error_cases.jsonl`
- `{stem}_table.csv`
- `{stem}_by_type.csv`

Current error labels include:

- text RAG: `retrieval_miss`, `low_top_score`, `empty_answer`, `short_answer`, `no_retrieved_chunks`, `runtime_error`
- ChartQA: `exact_miss`, `relaxed_miss`, `image_missing`, `empty_prediction`, `runtime_error`

## Ablation Study

PaperVLM-Agent includes lightweight ablation scripts for checking how retrieval depth, RAG evidence, and visual inputs affect small-scale QA behavior. These experiments are designed for quick project reports and debugging; they are small-scale ablations, not full benchmark results.

### Top-k Ablation

Compare `top_k = 1, 3, 5, 10` on the self-built text RAG QA set. The default backend is `mock` to avoid unnecessary API cost:

```powershell
python scripts/run_ablation_topk.py --llm-backend mock
```

Outputs:

```text
data/eval/ablation/topk_ablation_results.json
data/eval/ablation/topk_ablation_table.csv
```

### No-RAG vs Text-RAG Ablation

Compare direct question answering without retrieved evidence against the existing text RAG pipeline:

```powershell
python scripts/run_ablation_rag.py
```

To run the pipeline without a qwen API call, use:

```powershell
python scripts/run_ablation_rag.py --llm-backend mock
```

Outputs:

```text
data/eval/ablation/rag_ablation_results.json
data/eval/ablation/rag_ablation_table.csv
```

### Visual Input Ablation

Compare `text_only`, `image_only`, and `text_image` modes on a small visual QA JSONL file:

```powershell
python scripts/run_ablation_visual.py
```

The default input file is:

```text
data/eval/example_visual_qa.jsonl
```

If this file does not exist, the script prints the expected JSONL fields and exits without calling the API. Outputs are written to:

```text
data/eval/ablation/visual_ablation_results.json
data/eval/ablation/visual_ablation_table.csv
```

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

The backend in this repository can be started with:

```powershell
python scripts\run_api.py --host 0.0.0.0 --port 8000
```

It provides:

- `GET /health`
- `POST /api/process-pdf`
- `POST /api/ask`
- `POST /api/ask-visual`

Keep `DASHSCOPE_API_KEY` only on the backend. Do not expose it in the Netlify frontend.
