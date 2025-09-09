# genai_prototypes

Prototypes and utilities for building **GenAI-powered HR and RAG applications** using Google Gemini, FAISS, BM25, and Gradio.

## Features

- **Gemini usage logging**
  - Structured logging of queries, responses, and token counts
  - CSV persistence with automatic load/save (`GeminiUsageLogger`)
- **HR-specific LLM helpers**
  - CV (PDF) summarization → education, experience, skills
  - JD (Job Description) summarization → key responsibilities, must/should-have skills
  - CV × JD matching → structured template with ✓/✗ requirements, strengths/weaknesses, and matching score
- **Vector + Keyword retrieval**
  - Embedding with Gemini (`text-embedding-004`)
  - FAISS vector store with metadata persistence
  - BM25 keyword index
  - Reciprocal Rank Fusion (RRF) for hybrid retrieval
- **RAG pipeline**
  - Hybrid search (FAISS + BM25) → context block → Gemini answer generation
  - Source previews with scores
- **Interactive UI**
  - Gradio chat interface with history
  - Configurable retrieval knobs (k-values, weights, RRF damping)
  - Debug panel for sources and logs

## Project Structure

```
genai_prototypes/
├── ExampleCV 
├── vector_and_bm25_dbs
│   ├── bm25_metadata.jsonl
│   ├── vector_index.faiss
│   └── vector_metadata.jsonl
├── genai_functions/
│   ├── gemini_usage_logging.py      # CSV logger, FaissVectorStore, BM25Index
│   ├── gemini_textgen_functions     # Basic functions to generate Text
│   ├── gemini_texteva_functions     # Evaluation result of LLM (Not integrated with the rest yet)
│   ├── gemini_hr_llm_functions.py   # CV/JD summarization & matching helpers
│   ├── hybrid_vectordb_functions.py # FAISS vector db and BM25 index creation 
│   ├── hybrid_rag_functions.py      # Hybrid RAG search & fusion
│   ├── helper_functions.py          # Utility helpers
│   └── gradio_chatbot_functions.py  # Gradio chat demo factory
├── logs/                            # CSV usage logs (created at runtime)
├── call_gemini_api_w_hybrid_rag.ipynb   # Example notebook for creating FAISS vector db and BM25 index, then run Gradio for chatbot test (with RAG & History)
├── llm_evaluator.ipynb                  # Example notebook for creating LLM text evaluator function
├── mcp.ipynb                            # Example notebook for MCP with Calculation, API calling, SQL connection, and iterative single step planner (Pending multistep planner)
├── hf_finetuning_w_lora.ipynb           # Example notebook for full & lora finetunning on Huggingface model
└── opensource_ocr.ipynb                 # Example notebook for open source OCR with tesserocr and easyocr
```

## Installation

```bash
git clone https://github.com/nachai-l/genai_prototypes.git
cd genai_prototypes
pip install -r requirements.txt
```

Dependencies:
- Python 3.10+
- [google-genai](https://pypi.org/project/google-genai/) SDK
- `faiss`, `rank-bm25`, `pandas`, `gradio`

## Usage

### 1. Summarize CVs and build indexes

```python
from genai_functions.gemini_hr_llm_functions import master_gemini_upload_cv_prompt_w_log
from genai_functions.hybrid_rag_functions import master_loading_cvs_to_faiss_and_bm25

logs_df = master_loading_cvs_to_faiss_and_bm25(
    client=client,
    uploaded_filenames_list=["cv1.pdf", "cv2.pdf"],
    applied_position="Data Scientist",
)
```

### 2. Summarize a JD

```python
from genai_functions.gemini_hr_llm_functions import master_gemini_jd_prompt_w_log

resp, logs_df, formatted_text = master_gemini_jd_prompt_w_log(
    prompt_text=open("jd.txt").read(),
    client=client,
    log_path="logs/gemini_usage.csv",
)
print(formatted_text)
```

### 3. Match CV × JD

```python
from genai_functions.gemini_hr_llm_functions import master_gemini_cvxjd_matching_w_log

resp, logs_df, parsed_dict = master_gemini_cvxjd_matching_w_log(
    resp_cv=cv_summary_response,
    resp_jd=jd_summary_response,
    client=client,
    log_path="logs/gemini_usage.csv",
)
```

### 4. Run Hybrid RAG Q&A

```python
from genai_functions.hybrid_rag_functions import master_gemini_rag_bm25_answer_w_log

resp, logs_df, ctx = master_gemini_rag_bm25_answer_w_log(
    client=client,
    query="Which candidates worked in Singapore?",
    vector_store=loaded_store,
    bm25_index=bm25_index,
    log_path="logs/gemini_usage.csv",
)
print(resp.text)
```

### 5. Launch Gradio Chat

```python
from ui.rag_chat_ui import make_rag_chat_demo

app = make_rag_chat_demo(
    client=client,
    vector_store=loaded_store,
    bm25_index=bm25_index,
    log_path="logs/gemini_usage.csv",
)
app.launch(inline=True, share=False)
```

## Roadmap

### Core & Infrastructure
- [ ] **Docker image** for quick deployment (CPU + optional CUDA build)
- [ ] **Config system** (dotenv / Pydantic) for API keys, paths, and model knobs
- [ ] **CLI utilities** (e.g., `ingest-cv`, `build-index`, `rag-query`)
- [ ] **Batching & retries** for Gemini calls; exponential backoff
- [ ] **Async pipeline** for parallel CV ingestion & embedding
- [ ] **Pluggable embeddings** (Gemini, OpenAI, Voyage, E5) with a common interface
- [ ] **Storage adapters** (local FS, S3/GCS) for logs and indexes
      
### RAG & Retrieval Quality
- [ ] **Better hybrid fusion** (α-RRF, Borda, normalised z-scores) A/B tested
- [ ] **Query rewriting** (hyDE / multi-query) before retrieval
- [ ] **Reranking** (Cohere, Cross-Encoder) after fusion
- [ ] **Chunking strategy** options (semantic, sliding window)
- [ ] **Evaluation harness** (nDCG/MRR/Recall@k on a small HR test set)
      
### LLM Generation & Evaluation
- [ ] **Template library** for CV/JD prompts, few-shot examples
- [ ] **Guarded generation** (hallucination checks; answerability detection)
- [ ] **Integrate `gemini_texteva_functions.py`** in `llm_evaluator.ipynb` for auto-validation

### MCP (Model Context Protocol)
- [ ] **MCP tools POC** in `mcp.ipynb` (calc, API calling, SQL, single-step planner)
- [ ] **Multistep planner** + memory for iterative task execution
- [ ] **Secure connectors** (db creds via env/secret store; read-only policies)

### HR Features
- [ ] **JD parser** → structured schema (title, level, must/should skills, years, location)
- [ ] **CV parser** → extract contact, roles, durations, orgs, locations (fallback OCR)
- [ ] **CV×JD matching v2** → per-requirement scoring, weighting, rationales
- [ ] **Calibration tool** to tune weights with labeled examples
- [ ] **Bias/privacy guardrails** (mask PII before indexing, configurable)
