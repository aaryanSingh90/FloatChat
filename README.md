# 🌊 Float Chat – AI-Powered Conversational Oceanographic Data Explorer

> **⚠️ Data Notice:** All observations in the demo dataset are **synthetic** (randomly generated) for educational purposes only. They are **not** real Argo scientific measurements.

Float Chat is a full-stack AI/ML application that lets you explore oceanographic data using plain English. It combines a **RAG pipeline** (Sentence Transformers + FAISS), **Google Gemini** (optional), a safe **Pandas Query Engine**, and a **Streamlit** ocean-themed dashboard.

---

## 🏗️ Architecture

```
User question
     │
     ▼
┌─────────────────────────────────────────────┐
│              AI Pipeline (ai/pipeline.py)   │
│                                             │
│  1. RAG: embed question → FAISS search      │
│         → retrieve scientific docs          │
│                                             │
│  2. Intent: Gemini API (if key set)         │
│             OR rule-based fallback parser   │
│         → structured query plan (JSON)      │
│                                             │
│  3. Query Engine:                           │
│     validate → optimise → execute (Pandas) │
│                                             │
│  Result: DataFrame + scalar + plan + docs   │
└─────────────────────────────────────────────┘
     │
     ▼
Streamlit Dashboard (app.py)
```

---

## 📁 Project Structure

```
Float-Chat/
├── app.py                      # Streamlit entry point
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
│
├── ai/                         # AI layer
│   ├── pipeline.py             # End-to-end orchestrator
│   ├── gemini_client.py        # Google Gemini API wrapper
│   └── intent_parser.py        # Rule-based fallback parser
│
├── core/                       # Query engine
│   ├── query_engine.py         # Top-level orchestrator
│   ├── query_validator.py      # Allowlist-based validator (security)
│   ├── query_optimizer.py      # Pre-execution optimisation
│   └── query_executor.py       # Safe Pandas execution (no eval/exec)
│
├── rag/                        # RAG pipeline
│   ├── embedder.py             # Sentence Transformers + FAISS index
│   └── retriever.py            # High-level retriever + context formatter
│
├── evaluation/                 # Evaluation framework
│   ├── test_questions.py       # 32 ground-truth test cases
│   └── evaluator.py            # Metrics computation + report
│
├── visualization/
│   └── charts.py               # Plotly chart builders
│
├── data/
│   ├── generate_dataset.py     # 50-record synthetic dataset generator
│   └── sample_argo_data.csv    # Generated on first run
│
├── knowledge_base/
│   └── documents.py            # 18 scientific documentation chunks
│
├── config/
│   └── settings.py             # Env-var based configuration
│
└── tests/                      # Pytest test suite
    ├── conftest.py
    ├── test_dataset.py
    ├── test_query_validator.py
    ├── test_query_executor.py
    ├── test_intent_parser.py
    ├── test_pipeline.py
    └── test_rag.py
```

---

## 🚀 Quick Start

### 1. Clone / navigate to the project

```bash
cd "Float-Chat"
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate.bat    # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) Configure Gemini API key

```bash
cp .env.example .env
# Edit .env and set GEMINI_API_KEY=your_key_here
```

The application **works fully without an API key** in rule-based + RAG mode.

### 5. Generate the dataset (auto-runs on first app launch)

```bash
python data/generate_dataset.py
```

### 6. Run the Streamlit application

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

### 7. Run the test suite

```bash
pytest tests/ -v --tb=short
```

---

## 🤖 AI Modes

| Mode | Description | Requires |
|------|-------------|---------|
| **Gemini** | Google Gemini LLM parses intent, guided by RAG context | `GEMINI_API_KEY` env var |
| **Rule-Based** | Regex/keyword parser extracts intent deterministically | Nothing – always works |

Both modes feed their query plan through the same **validator → optimizer → executor** pipeline. No arbitrary code is ever executed.

---

## 🔍 RAG Pipeline

1. **Knowledge Base** – 18 scientific documents covering Argo floats, temperature, salinity, pressure, QC flags, regions, and oceanographic terminology.
2. **Embedding** – `all-MiniLM-L6-v2` (Sentence Transformers) encodes documents and queries locally.
3. **FAISS Index** – CPU-based L2 nearest-neighbour search retrieves the top-3 most relevant docs.
4. **Context injection** – Retrieved docs are prepended to the Gemini prompt (or shown in the UI for transparency).

RAG works **100% locally** – no external calls for retrieval.

---

## 📊 Dataset Schema

| Column | Type | Description |
|--------|------|-------------|
| `float_id` | string | WMO float identifier (e.g. `WMO_1234567`) |
| `latitude` | float | Decimal degrees N, −90 to +90 |
| `longitude` | float | Decimal degrees E, −180 to +180 |
| `date` | string | Observation date (YYYY-MM-DD) |
| `temperature` | float | In-situ temperature (°C) |
| `salinity` | float | Practical salinity (PSU) |
| `pressure` | float | Pressure (dbar) |
| `depth` | float | Approximate depth (m) |
| `region` | string | Named ocean region |
| `qc_flag` | int | Argo QC flag (1=good … 4=bad) |

---

## 🧪 Evaluation

The evaluation framework runs **32 natural-language test questions** and measures:

| Metric | Description |
|--------|-------------|
| Intent accuracy | Correct operation extracted |
| Variable accuracy | Correct scientific variable identified |
| Region accuracy | Correct ocean region extracted |
| Filter accuracy | Date/depth/QC filters correctly applied |
| Plan accuracy | All of the above correct together |
| RAG accuracy | Relevant docs retrieved for domain questions |

Run evaluation from the **🧪 Evaluation** tab in the Streamlit UI, or:

```python
from evaluation.evaluator import Evaluator
from ai.pipeline import AIPipeline
import pandas as pd

df = pd.read_csv("data/sample_argo_data.csv")
pipe = AIPipeline()
report = Evaluator(pipe, df).run()
print(f"Plan accuracy: {report.plan_accuracy:.0%}")
```

---

## 🔒 Security

- **No `eval()` or `exec()`** – all query operations are explicit Pandas method calls.
- **Allowlist validation** – every LLM-generated plan passes through `query_validator.py` before execution.
- **API key isolation** – credentials are environment variables only; never hard-coded.

---

## 📝 Supported Query Operations

| Operation | Example question |
|-----------|-----------------|
| `avg` | "What is the average temperature?" |
| `max` | "Find the warmest observation in the Caribbean" |
| `min` | "What is the coldest temperature?" |
| `count` | "How many observations are there?" |
| `sum` | "Total depth across all floats" |
| `median` | "Median salinity in the North Atlantic" |
| `select` | "Show me data from the Southern Ocean" |
| `compare` | "Compare temperature across all regions" |

Filters: **region**, **date range**, **depth range**, **QC flag**, **limit/top-N**  
Group by: **region**, **year**, **month**, **qc_flag**

---

## ⚠️ Limitations

1. Dataset is synthetic – not suitable for real scientific analysis.
2. Rule-based parser may mis-classify complex multi-clause questions.
3. FAISS index is rebuilt in memory on each application start (no persistence); this adds ~10 s startup time.
4. Gemini integration requires a valid API key with the Generative AI API enabled.
5. No authentication – intended for local / demo use only.

---

## 🏷️ Technology Stack

| Component | Library |
|-----------|---------|
| Dashboard | Streamlit ≥ 1.32 |
| LLM | Google Gemini (google-generativeai ≥ 0.5) |
| Embeddings | Sentence Transformers ≥ 2.6 |
| Vector DB | FAISS-CPU ≥ 1.8 |
| Data processing | Pandas ≥ 2.1, NumPy ≥ 1.26 |
| Visualisation | Plotly ≥ 5.19 |
| Testing | Pytest ≥ 8.1 |

---

*Built as an educational AI/ML portfolio project demonstrating RAG, LLM integration, safe query execution, and Streamlit dashboards.*

