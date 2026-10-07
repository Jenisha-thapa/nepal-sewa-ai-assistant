# Domain-Aware RAG Agent for Nepal Government Public Services

A citation-grounded, domain-aware Retrieval-Augmented Generation (RAG) agent
that answers Nepali-language questions about four public-service domains,
grounded in official government sources.

## Features

- **4 Domains**: Identity & Civil Registration, Land & Property, Employment & Labour, Health & Social Welfare
- **Domain Classification**: Embedding-based classifier with keyword baseline comparison
- **3 Retrieval Modes**: BM25, Dense (vector), and Hybrid retrieval
- **Structure-Aware Chunking**: Preserves FAQ pairs, sections, and headings
- **Citation-Grounded Generation**: Evidence-only answers with source citations
- **Refusal Mechanism**: Declines to answer when evidence is insufficient
- **100-Question Eval Set**: 25 questions per domain with gold annotations
- **Gradio Web Interface**: Interactive demo with Nepali/English support
- **Comprehensive Metrics**: Domain accuracy, retrieval P@K/R@K/MRR, citation correctness

## Architecture

```
Query → Domain Classifier → Retrieval (BM25/Dense/Hybrid) → Citation-Grounded Generation → Answer + Sources
```

## Quick Start

### 1. Install

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure

Copy `.env.example` to `.env` and add your API key:

```bash
cp .env.example .env
# Edit .env and add: OPENROUTER_API_KEY=your_key_here
```

### 3. Run the Pipeline

```bash
# Phase 1: Fetch government sources
python -m src.ingestion.fetch_sources

# Phase 2-3: Process and chunk documents
python -m src.main process --verbose

# Phase 4: Train domain classifier
python -m src.main classify

# Phase 5: Build retrieval index
python -m src.main index

# Phase 6: Ask a question
python -m src.main ask "नागरिकता कार्ड कसरी लिन सकिन्छ?"

# Interactive chat
python -m src.main chat

# Launch web interface
python -m src.main serve

# Run evaluation
python -m src.main evaluate
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `fetch` | Fetch government sources (Phase 1) |
| `process` | Process and chunk documents (Phase 2-3) |
| `index` | Build retrieval index (Phase 4-5) |
| `classify` | Train domain classifier |
| `chat` | Interactive chat session |
| `ask` | Single question-answer |
| `evaluate` | Run evaluation metrics |
| `serve` | Launch Gradio web interface |

## Project Structure

```
nepal-rag-agent/
├── config/
│   ├── domains.yaml          # Domain and source configuration
│   └── settings.yaml         # All project settings
├── data/
│   ├── raw/                  # Fetched government documents
│   ├── processed/            # Cleaned chunks
│   ├── eval/                 # Evaluation dataset
│   └── index/                # Vector store and classifier
├── src/
│   ├── ingestion/            # Phase 1: Source acquisition
│   ├── processing/           # Phase 2: Document cleaning
│   ├── chunking/             # Phase 3: Structure-aware chunking
│   ├── domain_classifier/    # Phase 4: Domain routing
│   ├── retrieval/            # Phase 5: BM25/Dense/Hybrid
│   ├── generation/           # Phase 6: Citation-grounded LLM
│   ├── evaluation/           # Phase 7: Metrics
│   ├── app/                  # Phase 8: Gradio interface
│   ├── utils.py              # Shared utilities
│   └── main.py               # CLI entry point
├── tests/                    # Unit tests
├── scripts/                  # Utility scripts
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Configuration

Edit `config/settings.yaml` to customize:
- Retrieval mode (bm25/dense/hybrid)
- Embedding model
- LLM provider and model
- Chunking parameters
- Evaluation settings

## Evaluation

The evaluation set contains 100 questions (25 per domain):

```bash
# Generate eval set
python scripts/generate_eval_set.py

# Run evaluation
python -m src.main evaluate --output results/eval.json
```

Metrics reported:
- **Domain Classification**: Accuracy, Precision, Recall, F1
- **Retrieval**: P@1, P@3, P@5, R@1, R@3, R@5, MRR
- **Generation**: Refusal rate, Citation rate

## Domains

1. **Identity & Civil Registration** — DoNIDCR
   - National ID card, birth/death/marriage registration
2. **Land & Property Services** — Department of Land Management and Archive
   - Land registration, ownership transfer (नामसारी)
3. **Employment & Labour Services** — DoFE / Social Security Fund
   - Foreign employment, SSF registration
4. **Health & Social Welfare Services** — HIB / DoNIDCR
   - Health insurance, senior citizen allowance

## Tech Stack

- **Language**: Python 3.10+
- **Embeddings**: sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2)
- **Vector Store**: ChromaDB
- **Retrieval**: BM25 (rank-bm25) + Dense (cosine similarity)
- **LLM**: OpenAI / Anthropic / Ollama / Openrouter
- **Evaluation**: scikit-learn
- **Interface**: Gradio

## License

MIT
