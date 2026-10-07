# 🇳🇵 Nepal Sewa AI — Setup & Run Guide

## Quick Start (3 Steps)

### 1. Start the Backend (FastAPI)

Open a terminal in VS Code (`Ctrl+``) and run:

```bash
cd C:\Users\jenisha.thapa\Downloads\nepal-rag-agent
.venv\Scripts\python.exe server.py
```

The backend will start on `http://localhost:8000`

### 2. Start the Web Frontend

Open **another** terminal tab and run:

```bash
cd C:\Users\jenisha.thapa\Downloads\nepal-rag-agent
.venv\Scripts\python.exe serve_web.py --port 3000
```

### 3. Open in Browser

Go to: **http://localhost:3000**

---

## Alternative: One-Click Start (Windows Batch File)

Double-click `start.bat` to start both servers automatically.

---

## What's Included

| File | Purpose |
|------|---------|
| `nepal_sewa_ai.html` | Main web UI (single-file, no build step) |
| `serve_web.py` | Lightweight HTTP server + API proxy |
| `start.bat` | One-click launcher for Windows |
| `server.py` | Existing FastAPI backend (RAG API) |

---

## Features

### ✅ Chat Interface
- Modern, ChatGPT-like UI with Nepali identity
- Supports Nepali, English, Hindi, Romanized Nepali
- Real-time typing indicators
- Follow-up question suggestions
- Source citations with clickable links
- Confidence badges (High/Medium/Low)
- Domain classification badges

### ✅ Category Navigation
- 🏛 Citizenship (नागरिकता)
- 🪪 National ID (राष्ट्रिय परिचयपत्र)
- 🚗 DOTM/Transport (यातायात व्यवस्था)
- 👷 SSF (सामाजिक सुरक्षा कोष)
- ⚖ Labour Law (श्रम कानून)

### ✅ FAQ Cards
- 18 pre-built common questions
- Click to ask instantly
- Auto-filtered by selected category

### ✅ Admin Dashboard
- System stats (sources, chunks, questions, domains)
- Source management view
- Backend health status
- Question count tracking

### ✅ DOTM Result Checker
- Check driving license exam results
- Applicant number lookup
- Transport office selection
- Privacy-conscious (no permanent storage)

### ✅ Mobile Responsive
- Collapsible sidebar
- Touch-friendly buttons
- Adaptive layout

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Backend health check |
| `/api/ask` | POST | Ask a question |
| `/api/stats` | GET | System statistics |

---

## Project Structure

```
nepal-rag-agent/
├── nepal_sewa_ai.html      # ← New web UI
├── serve_web.py            # ← New web server
├── start.bat               # ← New launcher
├── server.py               # Existing FastAPI backend
├── src/                    # Existing RAG pipeline
│   ├── main.py            # CLI entry point
│   ├── generation/chat.py # Chat session
│   ├── retrieval/         # BM25 + Dense retrieval
│   ├── domain_classifier/ # Domain routing
│   ├── ingestion/         # Source fetching
│   ├── processing/        # Document processing
│   ├── chunking/          # Structure-aware chunking
│   └── evaluation/        # Metrics
├── config/
│   ├── settings.yaml      # Project settings
│   └── domains.yaml       # Domain configuration
├── data/
│   ├── raw/               # Fetched documents
│   ├── processed/         # Cleaned chunks
│   └── index/             # Vector store
└── requirements.txt
```

---

## Environment Variables

Create a `.env` file (see `.env.example`):

```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=your-key-here
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Backend not reachable" | Start `server.py` first |
| Port 8000 in use | Change port: `python server.py --port 8001` |
| Port 3000 in use | Change port: `python serve_web.py --port 3001` |
| CORS errors | Backend already has CORS enabled |
| Empty answers | Run ingestion: `python -m src.main process` |

---

## Next Steps (Optional)

1. **Add more FAQ questions** — Edit `DEFAULT_TOPICS` in `nepal_sewa_ai.html`
2. **Connect real DOTM results** — Implement actual API calls in `checkDOTMResult()`
3. **Add authentication** — Extend the admin panel with login
4. **Deploy** — Use Docker or a cloud platform

---

## Stopping the Servers

Press `Ctrl+C` in each terminal window, or close the terminal tabs.
