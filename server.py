"""Production FastAPI backend for Nepal RAG Agent."""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

_chat_session = None


def get_chat_session():
    """Get or initialize the chat session."""
    global _chat_session
    if _chat_session is None:
        from src.generation.chat import ChatSession
        _chat_session = ChatSession(
            provider=os.getenv("LLM_PROVIDER", "openai"),
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        )
    return _chat_session


@asynccontextmanager
async def lifespan(app):
    """Startup: initialize chat session."""
    logger.info("Initializing Nepal RAG Agent...")
    try:
        get_chat_session()
        logger.info("Chat session ready")
    except Exception as e:
        logger.error("Failed to initialize chat session: %s", e)
    yield


from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(
    title="Nepal Government Services RAG",
    description="Ask questions about Nepal government public services",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    domain_id: str
    citations: list[dict[str, Any]]
    confidence: float
    refused: bool


@app.get("/api/health")
async def health():
    from src.utils import load_json, get_settings
    session = get_chat_session()
    settings = get_settings()
    chunks_path = Path(settings["paths"]["processed_dir"]) / "all_chunks.json"
    chunks_count = len(load_json(chunks_path)) if chunks_path.exists() else 0
    return {
        "status": "healthy",
        "version": "1.0.0",
        "classifier_trained": session.classifier.is_trained if session.classifier else False,
        "chunks_loaded": chunks_count,
    }


@app.post("/api/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    start_time = time.time()
    session = get_chat_session()
    
    try:
        response = session.ask(request.question)
        elapsed = time.time() - start_time
        
        logger.info(
            "Q: %.50s... | Domain: %s | Time: %.2fs",
            request.question,
            response.domain_id,
            elapsed,
        )
        
        return AskResponse(
            answer=response.answer,
            domain_id=response.domain_id,
            citations=response.citations,
            confidence=response.confidence,
            refused=response.refused,
        )
    except Exception as e:
        logger.error("Error processing question: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats")
async def stats():
    from src.utils import load_json, get_settings
    settings = get_settings()
    chunks_path = Path(settings["paths"]["processed_dir"]) / "all_chunks.json"
    
    if not chunks_path.exists():
        return {"total_chunks": 0, "total_domains": 0, "sources": []}
    
    chunks = load_json(chunks_path)
    domains = set(c.get("domain_id", "") for c in chunks)
    sources = {}
    for c in chunks:
        key = c.get("source_name", "")
        if key not in sources:
            sources[key] = {
                "name": key,
                "url": c.get("source_url", ""),
                "domain_id": c.get("domain_id", ""),
                "chunks": 0,
            }
        sources[key]["chunks"] += 1
    
    return {
        "total_chunks": len(chunks),
        "total_domains": len(domains),
        "sources": list(sources.values()),
    }


FRONTEND_HTML = """<!DOCTYPE html>
<html lang="ne">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Nepal Government Services RAG</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Noto Sans Devanagari', system-ui, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e0e0e0;
            min-height: 100vh;
        }
        .container { max-width: 900px; margin: 0 auto; padding: 2rem 1rem; }
        header { text-align: center; margin-bottom: 2rem; }
        header h1 { font-size: 1.8rem; color: #fff; margin-bottom: 0.5rem; }
        header p { color: #a0a0a0; }
        .chat-box {
            background: #1e293b;
            border-radius: 12px;
            padding: 1.5rem;
            min-height: 300px;
            max-height: 50vh;
            overflow-y: auto;
            margin-bottom: 1rem;
        }
        .msg { padding: 0.75rem 1rem; margin-bottom: 0.5rem; border-radius: 8px; line-height: 1.6; }
        .msg.user { background: #334155; margin-left: 2rem; border-left: 3px solid #3b82f6; }
        .msg.bot { background: #0f172a; margin-right: 2rem; border-left: 3px solid #10b981; white-space: pre-wrap; }
        .msg.error { background: #7f1d1d; }
        .input-row { display: flex; gap: 0.5rem; }
        textarea {
            flex: 1; padding: 0.75rem; border-radius: 8px; border: 2px solid #475569;
            background: #1e293b; color: #e0e0e0; font-size: 1rem; resize: vertical; min-height: 50px;
        }
        textarea:focus { border-color: #3b82f6; outline: none; }
        button {
            padding: 0.5rem 1.25rem; border: none; border-radius: 8px;
            background: #3b82f6; color: white; font-size: 1rem; cursor: pointer;
        }
        button:hover { background: #2563eb; }
        button:disabled { background: #475569; cursor: not-allowed; }
        .examples { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 1rem; }
        .ex-btn {
            background: #334155; color: #94a3b8; padding: 0.4rem 0.8rem;
            border-radius: 20px; border: 1px solid #475569; cursor: pointer; font-size: 0.85rem;
        }
        .ex-btn:hover { background: #475569; color: #e0e0e0; }
        .stats { text-align: center; padding: 1rem; color: #64748b; font-size: 0.85rem; }
        .loading { display: inline-block; width: 10px; height: 10px; border: 2px solid #e0e0e0; border-top: 2px solid #3b82f6; border-radius: 50%; animation: spin 1s linear infinite; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .src-link { color: #60a5fa; text-decoration: none; font-size: 0.85rem; }
        .src-link:hover { text-decoration: underline; }
        .domain-badge { display: inline-block; padding: 0.15rem 0.5rem; background: #1e40af; color: #bfdbfe; border-radius: 10px; font-size: 0.7rem; margin-left: 0.5rem; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Nepal Government Services RAG</h1>
            <p>नेपाल सरकारी सेवाहरूको बारेमा प्रश्न सोध्नुहोस्</p>
        </header>
        <div class="chat-box" id="chat"></div>
        <div class="examples">
            <button class="ex-btn" onclick="ask('नागरिकता कार्ड कसरी लिन सकिन्छ?')">नागरिकता कार्ड</button>
            <button class="ex-btn" onclick="ask('जग्गा नामसारी कसरी गर्ने?')">जग्गा नामसारी</button>
            <button class="ex-btn" onclick="ask('वैदेशिक रोजगार अनुमोदन कसरी लिने?')">वैदेशिक रोजगार</button>
            <button class="ex-btn" onclick="ask('स्वास्थ्य बीमा कसरी लिने?')">स्वास्थ्य बीमा</button>
            <button class="ex-btn" onclick="ask('बृद्धा भत्ता कसरी पाइन्छ?')">बृद्धा भत्ता</button>
            <button class="ex-btn" onclick="ask('SSF मा दर्ता कसरी गर्ने?')">SSF दर्ता</button>
        </div>
        <div class="input-row">
            <textarea id="q" placeholder="यहाँ आफ्नो प्रश्न लेख्नुहोस्..." rows="2"></textarea>
            <button id="btn" onclick="ask()">प्रश्न सोध्नुहोस्</button>
        </div>
        <div class="stats" id="stats"></div>
    </div>
    <script>
        async function loadStats() {
            try {
                const r = await fetch('/api/stats');
                const d = await r.json();
                document.getElementById('stats').textContent = `Chunks: ${d.total_chunks} | Domains: ${d.total_domains}`;
            } catch(e) {}
        }
        
        function addMsg(role, text) {
            const d = document.createElement('div');
            d.className = 'msg ' + role;
            d.textContent = text;
            document.getElementById('chat').appendChild(d);
            document.getElementById('chat').scrollTop = document.getElementById('chat').scrollHeight;
        }
        
        async function ask(q) {
            const input = document.getElementById('q');
            const question = q || input.value.trim();
            if (!question) return;
            
            addMsg('user', question);
            if (!q) input.value = '';
            
            const btn = document.getElementById('btn');
            btn.disabled = true;
            btn.innerHTML = '<span class="loading"></span>';
            
            try {
                const res = await fetch('/api/ask', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({question}),
                });
                
                if (!res.ok) {
                    throw new Error('Server error: ' + res.status);
                }
                
                const data = await res.json();
                
                // Create message div
                const d = document.createElement('div');
                d.className = 'msg bot';
                
                // Add answer text
                const answerDiv = document.createElement('div');
                answerDiv.textContent = data.answer;
                answerDiv.style.whiteSpace = 'pre-wrap';
                d.appendChild(answerDiv);
                
                // Add domain badge
                if (data.domain_id) {
                    const badge = document.createElement('span');
                    badge.className = 'domain-badge';
                    badge.textContent = data.domain_id;
                    d.appendChild(badge);
                }
                
                // Add citations
                if (data.citations && data.citations.length > 0) {
                    const srcDiv = document.createElement('div');
                    srcDiv.style.marginTop = '0.75rem';
                    srcDiv.style.fontSize = '0.85rem';
                    srcDiv.style.color = '#94a3b8';
                    
                    const strong = document.createElement('strong');
                    strong.textContent = 'स्रोत / Sources:';
                    srcDiv.appendChild(strong);
                    srcDiv.appendChild(document.createElement('br'));
                    
                    data.citations.forEach((c, i) => {
                        const link = document.createElement('a');
                        link.className = 'src-link';
                        link.href = c.url;
                        link.target = '_blank';
                        link.textContent = `[${i+1}] ${c.name}`;
                        srcDiv.appendChild(link);
                        srcDiv.appendChild(document.createElement('br'));
                    });
                    
                    d.appendChild(srcDiv);
                }
                
                document.getElementById('chat').appendChild(d);
                document.getElementById('chat').scrollTop = document.getElementById('chat').scrollHeight;
            } catch(e) {
                addMsg('error', 'Error: ' + e.message);
            }
            
            btn.disabled = false;
            btn.textContent = 'प्रश्न सोध्नुहोस्';
        }
        
        document.getElementById('q').addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); ask(); }
        });
        
        loadStats();
    </script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(content=FRONTEND_HTML)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
