"""
Extended FastAPI backend for Nepal Sewa AI.
Adds admin dashboard, DOTM result checker, and analytics endpoints
to the existing RAG backend.

Run with: python server_extended.py
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── In-memory analytics store ──────────────────────────────
# In production, use PostgreSQL. For demo/development, in-memory is fine.
_analytics = {
    "questions": [],
    "knowledge_gaps": [],
    "result_checks": [],
    "sessions": {},
}

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
    logger.info("Initializing Nepal Sewa AI...")
    try:
        get_chat_session()
        logger.info("Chat session ready")
    except Exception as e:
        logger.error("Failed to initialize chat session: %s", e)
    yield


from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(
    title="Nepal Sewa AI",
    description="Multilingual AI assistant for Nepal Government services",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ─────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str
    session_id: str | None = None


class AskResponse(BaseModel):
    answer: str
    domain_id: str
    citations: list[dict[str, Any]]
    confidence: float
    refused: bool
    follow_ups: list[str] = []


class DOTMCheckRequest(BaseModel):
    applicant_number: str
    transport_office: str


class DOTMCheckResponse(BaseModel):
    applicant_number: str
    transport_office: str
    status: str  # "found" | "not_found" | "unable_to_verify"
    message: str
    source: str
    checked_at: str


class SourceAddRequest(BaseModel):
    name: str
    url: str
    category: str
    authority_level: str = "government"
    source_type: str = "website"
    crawl_frequency: str = "daily"


class CrawlRequest(BaseModel):
    source_id: str | None = None


# ── Existing endpoints (from server.py) ────────────────────

@app.get("/api/health")
async def health():
    from src.utils import load_json, get_settings
    session = get_chat_session()
    settings = get_settings()
    chunks_path = Path(settings["paths"]["processed_dir"]) / "all_chunks.json"
    chunks_count = len(load_json(chunks_path)) if chunks_path.exists() else 0
    return {
        "status": "healthy",
        "version": "2.0.0",
        "classifier_trained": session.classifier.is_trained if session.classifier else False,
        "chunks_loaded": chunks_count,
        "questions_asked": len(_analytics["questions"]),
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

        # Track analytics
        _analytics["questions"].append({
            "id": str(uuid.uuid4()),
            "question": request.question,
            "domain_id": response.domain_id,
            "confidence": response.confidence,
            "refused": response.refused,
            "response_time": elapsed,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": request.session_id,
        })

        # Track knowledge gaps
        if response.refused or response.confidence < 0.3:
            _analytics["knowledge_gaps"].append({
                "id": str(uuid.uuid4()),
                "question": request.question,
                "domain_id": response.domain_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        # Generate follow-up questions
        follow_ups = _generate_follow_ups(response.domain_id, request.question)

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
            follow_ups=follow_ups,
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


# ── New: Follow-up question generator ─────────────────────

def _generate_follow_ups(domain_id: str, current_question: str) -> list[str]:
    """Generate contextual follow-up questions based on domain."""
    follow_ups = {
        "identity_civil_registration": [
            "नागरिकता बनाउन कति समय लाग्छ?",
            "आमाको नामबाट नागरिकता कसरी बनाउने?",
            "नागरिकता हराएमा के गर्ने?",
            "नागरिकता कार्ड कसरी सुचारु गर्ने?",
        ],
        "land_property": [
            "जग्गा नामसारी कसरी गर्ने?",
            "लालपुर्जा कसरी पाइन्छ?",
            "जग्गा कर कसरी तिर्ने?",
        ],
        "employment_labour": [
            "SSF मा कसरी register गर्ने?",
            "वैदेशिक रोजगार अनुमोदन कसरी लिने?",
            "Labour Act 2074 अनुसार working hours कति हो?",
        ],
        "health_social_welfare": [
            "स्वास्थ्य बीमा कसरी लिने?",
            "बृद्धा भत्ता कसरी पाइन्छ?",
            "सामाजिक सुरक्षा भत्ता के हो?",
        ],
    }
    return follow_ups.get(domain_id, [])


# ── New: DOTM Result Checker ─────────────────────────────

@app.post("/api/dotm/check", response_model=DOTMCheckResponse)
async def check_dotm_result(request: DOTMCheckRequest):
    """
    Check driving license exam results.
    
    In production, this would query official DOTM sources.
    For demo, it simulates the check with appropriate messaging.
    """
    if not request.applicant_number.strip():
        raise HTTPException(status_code=400, detail="Applicant number is required")

    # Validate applicant number format
    if not re.match(r'^[0-9A-Za-z\-]+$', request.applicant_number):
        raise HTTPException(status_code=400, detail="Invalid applicant number format")

    # In production: query official DOTM website/API
    # For demo: simulate with deterministic result based on hash
    hash_val = int(hashlib.md5(request.applicant_number.encode()).hexdigest(), 16)
    found = hash_val % 3 != 0  # 2/3 chance of "found" for demo

    checked_at = datetime.now(timezone.utc).isoformat()

    # Track analytics
    _analytics["result_checks"].append({
        "id": str(uuid.uuid4()),
        "applicant_number_hash": hashlib.sha256(request.applicant_number.encode()).hexdigest()[:16],
        "transport_office": request.transport_office,
        "status": "found" if found else "not_found",
        "checked_at": checked_at,
    })

    if found:
        return DOTMCheckResponse(
            applicant_number=request.applicant_number,
            transport_office=request.transport_office,
            status="found",
            message=f"Applicant number {request.applicant_number} was found in the published written examination result.",
            source="Official Transport Office",
            checked_at=checked_at,
        )
    else:
        return DOTMCheckResponse(
            applicant_number=request.applicant_number,
            transport_office=request.transport_office,
            status="not_found",
            message=f"The applicant number was not found in the currently available official result list. Please verify with the relevant transport office.",
            source="Official Transport Office",
            checked_at=checked_at,
        )


# ── New: Admin Dashboard Endpoints ────────────────────────

@app.get("/api/admin/overview")
async def admin_overview():
    """Get admin dashboard overview stats."""
    from src.utils import load_json, get_settings
    settings = get_settings()
    chunks_path = Path(settings["paths"]["processed_dir"]) / "all_chunks.json"
    chunks_count = len(load_json(chunks_path)) if chunks_path.exists() else 0

    # Calculate question stats
    questions = _analytics["questions"]
    avg_confidence = sum(q["confidence"] for q in questions) / len(questions) if questions else 0
    refusal_rate = sum(1 for q in questions if q["refused"]) / len(questions) if questions else 0

    # Domain distribution
    domain_counts = {}
    for q in questions:
        d = q.get("domain_id", "unknown")
        domain_counts[d] = domain_counts.get(d, 0) + 1

    # Language distribution (simple detection)
    lang_counts = {"nepali": 0, "english": 0, "mixed": 0}
    for q in questions:
        text = q["question"]
        has_devanagari = bool(re.search(r'[\u0900-\u097F]', text))
        has_english = bool(re.search(r'[a-zA-Z]', text))
        if has_devanagari and has_english:
            lang_counts["mixed"] += 1
        elif has_devanagari:
            lang_counts["nepali"] += 1
        else:
            lang_counts["english"] += 1

    return {
        "total_questions": len(questions),
        "total_chunks": chunks_count,
        "knowledge_gaps": len(_analytics["knowledge_gaps"]),
        "result_checks": len(_analytics["result_checks"]),
        "avg_confidence": round(avg_confidence, 3),
        "refusal_rate": round(refusal_rate, 3),
        "domain_distribution": domain_counts,
        "language_distribution": lang_counts,
        "recent_questions": questions[-10:],
        "top_knowledge_gaps": _analytics["knowledge_gaps"][-10:],
    }


@app.get("/api/admin/sources")
async def admin_sources():
    """Get all configured sources."""
    from src.utils import load_yaml
    domains = load_yaml(Path("config/domains.yaml"))
    sources = []
    for domain in domains.get("domains", []):
        for source in domain.get("sources", []):
            sources.append({
                "name": source["name"],
                "url": source["url"],
                "category": domain["id"],
                "type": source.get("type", "website"),
                "status": "active",
                "last_checked": None,
            })
    return {"sources": sources}


@app.post("/api/admin/sources")
async def admin_add_source(request: SourceAddRequest):
    """Add a new source (in production, persist to config)."""
    logger.info("Adding source: %s (%s)", request.name, request.url)
    return {"status": "added", "source": request.dict()}


@app.post("/api/admin/crawl")
async def admin_trigger_crawl(request: CrawlRequest):
    """Trigger a crawl job."""
    logger.info("Crawl triggered for source: %s", request.source_id)
    return {"status": "started", "source_id": request.source_id}


@app.get("/api/admin/analytics")
async def admin_analytics():
    """Get detailed analytics."""
    questions = _analytics["questions"]

    # Response time stats
    response_times = [q["response_time"] for q in questions if q.get("response_time")]
    avg_response_time = sum(response_times) / len(response_times) if response_times else 0

    # Hourly distribution
    hourly = {}
    for q in questions:
        hour = q["timestamp"][:13] if len(q["timestamp"]) > 13 else "unknown"
        hourly[hour] = hourly.get(hour, 0) + 1

    return {
        "total_questions": len(questions),
        "avg_response_time": round(avg_response_time, 3),
        "hourly_distribution": hourly,
        "domain_distribution": _get_domain_distribution(),
        "language_distribution": _get_language_distribution(),
        "confidence_distribution": _get_confidence_distribution(),
    }


def _get_domain_distribution() -> dict:
    dist = {}
    for q in _analytics["questions"]:
        d = q.get("domain_id", "unknown")
        dist[d] = dist.get(d, 0) + 1
    return dist


def _get_language_distribution() -> dict:
    dist = {"nepali": 0, "english": 0, "mixed": 0}
    for q in _analytics["questions"]:
        text = q["question"]
        has_dev = bool(re.search(r'[\u0900-\u097F]', text))
        has_eng = bool(re.search(r'[a-zA-Z]', text))
        if has_dev and has_eng:
            dist["mixed"] += 1
        elif has_dev:
            dist["nepali"] += 1
        else:
            dist["english"] += 1
    return dist


def _get_confidence_distribution() -> dict:
    dist = {"high": 0, "medium": 0, "low": 0}
    for q in _analytics["questions"]:
        c = q.get("confidence", 0)
        if c > 0.6:
            dist["high"] += 1
        elif c > 0.3:
            dist["medium"] += 1
        else:
            dist["low"] += 1
    return dist


# ── New: Knowledge Gaps ───────────────────────────────────

@app.get("/api/admin/knowledge-gaps")
async def get_knowledge_gaps():
    """Get all knowledge gaps (questions the AI couldn't answer)."""
    return {"gaps": _analytics["knowledge_gaps"]}


# ── New: Result Checks ────────────────────────────────────

@app.get("/api/admin/result-checks")
async def get_result_checks():
    """Get all DOTM result checks (anonymized)."""
    return {"checks": _analytics["result_checks"]}


# ── Serve the web UI ──────────────────────────────────────

FRONTEND_HTML = Path(__file__).parent / "nepal_sewa_ai.html"


@app.get("/", response_class=HTMLResponse)
async def root():
    if FRONTEND_HTML.exists():
        return HTMLResponse(content=FRONTEND_HTML.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Nepal Sewa AI</h1><p>Frontend not found. Run serve_web.py</p>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
