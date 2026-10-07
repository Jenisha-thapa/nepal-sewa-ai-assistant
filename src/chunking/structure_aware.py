"""Structure-aware chunking for Nepali/English government documents."""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.utils import get_settings, load_json, save_json, find_repo_root, count_tokens_approx

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A single chunk with metadata for citation."""
    chunk_id: str
    domain_id: str
    source_name: str
    source_url: str
    document_type: str
    title: str
    section: str
    text: str
    position: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    language: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "domain_id": self.domain_id,
            "source_name": self.source_name,
            "source_url": self.source_url,
            "document_type": self.document_type,
            "title": self.title,
            "section": self.section,
            "text": self.text,
            "position": self.position,
            "metadata": self.metadata,
            "language": self.language,
        }


def _make_chunk_id(source_name: str, text: str, position: int) -> str:
    """Create a deterministic chunk ID."""
    h = hashlib.md5(f"{source_name}:{text[:100]}:{position}".encode("utf-8")).hexdigest()[:10]
    safe_name = re.sub(r"[^\w]", "_", source_name)[:40]
    return f"chunk_{safe_name}_{position}_{h}"


def _detect_sections(text: str) -> list[tuple[str, str]]:
    """Detect sections using FAQ patterns."""
    sections = []
    
    # Pattern 1: Numbered FAQ (e.g., "१ . प्रश्न:" or "1. प्रश्न:" or "1.\tQuestion")
    nepali_num = r"[०-९\d]"
    
    # Try numbered questions with प्रश्न keyword
    pattern1 = re.compile(
        rf"^[\s]*({nepali_num}+[\.\)\s]+[^\n]*(?:प्रश्न|question)[^\n]*(?:\?|$))",
        re.MULTILINE | re.IGNORECASE,
    )
    
    # Try numbered questions (e.g., "1.\tQuestion text" or "१. Question text")
    pattern2 = re.compile(
        rf"^[\s]*({nepali_num}+[\.\)\s\t]+[^\n]+(?:\?|\s*$))",
        re.MULTILINE,
    )
    
    for pattern in [pattern1, pattern2]:
        matches = list(pattern.finditer(text))
        if len(matches) >= 3:
            for i, match in enumerate(matches):
                start = match.start()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                block = text[start:end].strip()
                if block and len(block) > 20:
                    heading = block[:80].rsplit('\n', 1)[0]
                    sections.append((heading, block))
            if sections:
                return sections
    
    # Pattern 2: Markdown headings
    heading_pattern = re.compile(r"^#{1,3}\s+(.+)$", re.MULTILINE)
    parts = heading_pattern.split(text)
    if len(parts) > 2:
        current = "Introduction"
        for i, part in enumerate(parts):
            if i == 0:
                if part.strip():
                    sections.append((current, part))
            elif i % 2 == 1:
                current = part.strip()
            else:
                sections.append((current, part))
        return sections
    
    # No structure - return as single section
    if text.strip():
        sections.append(("Full Document", text))
    return sections


def _split_long_text(
    text: str,
    max_tokens: int = 500,
    overlap_tokens: int = 50,
) -> list[str]:
    """Split long text into overlapping chunks."""
    chunks = []
    
    # Split by paragraphs or sentence-ending punctuation
    paragraphs = re.split(r"\n{2,}|[।।]\s+|\.\s+(?=[A-Z\u0900-\u097F])", text)
    
    current_chunk = ""
    for para in paragraphs:
        para = para.strip()
        if not para or len(para) < 20:
            continue
        
        test_chunk = f"{current_chunk} {para}".strip() if current_chunk else para
        if count_tokens_approx(test_chunk) <= max_tokens:
            current_chunk = test_chunk
        else:
            if current_chunk:
                chunks.append(current_chunk)
                words = current_chunk.split()
                overlap_text = " ".join(words[-overlap_tokens:]) if len(words) > overlap_tokens else current_chunk
                current_chunk = f"{overlap_text} {para}".strip()
            else:
                current_chunk = para
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks


def chunk_document(
    processed_doc: dict[str, Any],
    max_tokens: int = 500,
    overlap_tokens: int = 50,
) -> list[Chunk]:
    """Chunk a single document."""
    text = processed_doc.get("cleaned_text", "")
    source_name = processed_doc.get("source_name", "Unknown")
    domain_id = processed_doc.get("domain_id", "unknown")
    source_url = processed_doc.get("source_url", "")
    document_type = processed_doc.get("document_type", "unknown")
    title = processed_doc.get("title", "")
    language = processed_doc.get("language", "unknown")
    metadata = processed_doc.get("metadata", {})
    
    if not text or len(text) < 100:
        return []
    
    sections = _detect_sections(text)
    chunks: list[Chunk] = []
    position = 0
    
    for section_heading, section_text in sections:
        if len(section_text) < 50:
            continue
        
        text_chunks = _split_long_text(section_text, max_tokens, overlap_tokens)
        
        if not text_chunks:
            if len(section_text) >= 100:
                text_chunks = [section_text]
        
        for chunk_text in text_chunks:
            chunk_text = chunk_text.strip()
            if not chunk_text or len(chunk_text) < 100:
                continue
            
            position += 1
            chunk = Chunk(
                chunk_id=_make_chunk_id(source_name, chunk_text, position),
                domain_id=domain_id,
                source_name=source_name,
                source_url=source_url,
                document_type=document_type,
                title=title,
                section=section_heading,
                text=chunk_text,
                position=position,
                metadata=metadata,
                language=language,
            )
            chunks.append(chunk)
    
    logger.debug("Chunked '%s': %d sections -> %d chunks", source_name, len(sections), len(chunks))
    return chunks


def chunk_all_processed(
    domain_id: str | None = None,
    max_tokens: int = 500,
    overlap_tokens: int = 50,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """Chunk all processed documents."""
    settings = get_settings()
    processed_dir = Path(settings["paths"]["processed_dir"])
    
    if not processed_dir.exists():
        logger.warning("Processed directory not found: %s", processed_dir)
        return []
    
    all_chunks: list[Chunk] = []
    
    # Find all processed JSON files (exclude all_chunks.json)
    json_files = [f for f in processed_dir.rglob("*.json") if f.name != "all_chunks.json"]
    
    for json_path in json_files:
        try:
            doc = load_json(json_path)
            if domain_id and doc.get("domain_id") != domain_id:
                continue
            
            chunks = chunk_document(doc, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
            all_chunks.extend(chunks)
            
            if verbose and chunks:
                print(f"  Chunked: {doc.get('source_name', json_path.name)} -> {len(chunks)} chunks")
        except Exception as e:
            logger.error("Error chunking %s: %s", json_path, e)
    
    logger.info("Total chunks created: %d", len(all_chunks))
    
    # Save all chunks
    chunks_data = [c.to_dict() for c in all_chunks]
    chunks_path = processed_dir / "all_chunks.json"
    save_json(chunks_data, chunks_path)
    
    return chunks_data
