"""Phase 2: Document processing — clean HTML/PDF, OCR fallback, Unicode normalization."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
from bs4 import BeautifulSoup

from src.utils import normalize_nepali_text, load_json, save_json, get_settings

logger = logging.getLogger(__name__)


@dataclass
class ProcessedDocument:
    """A processed document ready for chunking."""
    domain_id: str
    source_name: str
    source_url: str
    document_type: str
    title: str
    language: str
    raw_text: str
    cleaned_text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    raw_path: str = ""
    processed_path: str = ""


def detect_language(text: str) -> str:
    """Detect if text is primarily Nepali, English, or mixed."""
    devanagari = len(re.findall(r"[\u0900-\u097F]", text))
    english = len(re.findall(r"[a-zA-Z]", text))
    total = devanagari + english
    if total == 0:
        return "unknown"
    nepali_ratio = devanagari / total
    if nepali_ratio > 0.7:
        return "ne"
    elif nepali_ratio < 0.3:
        return "en"
    return "mixed"


def _is_navigation_text(line: str) -> bool:
    """Check if a line is navigation/menu content."""
    line = line.strip()
    if not line:
        return True
    
    # Common navigation patterns
    nav_patterns = [
        r'^(home|about|contact|login|register|logout|search|help|faq|sitemap)$',
        r'^(http|www|ftp)',
        r'^(facebook|twitter|instagram|youtube|linkedin)$',
        r'^(privacy|terms|policy|copyright|all rights)',
        r'^\d{4}[-\s]?\d{2}[-\s]?\d{2}$',  # dates
        r'^\+?[\d\s\-\(\)]{7,15}$',  # phone numbers
        r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',  # emails
        r'^(skip to|go to|back to|click here|read more|learn more)',
        r'^(page \d+|previous|next|start|end)$',
    ]
    
    for pattern in nav_patterns:
        if re.match(pattern, line, re.IGNORECASE):
            return True
    
    return False


def _filter_navigation_lines(text: str) -> str:
    """Remove navigation/menu/footer lines from text."""
    lines = text.split('\n')
    filtered = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check if it's navigation
        if _is_navigation_text(line):
            continue
        
        # Skip very short lines that are likely UI elements (but keep Nepali text)
        if len(line) < 20 and not re.search(r'[ऀ-ॿ]', line):
            continue
        
        # Skip lines that are just symbols/emojis
        if re.match(r'^[•·●○■□▪▫†‡§¶©®™]+$', line):
            continue
        
        filtered.append(line)
    
    return '\n'.join(filtered)


def extract_text_from_html(html_content: str) -> tuple[str, str]:
    """Extract clean text and title from HTML content."""
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Get title
    title = ""
    if soup.title:
        title = soup.title.get_text(strip=True)
    elif soup.find("h1"):
        title = soup.find("h1").get_text(strip=True)
    
    # Remove script, style, nav, footer, header, aside
    for element in soup.find_all(["script", "style", "nav", "footer", "header", "aside"]):
        element.decompose()
    
    # Remove elements with common navigation class/ID names
    nav_classes = re.compile(r'nav|menu|sidebar|breadcrumb|pagination|toolbar|social|share|comment|advertisement|banner', re.I)
    for element in soup.find_all(class_=nav_classes):
        element.decompose()
    for element in soup.find_all(id=nav_classes):
        element.decompose()
    
    # Try to find main content area
    main_content = (
        soup.find("main") 
        or soup.find("article")
        or soup.find("div", class_=re.compile(r'content|main', re.I))
        or soup.find("div", id=re.compile(r'content|main', re.I))
        or soup.find("body")
    )
    
    if main_content:
        text = main_content.get_text(separator="\n", strip=True)
    else:
        text = soup.get_text(separator="\n", strip=True)
    
    # Filter navigation lines
    text = _filter_navigation_lines(text)
    
    # Don't filter too aggressively - keep at least 50% of original
    original_lines = text.split('\n')
    if len(original_lines) < 5:
        # Too few lines - use original body text
        if main_content:
            text = main_content.get_text(separator="\n", strip=True)
        else:
            text = soup.get_text(separator="\n", strip=True)
        text = _filter_navigation_lines(text)
    
    return title, text


def extract_text_from_pdf(pdf_path: Path) -> tuple[str, str]:
    """Extract text from PDF using PyMuPDF."""
    doc = fitz.open(pdf_path)
    title = doc.metadata.get("title", "") or ""
    text_parts = []
    for page in doc:
        page_text = page.get_text()
        if page_text.strip():
            text_parts.append(page_text)
    doc.close()
    full_text = "\n\n".join(text_parts)
    return title, full_text


def needs_ocr(pdf_path: Path, text_threshold: int = 100) -> bool:
    """Check if a PDF needs OCR (scanned/image-only pages)."""
    doc = fitz.open(pdf_path)
    total_text = 0
    for page in doc:
        total_text += len(page.get_text().strip())
    doc.close()
    return total_text < text_threshold


def ocr_pdf(pdf_path: Path, language: str = "neng") -> str:
    """OCR a scanned PDF using pytesseract."""
    try:
        import pytesseract
        from PIL import Image
        import io
    except ImportError:
        logger.warning("pytesseract or PIL not installed, skipping OCR")
        return ""
    
    doc = fitz.open(pdf_path)
    text_parts = []
    for page_num, page in enumerate(doc):
        mat = fitz.Matrix(3, 3)  # 3x zoom
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        text = pytesseract.image_to_string(img, lang=language)
        if text.strip():
            text_parts.append(text)
    doc.close()
    return "\n\n".join(text_parts)


def process_raw_file(raw_path: Path, meta: dict[str, Any]) -> ProcessedDocument | None:
    """Process a single raw file into a cleaned document."""
    suffix = raw_path.suffix.lower()
    
    if suffix == ".html":
        html_content = raw_path.read_text(encoding="utf-8", errors="replace")
        title, raw_text = extract_text_from_html(html_content)
    elif suffix == ".pdf":
        title, raw_text = extract_text_from_pdf(raw_path)
        if needs_ocr(raw_path):
            logger.info("PDF appears scanned, running OCR: %s", raw_path.name)
            ocr_text = ocr_pdf(raw_path)
            if ocr_text:
                raw_text = ocr_text
    elif suffix == ".txt":
        raw_text = raw_path.read_text(encoding="utf-8", errors="replace")
        title = ""
    else:
        logger.warning("Unknown file type: %s", raw_path.name)
        return None
    
    # Normalize
    cleaned_text = normalize_nepali_text(raw_text)
    language = detect_language(cleaned_text)
    
    return ProcessedDocument(
        domain_id=meta.get("domain_id", "unknown"),
        source_name=meta.get("source_name", raw_path.stem),
        source_url=meta.get("source_url", ""),
        document_type=meta.get("document_type", "unknown"),
        title=title,
        language=language,
        raw_text=raw_text,
        cleaned_text=cleaned_text,
        metadata=meta,
        raw_path=str(raw_path),
    )


def process_all(domain_id: str | None = None, verbose: bool = False) -> list[ProcessedDocument]:
    """Process all raw documents."""
    settings = get_settings()
    raw_dir = Path(settings["paths"]["raw_dir"])
    processed_dir = Path(settings["paths"]["processed_dir"])
    
    processed_docs: list[ProcessedDocument] = []
    
    # Find all meta files
    meta_files = list(raw_dir.rglob("*.meta.json"))
    
    for meta_path in meta_files:
        meta = load_json(meta_path)
        if domain_id and meta.get("domain_id") != domain_id:
            continue
        
        # Find the corresponding raw file
        meta_stem = str(meta_path).replace(".meta.json", "")
        raw_path = Path(meta_stem)
        if not raw_path.exists():
            for ext in [".html", ".pdf", ".txt"]:
                candidate = Path(meta_stem + ext)
                if candidate.exists():
                    raw_path = candidate
                    break
            else:
                logger.warning("Raw file not found for meta: %s", meta_path)
                continue
        
        doc = process_raw_file(raw_path, meta)
        if doc and len(doc.cleaned_text) > 100:  # Skip empty/short docs
            # Save processed document
            domain_dir = processed_dir / doc.domain_id
            domain_dir.mkdir(parents=True, exist_ok=True)
            
            safe_name = re.sub(r"[^\w\-.]", "_", doc.source_name)[:80]
            processed_path = domain_dir / f"{safe_name}.json"
            
            save_json({
                "domain_id": doc.domain_id,
                "source_name": doc.source_name,
                "source_url": doc.source_url,
                "document_type": doc.document_type,
                "title": doc.title,
                "language": doc.language,
                "cleaned_text": doc.cleaned_text,
                "metadata": doc.metadata,
            }, processed_path)
            
            doc.processed_path = str(processed_path)
            processed_docs.append(doc)
            
            if verbose:
                print(f"  Processed: {doc.source_name} ({len(doc.cleaned_text)} chars, {doc.language})")
    
    logger.info("Processed %d documents", len(processed_docs))
    return processed_docs
