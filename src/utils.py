"""Shared utilities for the Nepal RAG Agent."""

from __future__ import annotations

import logging
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return its contents."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    """Save data to a YAML file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON file."""
    import json
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    """Save data to a JSON file."""
    import json
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def normalize_nepali_text(text: str) -> str:
    """Normalize Nepali/Unicode text for consistent processing.
    
    - NFC normalization
    - Remove zero-width characters
    - Normalize whitespace
    - Remove control characters
    """
    if not text:
        return ""
    # NFC normalization
    text = unicodedata.normalize("NFC", text)
    # Remove zero-width characters
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    # Remove control characters except newlines
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Normalize whitespace (preserve newlines)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def truncate_text(text: str, max_chars: int = 100) -> str:
    """Truncate text for display."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."


@lru_cache(maxsize=1)
def get_settings() -> dict[str, Any]:
    """Load and cache project settings."""
    repo_root = find_repo_root()
    return load_yaml(repo_root / "config" / "settings.yaml")


def find_repo_root() -> Path:
    """Find the repository root by looking for config/settings.yaml."""
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / "config" / "settings.yaml").exists():
            return parent
    return current


def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """Configure logging for the project."""
    log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=level.upper(), format=log_format, handlers=handlers)


def count_tokens_approx(text: str) -> int:
    """Approximate token count (rough heuristic for Devanagari + English)."""
    # Devanagari characters are roughly 1 token each
    # English words are roughly 1.3 tokens each
    devanagari = len(re.findall(r"[\u0900-\u097F]", text))
    english = len(re.findall(r"[a-zA-Z]+", text))
    numbers = len(re.findall(r"\d+", text))
    return devanagari + int(english * 1.3) + numbers
