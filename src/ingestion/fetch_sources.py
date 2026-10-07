"""
Phase 1: Source acquisition.

Reads config/domains.yaml and fetches every configured government source,
saving the raw content to data/raw/<domain_id>/ along with a metadata
sidecar file. The metadata captured here (source URL, issuing authority,
document type, fetch timestamp, content hash) is exactly what the proposal's
"Data Collection and Preparation" section calls for, and it's what later
lets the generation component cite a source and avoid claiming real-time
currency for information it can't verify (see proposal's Ethical section).

This script is intentionally conservative:
- it does not guess at pages that aren't explicitly listed in domains.yaml
- it never overwrites a previous fetch silently — it versions by timestamp
- every failure is recorded, not swallowed, so gaps in the corpus are visible

Usage:
    python -m src.ingestion.fetch_sources
    python -m src.ingestion.fetch_sources --domain identity_civil_registration
    python -m src.ingestion.fetch_sources --config config/domains.yaml --delay 2.0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("config/domains.yaml")
DEFAULT_OUTPUT_DIR = Path("data/raw")
USER_AGENT = (
    "Mozilla/5.0 (compatible; NepalPublicServiceRAG-CapstoneBot/0.1; "
    "research project, contact: <add your project contact email here>)"
)
REQUEST_TIMEOUT_SECONDS = 20


@dataclass
class FetchResult:
    """Everything downstream components need to know about one fetch attempt."""

    domain_id: str
    domain_name_en: str
    source_name: str
    source_url: str
    document_type: str
    fetched_at: str  # ISO 8601, UTC
    status: str  # "ok" | "http_error" | "request_error"
    http_status_code: int | None
    content_type: str | None
    content_length_bytes: int | None
    sha256: str | None
    saved_path: str | None
    error_message: str | None = None


def load_domain_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(
            f"Domain config not found at {config_path}. "
            "Run this from the repo root, or pass --config explicitly."
        )
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def slugify(text: str, max_length: int = 80) -> str:
    """Turn a name or URL path into a filesystem-safe slug."""
    text = re.sub(r"https?://", "", text)
    text = re.sub(r"[^\w\-.]+", "-", text, flags=re.UNICODE)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text[:max_length] if text else "source"


def guess_extension(content_type: str | None) -> str:
    if not content_type:
        return ".bin"
    content_type = content_type.split(";")[0].strip().lower()
    return {
        "text/html": ".html",
        "application/pdf": ".pdf",
        "text/plain": ".txt",
    }.get(content_type, ".bin")


def fetch_one_source(
    domain_id: str,
    domain_name_en: str,
    source: dict[str, Any],
    output_dir: Path,
) -> FetchResult:
    """Fetch a single configured source and persist raw content + metadata."""
    source_name = source["name"]
    source_url = source["url"]
    document_type = source.get("type", "unknown")
    fetched_at = datetime.now(timezone.utc).isoformat()

    headers = {"User-Agent": USER_AGENT}

    try:
        response = requests.get(
            source_url, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS, verify=False
        )
    except requests.RequestException as exc:
        logger.warning("Request failed for %s (%s): %s", source_name, source_url, exc)
        return FetchResult(
            domain_id=domain_id,
            domain_name_en=domain_name_en,
            source_name=source_name,
            source_url=source_url,
            document_type=document_type,
            fetched_at=fetched_at,
            status="request_error",
            http_status_code=None,
            content_type=None,
            content_length_bytes=None,
            sha256=None,
            saved_path=None,
            error_message=str(exc),
        )

    content_type = response.headers.get("Content-Type")
    status = "ok" if response.ok else "http_error"

    saved_path: str | None = None
    sha256: str | None = None
    content_length_bytes: int | None = None

    if response.ok:
        domain_dir = output_dir / domain_id
        domain_dir.mkdir(parents=True, exist_ok=True)

        ext = guess_extension(content_type)
        timestamp_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        filename = f"{slugify(source_name)}__{timestamp_tag}{ext}"
        file_path = domain_dir / filename

        file_path.write_bytes(response.content)
        saved_path = str(file_path)
        sha256 = hashlib.sha256(response.content).hexdigest()
        content_length_bytes = len(response.content)

        logger.info("Saved %s -> %s (%d bytes)", source_url, file_path, content_length_bytes)
    else:
        logger.warning(
            "HTTP %s for %s (%s)", response.status_code, source_name, source_url
        )

    result = FetchResult(
        domain_id=domain_id,
        domain_name_en=domain_name_en,
        source_name=source_name,
        source_url=source_url,
        document_type=document_type,
        fetched_at=fetched_at,
        status=status,
        http_status_code=response.status_code,
        content_type=content_type,
        content_length_bytes=content_length_bytes,
        sha256=sha256,
        saved_path=saved_path,
    )

    if saved_path:
        meta_path = Path(saved_path).with_suffix(Path(saved_path).suffix + ".meta.json")
        meta_path.write_text(json.dumps(asdict(result), indent=2, ensure_ascii=False), encoding="utf-8")

    return result


def run(config_path: Path, output_dir: Path, domain_filter: str | None, delay_seconds: float) -> list[FetchResult]:
    config = load_domain_config(config_path)
    domains = config.get("domains", [])

    if domain_filter:
        domains = [d for d in domains if d["id"] == domain_filter]
        if not domains:
            raise ValueError(f"No domain with id '{domain_filter}' found in {config_path}")

    results: list[FetchResult] = []
    for domain in domains:
        domain_id = domain["id"]
        domain_name_en = domain.get("name_en", domain_id)
        sources = domain.get("sources", [])
        logger.info("Domain '%s': %d configured source(s)", domain_name_en, len(sources))

        for source in sources:
            result = fetch_one_source(domain_id, domain_name_en, source, output_dir)
            results.append(result)
            time.sleep(delay_seconds)  # be polite to government servers

    return results


def print_summary(results: list[FetchResult]) -> None:
    ok = [r for r in results if r.status == "ok"]
    failed = [r for r in results if r.status != "ok"]

    print(f"\n{'=' * 60}")
    print(f"Fetch summary: {len(ok)} succeeded, {len(failed)} failed (of {len(results)} total)")
    print(f"{'=' * 60}")
    if failed:
        print("\nFailed sources (fix these before Phase 2):")
        for r in failed:
            reason = r.error_message or f"HTTP {r.http_status_code}"
            print(f"  - [{r.domain_id}] {r.source_name}: {reason}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch configured government sources (Phase 1).")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--domain", type=str, default=None, help="Fetch only this domain id")
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds to wait between requests")
    args = parser.parse_args()

    results = run(args.config, args.output_dir, args.domain, args.delay)
    print_summary(results)


if __name__ == "__main__":
    main()
