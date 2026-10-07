"""Processing pipeline — orchestrates document processing and chunking."""

from __future__ import annotations

import logging
from pathlib import Path

from src.processing import process_all as process_raw_documents
from src.chunking.structure_aware import chunk_all_processed
from src.utils import get_settings

logger = logging.getLogger(__name__)


def process_all(domain_id: str | None = None, verbose: bool = False) -> dict:
    """Run full processing pipeline: raw -> cleaned -> chunks."""
    settings = get_settings()
    
    # Phase 2: Process raw documents
    logger.info("Phase 2: Processing raw documents...")
    processed_docs = process_raw_documents(domain_id=domain_id, verbose=verbose)
    
    if not processed_docs:
        logger.warning("No documents were processed. Run 'fetch' first.")
        return {"processed": 0, "chunks": 0}
    
    # Phase 3: Chunk processed documents
    logger.info("Phase 3: Chunking processed documents...")
    chunks = chunk_all_processed(
        domain_id=domain_id,
        max_tokens=settings["chunking"]["max_chunk_tokens"],
        overlap_tokens=settings["chunking"]["chunk_overlap_tokens"],
        verbose=verbose,
    )
    
    result = {
        "processed": len(processed_docs),
        "chunks": len(chunks),
        "domains": list({d.domain_id for d in processed_docs}),
    }
    
    logger.info(
        "Pipeline complete: %d documents processed, %d chunks created",
        result["processed"],
        result["chunks"],
    )
    return result
