"""Phase 5: Retrieval — BM25, dense, and hybrid retrieval with ChromaDB."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from src.utils import get_settings, load_json, save_json, find_repo_root

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """A single retrieval result."""
    chunk_id: str
    domain_id: str
    source_name: str
    source_url: str
    document_type: str
    title: str
    section: str
    text: str
    score: float
    position: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class BM25Retriever:
    """BM25 lexical retriever."""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.index: BM25Okapi | None = None
        self.chunks: list[dict[str, Any]] = []
        self.tokenized_corpus: list[list[str]] = []
    
    def build_index(self, chunks: list[dict[str, Any]]) -> None:
        """Build BM25 index from chunks."""
        self.chunks = chunks
        self.tokenized_corpus = [self._tokenize(c.get("text", "")) for c in chunks]
        self.index = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)
        logger.info("BM25 index built with %d chunks", len(chunks))
    
    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization for BM25."""
        import re
        return re.findall(r"[\w\u0900-\u097F]+", text.lower())
    
    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Search using BM25."""
        if not self.index:
            raise RuntimeError("Index not built. Call build_index() first.")
        
        tokenized_query = self._tokenize(query)
        scores = self.index.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            chunk = self.chunks[idx]
            results.append(RetrievalResult(
                chunk_id=chunk.get("chunk_id", f"bm25_{idx}"),
                domain_id=chunk.get("domain_id", ""),
                source_name=chunk.get("source_name", ""),
                source_url=chunk.get("source_url", ""),
                document_type=chunk.get("document_type", ""),
                title=chunk.get("title", ""),
                section=chunk.get("section", ""),
                text=chunk.get("text", ""),
                score=float(scores[idx]),
                position=chunk.get("position", 0),
                metadata=chunk.get("metadata", {}),
            ))
        
        return results


class DenseRetriever:
    """Dense retriever using sentence-transformers and ChromaDB."""
    
    def __init__(self, model_name: str, persist_dir: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.persist_dir = persist_dir
        self.collection = None
        self.chroma_client = None
    
    def build_index(self, chunks: list[dict[str, Any]]) -> None:
        """Build vector index in ChromaDB."""
        import chromadb
        
        if self.persist_dir:
            Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(path=self.persist_dir)
        else:
            self.chroma_client = chromadb.Client()
        
        # Create or get collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="nepal_gov_services",
            metadata={"hnsw:space": "cosine"},
        )
        
        # Encode and add in batches
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [c.get("text", "") for c in batch]
            embeddings = self.model.encode(texts, show_progress_bar=False).tolist()
            
            ids = [c.get("chunk_id", f"chunk_{i + j}") for j, c in enumerate(batch)]
            metadatas = [{
                "domain_id": c.get("domain_id", ""),
                "source_name": c.get("source_name", ""),
                "source_url": c.get("source_url", ""),
                "document_type": c.get("document_type", ""),
                "title": c.get("title", ""),
                "section": c.get("section", ""),
                "position": c.get("position", 0),
            } for c in batch]
            
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )
        
        logger.info("Dense index built with %d chunks", len(chunks))
    
    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Search using dense retrieval."""
        if not self.collection:
            raise RuntimeError("Index not built. Call build_index() first.")
        
        query_embedding = self.model.encode([query]).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
        )
        
        retrieval_results = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 0
                # Convert distance to similarity score (cosine distance -> similarity)
                score = 1.0 - distance
                
                retrieval_results.append(RetrievalResult(
                    chunk_id=chunk_id,
                    domain_id=metadata.get("domain_id", ""),
                    source_name=metadata.get("source_name", ""),
                    source_url=metadata.get("source_url", ""),
                    document_type=metadata.get("document_type", ""),
                    title=metadata.get("title", ""),
                    section=metadata.get("section", ""),
                    text=results["documents"][0][i] if results["documents"] else "",
                    score=score,
                    position=metadata.get("position", 0),
                ))
        
        return retrieval_results


class HybridRetriever:
    """Hybrid retriever combining BM25 and dense retrieval."""
    
    def __init__(
        self,
        bm25_weight: float = 0.4,
        dense_weight: float = 0.6,
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        persist_dir: str | None = None,
    ) -> None:
        self.bm25 = BM25Retriever()
        self.dense = DenseRetriever(embedding_model, persist_dir)
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
    
    def build_index(self, chunks: list[dict[str, Any]]) -> None:
        """Build both BM25 and dense indices."""
        self.bm25.build_index(chunks)
        self.dense.build_index(chunks)
    
    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Search using hybrid approach with score fusion."""
        bm25_results = self.bm25.search(query, top_k=top_k * 2)
        dense_results = self.dense.search(query, top_k=top_k * 2)
        
        # Normalize scores
        if bm25_results:
            max_bm25 = max(r.score for r in bm25_results)
            if max_bm25 > 0:
                for r in bm25_results:
                    r.score = r.score / max_bm25
        
        if dense_results:
            max_dense = max(r.score for r in dense_results)
            if max_dense > 0:
                for r in dense_results:
                    r.score = r.score / max_dense
        
        # Merge results by chunk_id
        merged: dict[str, dict[str, Any]] = {}
        
        for r in bm25_results:
            merged[r.chunk_id] = {
                "result": r,
                "bm25_score": r.score,
                "dense_score": 0.0,
            }
        
        for r in dense_results:
            if r.chunk_id in merged:
                merged[r.chunk_id]["dense_score"] = r.score
            else:
                merged[r.chunk_id] = {
                    "result": r,
                    "bm25_score": 0.0,
                    "dense_score": r.score,
                }
        
        # Compute weighted score
        final_results = []
        for chunk_id, data in merged.items():
            weighted_score = (
                self.bm25_weight * data["bm25_score"]
                + self.dense_weight * data["dense_score"]
            )
            result = data["result"]
            result.score = weighted_score
            final_results.append(result)
        
        # Sort by score and return top_k
        final_results.sort(key=lambda x: x.score, reverse=True)
        return final_results[:top_k]


def build_index(domain_id: str | None = None, force_rebuild: bool = False) -> None:
    """Build retrieval index from processed chunks."""
    settings = get_settings()
    repo_root = find_repo_root()
    
    chunks_path = Path(settings["paths"]["processed_dir"]) / "all_chunks.json"
    if not chunks_path.exists():
        logger.error("No chunks found. Run 'process' first.")
        return
    
    chunks = load_json(chunks_path)
    
    if domain_id:
        chunks = [c for c in chunks if c.get("domain_id") == domain_id]
    
    if not chunks:
        logger.warning("No chunks to index.")
        return
    
    mode = settings["retrieval"]["mode"]
    
    if mode == "bm25":
        retriever = BM25Retriever(
            k1=settings["retrieval"]["bm25"]["k1"],
            b=settings["retrieval"]["bm25"]["b"],
        )
    elif mode == "dense":
        retriever = DenseRetriever(
            model_name=settings["retrieval"]["embedding_model"],
            persist_dir=settings["retrieval"]["chromadb"]["persist_dir"],
        )
    elif mode == "hybrid":
        retriever = HybridRetriever(
            bm25_weight=settings["retrieval"]["hybrid"]["bm25_weight"],
            dense_weight=settings["retrieval"]["hybrid"]["dense_weight"],
            embedding_model=settings["retrieval"]["embedding_model"],
            persist_dir=settings["retrieval"]["chromadb"]["persist_dir"],
        )
    else:
        raise ValueError(f"Unknown retrieval mode: {mode}")
    
    retriever.build_index(chunks)
    
    # Save retriever state
    index_dir = repo_root / "data" / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Index built successfully with mode=%s", mode)
