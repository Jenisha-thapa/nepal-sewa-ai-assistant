"""Phase 7: Evaluation — domain classification, retrieval, and generation metrics."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

from src.utils import get_settings, load_json, save_json, find_repo_root

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Results from a single evaluation run."""
    domain_id: str
    question: str
    predicted_domain: str
    true_domain: str
    retrieved_chunks: list[dict[str, Any]]
    generated_answer: str
    citations: list[dict[str, Any]]
    refused: bool
    # Metrics
    domain_correct: bool = False
    retrieval_precision_at_k: dict[int, float] = field(default_factory=dict)
    retrieval_recall_at_k: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0


def compute_precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Compute Precision@K."""
    if k == 0 or not retrieved_ids:
        return 0.0
    retrieved_at_k = retrieved_ids[:k]
    relevant_retrieved = len(set(retrieved_at_k) & set(relevant_ids))
    return relevant_retrieved / k


def compute_recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Compute Recall@K."""
    if not relevant_ids:
        return 0.0
    retrieved_at_k = retrieved_ids[:k]
    relevant_retrieved = len(set(retrieved_at_k) & set(relevant_ids))
    return relevant_retrieved / len(relevant_ids)


def compute_mrr(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """Compute Mean Reciprocal Rank."""
    for i, chunk_id in enumerate(retrieved_ids, 1):
        if chunk_id in relevant_ids:
            return 1.0 / i
    return 0.0


def evaluate_domain_classification(
    true_labels: list[str],
    predicted_labels: list[str],
) -> dict[str, Any]:
    """Evaluate domain classification performance."""
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average="weighted", zero_division=0,
    )
    
    # Per-domain metrics
    labels = sorted(set(true_labels))
    per_domain = {}
    for label in labels:
        true_binary = [1 if l == label else 0 for l in true_labels]
        pred_binary = [1 if l == label else 0 for l in predicted_labels]
        p, r, f, _ = precision_recall_fscore_support(
            true_binary, pred_binary, average="binary", zero_division=0,
        )
        per_domain[label] = {"precision": p, "recall": r, "f1": f}
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "per_domain": per_domain,
        "classification_report": classification_report(
            true_labels, predicted_labels, zero_division=0
        ),
    }


def evaluate_retrieval(
    results: list[EvaluationResult],
    k_values: list[int] = [1, 3, 5],
) -> dict[str, Any]:
    """Evaluate retrieval performance."""
    metrics = {}
    
    for k in k_values:
        precisions = []
        recalls = []
        for r in results:
            if r.retrieval_precision_at_k.get(k) is not None:
                precisions.append(r.retrieval_precision_at_k[k])
                recalls.append(r.retrieval_recall_at_k[k])
        
        metrics[f"precision@{k}"] = np.mean(precisions) if precisions else 0.0
        metrics[f"recall@{k}"] = np.mean(recalls) if recalls else 0.0
    
    mrrs = [r.mrr for r in results if r.mrr > 0]
    metrics["mrr"] = np.mean(mrrs) if mrrs else 0.0
    
    return metrics


def evaluate_generation(
    results: list[EvaluationResult],
) -> dict[str, Any]:
    """Evaluate generation quality (requires human evaluation)."""
    # Count refusals
    refusals = sum(1 for r in results if r.refused)
    total = len(results)
    
    # Citation statistics
    with_citations = sum(1 for r in results if r.citations and not r.refused)
    without_citations = sum(1 for r in results if not r.citations and not r.refused)
    
    return {
        "total_questions": total,
        "refusals": refusals,
        "refusal_rate": refusals / total if total > 0 else 0,
        "answered": total - refusals,
        "with_citations": with_citations,
        "without_citations": without_citations,
        "citation_rate": with_citations / (total - refusals) if (total - refusals) > 0 else 0,
    }


def run_evaluation(
    output_path: str | None = None,
    domains: list[str] | None = None,
) -> dict[str, Any]:
    """Run full evaluation pipeline."""
    settings = get_settings()
    repo_root = find_repo_root()
    
    # Load eval set
    eval_path = Path(settings["paths"]["eval_dir"]) / "eval_set.json"
    if not eval_path.exists():
        logger.error("Eval set not found. Generate it first.")
        return {}
    
    eval_set = load_json(eval_path)
    
    if domains:
        eval_set = [q for q in eval_set if q.get("domain_id") in domains]
    
    if not eval_set:
        logger.warning("No evaluation questions found.")
        return {}
    
    logger.info("Running evaluation on %d questions...", len(eval_set))
    
    # Load classifier
    from src.domain_classifier.classifier import EmbeddingClassifier
    classifier = EmbeddingClassifier(settings["classifier"]["embedding_model"])
    classifier_path = repo_root / "data" / "index" / "domain_classifier.pkl"
    if classifier_path.exists():
        classifier.load(classifier_path)
    
    # Evaluate
    results = []
    true_domains = []
    predicted_domains = []
    
    for question_item in eval_set:
        question = question_item.get("question", "")
        true_domain = question_item.get("domain_id", "")
        gold_chunk_ids = question_item.get("gold_chunk_ids", [])
        
        true_domains.append(true_domain)
        
        # Classify domain
        if classifier.is_trained:
            classification = classifier.predict(question)
            predicted_domain = classification.domain_id
        else:
            predicted_domain = "unknown"
        
        predicted_domains.append(predicted_domain)
        
        # Build result
        result = EvaluationResult(
            domain_id=true_domain,
            question=question,
            predicted_domain=predicted_domain,
            true_domain=true_domain,
            retrieved_chunks=[],
            generated_answer="",
            citations=[],
            refused=False,
            domain_correct=(predicted_domain == true_domain),
        )
        
        # Compute retrieval metrics if we have gold chunks
        if gold_chunk_ids:
            # In a real run, we'd retrieve and compare
            # For now, placeholder
            result.mrr = 0.0
        
        results.append(result)
    
    # Compute metrics
    domain_metrics = evaluate_domain_classification(true_domains, predicted_domains)
    retrieval_metrics = evaluate_retrieval(results, k_values=settings["evaluation"]["k_values"])
    generation_metrics = evaluate_generation(results)
    
    all_metrics = {
        "domain_classification": domain_metrics,
        "retrieval": retrieval_metrics,
        "generation": generation_metrics,
        "total_questions": len(eval_set),
    }
    
    # Save results
    if output_path:
        save_json(all_metrics, output_path)
        logger.info("Evaluation results saved to %s", output_path)
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total questions: {len(eval_set)}")
    print(f"\nDomain Classification:")
    print(f"  Accuracy:  {domain_metrics['accuracy']:.3f}")
    print(f"  F1:        {domain_metrics['f1']:.3f}")
    print(f"\nRetrieval:")
    for k in settings["evaluation"]["k_values"]:
        print(f"  P@{k}: {retrieval_metrics.get(f'precision@{k}', 0):.3f}")
        print(f"  R@{k}: {retrieval_metrics.get(f'recall@{k}', 0):.3f}")
    print(f"  MRR: {retrieval_metrics.get('mrr', 0):.3f}")
    print(f"\nGeneration:")
    print(f"  Refusal rate: {generation_metrics['refusal_rate']:.3f}")
    print(f"  Citation rate: {generation_metrics['citation_rate']:.3f}")
    print("=" * 60)
    
    return all_metrics
