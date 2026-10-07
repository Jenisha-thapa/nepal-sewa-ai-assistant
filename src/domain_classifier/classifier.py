"""Phase 4: Domain classifier — routes queries to the correct domain."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from src.utils import get_settings, load_json, save_json, find_repo_root

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    domain_id: str
    confidence: float
    all_scores: dict[str, float]


class KeywordClassifier:
    """Simple keyword-matching baseline classifier."""
    
    def __init__(self, keyword_map: dict[str, list[str]]) -> None:
        self.keyword_map = {k: [kw.lower() for kw in v] for k, v in keyword_map.items()}
    
    def predict(self, query: str) -> ClassificationResult:
        query_lower = query.lower()
        scores: dict[str, float] = {}
        
        for domain_id, keywords in self.keyword_map.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            # Normalize by keyword count
            scores[domain_id] = score / max(len(keywords), 1)
        
        # Get best match
        if any(scores.values()):
            best_domain = max(scores, key=scores.get)
            total = sum(scores.values())
            confidence = scores[best_domain] / total if total > 0 else 0.0
        else:
            best_domain = "unknown"
            confidence = 0.0
        
        return ClassificationResult(
            domain_id=best_domain,
            confidence=confidence,
            all_scores=scores,
        )


class EmbeddingClassifier:
    """Embedding-based classifier using logistic regression."""
    
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.classifier = LogisticRegression(max_iter=1000, C=1.0)
        self.label_to_domain: dict[int, str] = {}
        self.is_trained = False
    
    def _encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, show_progress_bar=False, batch_size=32)
    
    def train(self, X: list[str], y: list[str]) -> dict[str, Any]:
        """Train the classifier on text-domain pairs."""
        # Create label mapping
        unique_labels = sorted(set(y))
        self.label_to_domain = {i: label for i, label in enumerate(unique_labels)}
        domain_to_label = {label: i for i, label in enumerate(unique_labels)}
        
        y_encoded = [domain_to_label[label] for label in y]
        
        # Encode texts
        X_encoded = self._encode(X)
        
        # Split
        # Check if we have enough samples per class for stratified split
        min_samples = min(y_encoded.count(label) for label in set(y_encoded))
        if min_samples >= 5:
            X_train, X_test, y_train, y_test = train_test_split(
                X_encoded, y_encoded,
                test_size=0.2,
                random_state=42,
                stratify=y_encoded,
            )
        else:
            # No stratification for small datasets
            X_train, X_test, y_train, y_test = train_test_split(
                X_encoded, y_encoded,
                test_size=0.2,
                random_state=42,
            )
        
        # Train
        self.classifier.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="weighted", zero_division=0,
            labels=list(self.label_to_domain.keys()),
        )
        
        # Per-domain report
        report = classification_report(y_test, y_pred, zero_division=0)
        
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "classification_report": report,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "num_classes": len(unique_labels),
        }
        
        logger.info(
            "Classifier trained — accuracy: %.3f, F1: %.3f", accuracy, f1
        )
        return metrics
    
    def predict(self, query: str) -> ClassificationResult:
        """Predict domain for a query."""
        if not self.is_trained:
            raise RuntimeError("Classifier not trained. Call train() first.")
        
        embedding = self._encode([query])
        probs = self.classifier.predict_proba(embedding)[0]
        
        # Build scores dict
        scores = {
            self.label_to_domain[i]: float(prob)
            for i, prob in enumerate(probs)
        }
        
        best_idx = np.argmax(probs)
        
        return ClassificationResult(
            domain_id=self.label_to_domain[best_idx],
            confidence=float(probs[best_idx]),
            all_scores=scores,
        )
    
    def save(self, path: str | Path) -> None:
        """Save trained classifier."""
        import pickle
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "classifier": self.classifier,
                "label_to_domain": self.label_to_domain,
                "is_trained": self.is_trained,
            }, f)
        logger.info("Classifier saved to %s", path)
    
    def load(self, path: str | Path) -> None:
        """Load a trained classifier."""
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.classifier = data["classifier"]
        self.label_to_domain = data["label_to_domain"]
        self.is_trained = data["is_trained"]
        logger.info("Classifier loaded from %s", path)


def generate_training_data() -> tuple[list[str], list[str]]:
    """Generate training data from processed documents."""
    settings = get_settings()
    processed_dir = Path(settings["paths"]["processed_dir"])
    chunks_path = processed_dir / "all_chunks.json"
    
    if not chunks_path.exists():
        logger.error("No chunks found. Run 'process' first.")
        return [], []
    
    chunks = load_json(chunks_path)
    
    texts = []
    labels = []
    
    for chunk in chunks:
        text = chunk.get("text", "")
        domain = chunk.get("domain_id", "")
        if text and domain:
            texts.append(text)
            labels.append(domain)
    
    logger.info("Generated %d training samples from chunks", len(texts))
    return texts, labels


def train_and_evaluate() -> dict[str, Any]:
    """Train and evaluate domain classifier, compare with keyword baseline."""
    settings = get_settings()
    repo_root = find_repo_root()
    
    # Generate training data
    texts, labels = generate_training_data()
    
    if not texts:
        # Use synthetic training data if no chunks available
        texts, labels = _generate_synthetic_training_data()
    
    results = {}
    
    # Train embedding classifier
    logger.info("Training embedding-based classifier...")
    classifier = EmbeddingClassifier(settings["classifier"]["embedding_model"])
    metrics = classifier.train(texts, labels)
    results["embedding_classifier"] = metrics
    
    # Save classifier
    model_path = repo_root / "data" / "index" / "domain_classifier.pkl"
    classifier.save(model_path)
    
    # Keyword baseline comparison
    logger.info("Evaluating keyword baseline...")
    keyword_clf = KeywordClassifier(settings["classifier"]["keyword_baseline"])
    
    # Evaluate keyword baseline on same data
    correct = 0
    for text, true_domain in zip(texts, labels):
        result = keyword_clf.predict(text[:200])  # Use first 200 chars
        if result.domain_id == true_domain:
            correct += 1
    
    keyword_accuracy = correct / len(texts) if texts else 0
    results["keyword_baseline"] = {"accuracy": keyword_accuracy}
    
    logger.info(
        "Comparison — Embedding: %.3f | Keyword: %.3f",
        metrics["accuracy"],
        keyword_accuracy,
    )
    
    return results


def _generate_synthetic_training_data() -> tuple[list[str], list[str]]:
    """Generate synthetic training data for demonstration."""
    random.seed(42)
    
    templates = {
        "identity_civil_registration": [
            "नागरिकता कार्ड कसरी लिन सकिन्छ?",
            "नागरिकता कार्डको आवेदन फारम कहाँ पाइन्छ?",
            "नागरिकता कार्ड कति दिनमा आउछ?",
            "जन्म दर्ता कसरी गर्ने?",
            "मृत्यु दर्ता कसरी गर्ने?",
            "विवाह दर्ता कसरी गर्ने?",
            "नागरिकता कार्ड हराएमा के गर्ने?",
            "National ID card application process in Nepal?",
        ],
        "land_property": [
            "जग्गा नामसारी कसरी गर्ने?",
            "जग्गा दर्ता कसरी गर्ने?",
            "लालपुर्जा कसरी पाइन्छ?",
            "जग्गाको मूल्यांकन कसरी हुन्छ?",
            "Land ownership transfer process in Nepal?",
            "जग्गाको नक्सा कहाँ पाइन्छ?",
            "जग्गा कर कसरी तिर्ने?",
            "How to register land in Nepal?",
        ],
        "employment_labour": [
            "वैदेशिक रोजगार अनुमोदन कसरी लिने?",
            "सामाजिक सुरक्षा कोष (SSF) के हो?",
            "SSF मा दर्ता कसरी गर्ने?",
            "वैदेशिक रोजगार बीमा कसरी लिने?",
            "Foreign employment approval process?",
            "सामाजिक सुरक्षा कोषको योगदान दर कति हो?",
            "How to register for Social Security Fund?",
            "वैदेशिक रोजगार कार्यक्रम के हो?",
        ],
        "health_social_welfare": [
            "स्वास्थ्य बीमा कसरी लिने?",
            "स्वास्थ्य बीमा योजना के हो?",
            "बृद्धा भत्ता कसरी पाइन्छ?",
            "बृद्धा भत्ता आवेदन कसरी गर्ने?",
            "Health insurance board Nepal?",
            "मातृत्व भत्ता कसरी पाइन्छ?",
            "How to get health insurance in Nepal?",
            "सामाजिक सुरक्षा भत्ता के हो?",
        ],
    }
    
    texts = []
    labels = []
    for domain_id, questions in templates.items():
        for q in questions:
            texts.append(q)
            labels.append(domain_id)
    
    return texts, labels
