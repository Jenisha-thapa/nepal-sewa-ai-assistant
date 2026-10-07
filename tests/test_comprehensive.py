"""Comprehensive tests for the Nepal RAG Agent."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.utils import (
    normalize_nepali_text,
    count_tokens_approx,
    load_json,
    save_json,
    load_yaml,
    save_yaml,
    find_repo_root,
)
from src.chunking import chunk_document, Chunk
from src.domain_classifier.classifier import KeywordClassifier, EmbeddingClassifier
from src.evaluation.runner import (
    compute_precision_at_k,
    compute_recall_at_k,
    compute_mrr,
    evaluate_domain_classification,
)


class TestNormalizeNepaliText:
    def test_removes_zero_width(self):
        text = "नागरिकता\u200bकार्ड"
        assert normalize_nepali_text(text) == "नागरिकताकार्ड"

    def test_normalizes_whitespace(self):
        text = "नागरिकता   कार्ड"
        assert normalize_nepali_text(text) == "नागरिकता कार्ड"

    def test_removes_control_chars(self):
        text = "नागरिकता\x00कार्ड"
        assert normalize_nepali_text(text) == "नागरिकताकार्ड"

    def test_nfc_normalization(self):
        # Devanagari composed vs decomposed
        text = "क\u094dष"  # decomposed
        result = normalize_nepali_text(text)
        assert result == "क्ष"  # composed

    def test_empty_string(self):
        assert normalize_nepali_text("") == ""

    def test_strips_whitespace(self):
        text = "  नागरिकता कार्ड  \n"
        assert normalize_nepali_text(text) == "नागरिकता कार्ड"


class TestCountTokensApprox:
    def test_devanagari(self):
        text = "नागरिकता कार्ड"
        tokens = count_tokens_approx(text)
        assert tokens > 0

    def test_english(self):
        text = "national id card"
        tokens = count_tokens_approx(text)
        assert tokens > 0

    def test_empty(self):
        assert count_tokens_approx("") == 0


class TestChunkDocument:
    def test_basic_chunking(self):
        doc = {
            "domain_id": "identity_civil_registration",
            "source_name": "Test Source",
            "source_url": "https://example.com",
            "document_type": "faq",
            "title": "Test FAQ",
            "cleaned_text": "प्रश्न: नागरिकता कार्ड कसरी लिने?\nउत्तर: आधार कार्ड र फोटो लिएर जानुहोस्।",
        }
        chunks = chunk_document(doc, max_tokens=500, overlap_tokens=50)
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_empty_text(self):
        doc = {
            "domain_id": "test",
            "source_name": "Test",
            "source_url": "https://example.com",
            "document_type": "faq",
            "title": "Test",
            "cleaned_text": "",
        }
        chunks = chunk_document(doc)
        assert chunks == []

    def test_preserves_metadata(self):
        doc = {
            "domain_id": "land_property",
            "source_name": "Land FAQ",
            "source_url": "https://dolma.gov.np/faqs",
            "document_type": "faq",
            "title": "Land FAQ",
            "cleaned_text": "जग्गा नामसारी कसरी गर्ने?\nआधार कार्ड र लालपुर्जा लिएर जानुहोस्।",
        }
        chunks = chunk_document(doc)
        assert all(c.domain_id == "land_property" for c in chunks)
        assert all(c.source_name == "Land FAQ" for c in chunks)


class TestKeywordClassifier:
    def test_identity_keywords(self):
        clf = KeywordClassifier({
            "identity_civil_registration": ["नागरिकता", "nagarikta", "national id", "birth"],
            "land_property": ["जग्गा", "jagga", "land", "namasari"],
        })
        result = clf.predict("नागरिकता कार्ड कसरी लिने?")
        assert result.domain_id == "identity_civil_registration"

    def test_land_keywords(self):
        clf = KeywordClassifier({
            "identity_civil_registration": ["नागरिकता", "nagarikta", "national id"],
            "land_property": ["जग्गा", "jagga", "land", "namasari"],
        })
        result = clf.predict("जग्गा नामसारी कसरी गर्ने?")
        assert result.domain_id == "land_property"

    def test_no_match(self):
        clf = KeywordClassifier({
            "identity_civil_registration": ["nagarikta"],
        })
        result = clf.predict("random unrelated text")
        assert result.domain_id == "unknown"


class TestRetrievalMetrics:
    def test_precision_at_k(self):
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = ["a", "c", "e"]
        assert compute_precision_at_k(retrieved, relevant, 1) == 1.0
        assert compute_precision_at_k(retrieved, relevant, 3) == 2 / 3
        assert compute_precision_at_k(retrieved, relevant, 5) == 3 / 5

    def test_recall_at_k(self):
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = ["a", "c", "e"]
        assert compute_recall_at_k(retrieved, relevant, 1) == 1 / 3
        assert compute_recall_at_k(retrieved, relevant, 3) == 2 / 3
        assert compute_recall_at_k(retrieved, relevant, 5) == 1.0

    def test_mrr(self):
        retrieved = ["a", "b", "c"]
        relevant = ["b"]
        assert compute_mrr(retrieved, relevant) == 1 / 2

        retrieved = ["a", "b", "c"]
        relevant = ["a"]
        assert compute_mrr(retrieved, relevant) == 1.0

        retrieved = ["a", "b", "c"]
        relevant = ["d"]
        assert compute_mrr(retrieved, relevant) == 0.0

    def test_empty_retrieved(self):
        assert compute_precision_at_k([], ["a"], 5) == 0.0
        assert compute_recall_at_k([], ["a"], 5) == 0.0
        assert compute_mrr([], ["a"]) == 0.0


class TestEvaluateDomainClassification:
    def test_perfect_accuracy(self):
        labels = ["a", "b", "c", "a", "b"]
        result = evaluate_domain_classification(labels, labels)
        assert result["accuracy"] == 1.0
        assert result["f1"] == 1.0

    def test_zero_accuracy(self):
        true = ["a", "b", "c"]
        pred = ["b", "c", "a"]
        result = evaluate_domain_classification(true, pred)
        assert result["accuracy"] == 0.0


class TestJSONUtils:
    def test_save_and_load_json(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name
        try:
            data = {"key": "value", "list": [1, 2, 3]}
            save_json(data, path)
            loaded = load_json(path)
            assert loaded == data
        finally:
            Path(path).unlink()

    def test_save_and_load_yaml(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            path = f.name
        try:
            data = {"key": "value", "nested": {"a": 1}}
            save_yaml(data, path)
            loaded = load_yaml(path)
            assert loaded == data
        finally:
            Path(path).unlink()


class TestFindRepoRoot:
    def test_returns_path(self):
        result = find_repo_root()
        assert isinstance(result, Path)
