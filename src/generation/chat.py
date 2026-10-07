"""Phase 6: Citation-grounded generation — evidence-only answers with source citations."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.utils import get_settings, load_json, find_repo_root

logger = logging.getLogger(__name__)


@dataclass
class GeneratedResponse:
    """A generated response with citations."""
    answer: str
    citations: list[dict[str, Any]]
    domain_id: str
    confidence: float
    refused: bool = False
    refusal_reason: str = ""


class LLMClient:
    """Unified LLM client for OpenAI, OpenRouter, Anthropic, and Ollama."""

    def __init__(self, provider: str, model: str, api_key: str = "", base_url: str = ""):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

    def generate(self, messages: list[dict[str, str]], temperature: float = 0.1, max_tokens: int = 1024) -> str:
        if self.provider == "openai":
            return self._call_openai(messages, temperature, max_tokens)
        elif self.provider == "openrouter":
            return self._call_openrouter(messages, temperature, max_tokens)
        elif self.provider == "anthropic":
            return self._call_anthropic(messages, temperature, max_tokens)
        elif self.provider == "ollama":
            return self._call_ollama(messages, temperature, max_tokens)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _call_openai(self, messages: list[dict[str, str]], temperature: float, max_tokens: int) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model, messages=messages, temperature=temperature, max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    def _call_openrouter(self, messages: list[dict[str, str]], temperature: float, max_tokens: int) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key, base_url="https://openrouter.ai/api/v1")
        response = client.chat.completions.create(
            model=self.model, messages=messages, temperature=temperature, max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    def _call_anthropic(self, messages: list[dict[str, str]], temperature: float, max_tokens: int) -> str:
        import anthropic
        client = anthropic.Anthropic(api_key=self.api_key)
        system = ""
        user_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                user_messages.append(msg)
        response = client.messages.create(
            model=self.model, system=system, messages=user_messages,
            temperature=temperature, max_tokens=max_tokens,
        )
        return response.content[0].text if response.content else ""

    def _call_ollama(self, messages: list[dict[str, str]], temperature: float, max_tokens: int) -> str:
        import ollama
        response = ollama.chat(
            model=self.model, messages=messages,
            options={"temperature": temperature, "num_predict": max_tokens},
        )
        return response["message"]["content"]


class CitationGroundedGenerator:
    """Generates citation-grounded answers from retrieved context using an LLM."""

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        system_language: str = "nepali",
    ) -> None:
        settings = get_settings()
        self.provider = provider
        self.model = model
        self.system_language = system_language
        self.temperature = settings["generation"]["temperature"]
        self.max_tokens = settings["generation"]["max_tokens"]
        self.refusal_message = settings["generation"]["refusal"]["message"]
        self.llm = self._init_llm()

    def _init_llm(self) -> LLMClient | None:
        provider = os.getenv("LLM_PROVIDER", self.provider)
        model = os.getenv("LLM_MODEL", self.model)
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY", "")
            if not api_key or api_key == "your-openai-api-key-here":
                logger.warning("OPENAI_API_KEY not set — using fallback generation")
                return None
            return LLMClient("openai", model, api_key=api_key)
        elif provider == "openrouter":
            api_key = os.getenv("OPENROUTER_API_KEY", "")
            if not api_key or api_key == "your-openrouter-api-key-here":
                logger.warning("OPENROUTER_API_KEY not set — using fallback generation")
                return None
            return LLMClient("openrouter", model, api_key=api_key)
        elif provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY", "")
            if not api_key or api_key == "your-anthropic-api-key-here":
                logger.warning("ANTHROPIC_API_KEY not set — using fallback generation")
                return None
            return LLMClient("anthropic", model, api_key=api_key)
        elif provider == "ollama":
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            model = os.getenv("OLLAMA_MODEL", "llama3")
            return LLMClient("ollama", model, base_url=base_url)
        return None

    def _build_system_prompt(self) -> str:
        return """तपाईं नेपाल सेवा AI हुनुहुन्छ — नेपाल सरकारी सेवाहरूको बारेमा जानकारी दिने कृत्रिम बुद्धिमत्ता सहायक।

नियमहरू:
१. तपाईंले केवल तल दिइएको स्रोत सामग्री (Source Context) को आधारमा उत्तर दिनुपर्छ।
२. यदि स्रोत सामग्रीमा उत्तर छैन भने, "माफ गर्नुहोस्, मसँग यस प्रश्नको उत्तर दिने पर्याप्त प्रमाण छैन।" भन्नुहोस्।
३. कहिल्यै नै बनावट जानकारी नदिनुहोस् — न त शुल्क, न त समय, न त कानुनी प्रावधान।
४. उत्तर स्पष्ट, संक्षिप्त र सजिलो भाषामा दिनुहोस्।
५. प्रश्नको भाषा अनुसार उत्तर दिनुहोस् (नेपाली प्रश्न → नेपाली उत्तर, अंग्रेजी प्रश्न → अंग्रेजी उत्तर)।
६. उत्तरको अन्तमा स्रोतहरूको सूची दिनुहोस्।
७. यदि स्रोतहरूमा विपरीत जानकारी छ भने, दुवै देखाउनुहोस्।

उत्तर ढाँचा:
- सङ्क्षिप्त उत्तर (२-३ हरफ)
- आवश्यक कागजातहरू (भएमा)
- प्रक्रिया (भएमा)
- शुल्क (भएमा)
- स्रोतहरू"""

    def _build_user_prompt(self, query: str, context: str) -> str:
        return f"""प्रश्न: {query}

स्रोत सामग्री (Source Context):
{context}

कृपया माथिको स्रोत सामग्रीको आधारमा प्रश्नको उत्तर दिनुहोस्।"""

    def _clean_text(self, text: str) -> str:
        """Remove punctuation for comparison."""
        return re.sub(r'[।?!.,;:\[\]{}()@#$%^&*_+=\-|\/\"\'~`]', '', text.lower())

    def _keyword_score(self, query: str, text: str) -> int:
        """Score how relevant a text is to the query."""
        q = self._clean_text(query)
        t = self._clean_text(text)

        score = 0
        for word in q.split():
            if word in t:
                score += 1
        if q in t:
            score += 10
        return score

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences (handles Nepali and English)."""
        # Split on Nepali danda, English period, question mark, exclamation
        sentences = re.split(r'[।!?\.]\s*', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    def _score_sentence(self, sentence: str, query: str) -> float:
        """Score a sentence's relevance to the query."""
        q = self._clean_text(query)
        s = self._clean_text(sentence)

        query_words = set(q.split())
        sentence_words = set(s.split())

        if not query_words:
            return 0.0

        # Jaccard similarity
        intersection = query_words & sentence_words
        union = query_words | sentence_words
        jaccard = len(intersection) / len(union) if union else 0.0

        # Word overlap ratio
        overlap = len(intersection) / len(query_words) if query_words else 0.0

        # Phrase bonus
        phrase_bonus = 0.0
        if q in s:
            phrase_bonus = 0.5

        return jaccard * 0.3 + overlap * 0.5 + phrase_bonus

    def _extract_key_info(self, text: str) -> dict[str, list[str]]:
        """Extract key information like fees, documents, time periods."""
        info = {
            'fees': [],
            'documents': [],
            'time': [],
            'process': [],
        }

        # Fee patterns (Nepali and English)
        fee_patterns = [
            r'रु[\s\.]?\d+',
            r'Rs[\s\.]?\d+',
            r'शुल्क[:\s]*[^\n]+',
            r'fee[:\s]*[^\n]+',
            r'\d+\s*रुपैयाँ',
        ]
        for pattern in fee_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            info['fees'].extend(matches)

        # Document patterns
        doc_patterns = [
            r'कागजात[:\s]*[^\n]+',
            r'दस्तावेज[:\s]*[^\n]+',
            r'document[:\s]*[^\n]+',
            r'प्रमाणपत्र[:\s]*[^\n]+',
        ]
        for pattern in doc_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            info['documents'].extend(matches)

        # Time patterns
        time_patterns = [
            r'\d+\s*दिन',
            r'\d+\s*महिना',
            r'\d+\s*वर्ष',
            r'\d+\s*days?',
            r'\d+\s*months?',
            r'समय[:\s]*[^\n]+',
        ]
        for pattern in time_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            info['time'].extend(matches)

        return info

    def _parse_qa_pairs(self, text: str) -> list[dict[str, str]]:
        """Parse FAQ-style text into Q&A pairs."""
        qa_pairs = []

        # Pattern: "प्रश्न: ... उत्तर: ..." (Nepali FAQ format)
        # Also handles "प्रश्न: ... उत्तर: ..." with various numbering
        patterns = [
            # Nepali: प्रश्न: ... उत्तर: ...
            r'प्रश्न[:\s]*(.+?)\s*उत्तर[:\s]*(.+?)(?=प्रश्न[:\s]|$)',
            # English: Question: ... Answer: ...
            r'Question[:\s]*(.+?)\s*Answer[:\s]*(.+?)(?=Question[:\s]|$)',
            # Numbered: १. प्रश्न: ... उत्तर: ...
            r'[०-९\d]+[.\s]*प्रश्न[:\s]*(.+?)\s*उत्तर[:\s]*(.+?)(?=[०-९\d]+[.\s]*प्रश्न[:\s]|$)',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            for q, a in matches:
                q_clean = q.strip()
                a_clean = a.strip()
                if len(q_clean) > 5 and len(a_clean) > 10:
                    qa_pairs.append({
                        'question': q_clean,
                        'answer': a_clean,
                    })

        return qa_pairs

    def _score_qa_relevance(self, query: str, qa_question: str) -> float:
        """Score how relevant a QA pair's question is to the user's query."""
        q = self._clean_text(query)
        qa_q = self._clean_text(qa_question)

        if not q or not qa_q:
            return 0.0

        query_words = set(q.split())
        qa_words = set(qa_q.split())

        if not query_words:
            return 0.0

        # Jaccard similarity
        intersection = query_words & qa_words
        union = query_words | qa_words
        jaccard = len(intersection) / len(union) if union else 0.0

        # Word overlap ratio
        overlap = len(intersection) / len(query_words) if query_words else 0.0

        # Phrase bonus
        phrase_bonus = 0.0
        if q in qa_q or qa_q in q:
            phrase_bonus = 0.5

        # Key term bonus (Nepali + English key terms)
        key_terms = {
            'नागरिकता', 'nagarikta', 'citizenship', 'राष्ट्रिय', 'national',
            'परिचयपत्र', 'id', 'card', 'बनाउन', 'कसरी', 'कति',
            'समय', 'शुल्क', 'कागजात', 'दस्तावेज', 'प्रक्रिया',
            'जग्गा', 'नामसारी', 'land', 'ssf', 'सामाजिक',
            'श्रम', 'labour', 'स्वास्थ्य', 'health', 'बीमा', 'insurance',
        }
        key_matches = sum(1 for term in key_terms if term in q and term in qa_q)
        key_bonus = key_matches * 0.1

        return jaccard * 0.2 + overlap * 0.4 + phrase_bonus + key_bonus

    def _fallback_generate(self, query: str, retrieval_results: list[Any]) -> str:
        """Generate answer from context without LLM — intelligent Q&A extraction."""
        if not retrieval_results:
            return self.refusal_message

        # Parse Q&A pairs from all results
        all_qa_pairs = []
        for result in retrieval_results:
            qa_pairs = self._parse_qa_pairs(result.text)
            for qa in qa_pairs:
                relevance = self._score_qa_relevance(query, qa['question'])
                all_qa_pairs.append({
                    **qa,
                    'relevance': relevance,
                    'source': result.source_name,
                    'url': result.source_url,
                })

        # Sort by relevance
        all_qa_pairs.sort(key=lambda x: x['relevance'], reverse=True)

        # Take top Q&A pairs with good relevance
        top_qa = [qa for qa in all_qa_pairs if qa['relevance'] > 0.15][:3]

        if not top_qa:
            # Fallback: extract key sentences from raw text
            return self._extract_key_sentences(query, retrieval_results)

        # Build clean answer from Q&A pairs
        parts = []
        for i, qa in enumerate(top_qa, 1):
            answer_text = qa['answer']
            # Clean up the answer text
            answer_text = re.sub(r'\s+', ' ', answer_text).strip()
            # Truncate if too long
            if len(answer_text) > 500:
                answer_text = answer_text[:500] + "..."
            parts.append(answer_text)

        answer = "\n\n".join(parts)

        # Add source attribution
        sources_used = list(set(qa['source'] for qa in top_qa))
        if sources_used:
            answer += f"\n\nस्रोत (Sources): {', '.join(sources_used)}"

        return answer

    def _extract_key_sentences(self, query: str, retrieval_results: list[Any]) -> str:
        """Extract key sentences from raw text when no Q&A pairs found."""
        all_sentences = []
        for result in retrieval_results:
            sentences = self._split_into_sentences(result.text)
            for sent in sentences:
                score = self._score_sentence(sent, query)
                all_sentences.append({
                    'text': sent,
                    'score': score,
                    'source': result.source_name,
                })

        all_sentences.sort(key=lambda x: x['score'], reverse=True)
        top_sentences = [s for s in all_sentences if s['score'] > 0.1][:3]

        if not top_sentences:
            return self.refusal_message

        parts = [s['text'] for s in top_sentences]
        answer = " ".join(parts)

        sources_used = list(set(s['source'] for s in top_sentences))
        if sources_used:
            answer += f"\n\nस्रोत (Sources): {', '.join(sources_used)}"

        return answer

    def generate(
        self,
        query: str,
        retrieval_results: list[Any],
        domain_id: str = "",
    ) -> GeneratedResponse:
        """Generate a citation-grounded answer."""
        if not retrieval_results:
            return GeneratedResponse(
                answer=self.refusal_message,
                citations=[],
                domain_id=domain_id,
                confidence=0.0,
                refused=True,
                refusal_reason="No results",
            )

        # Format citations
        citations = []
        for i, result in enumerate(retrieval_results, 1):
            citations.append({
                "index": i,
                "name": result.source_name,
                "url": result.source_url,
                "section": result.section,
                "text_snippet": result.text[:200],
            })

        # Build context for LLM
        context_parts = []
        for i, result in enumerate(retrieval_results, 1):
            context_parts.append(f"[स्रोत {i}] {result.source_name}\nURL: {result.source_url}\n\n{result.text[:800]}")
        context = "\n\n---\n\n".join(context_parts)

        # Try LLM generation
        if self.llm:
            try:
                messages = [
                    {"role": "system", "content": self._build_system_prompt()},
                    {"role": "user", "content": self._build_user_prompt(query, context)},
                ]
                answer = self.llm.generate(
                    messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                confidence = sum(r.score for r in retrieval_results) / len(retrieval_results)
                return GeneratedResponse(
                    answer=answer,
                    citations=citations,
                    domain_id=domain_id,
                    confidence=confidence,
                    refused=False,
                )
            except Exception as e:
                logger.error("LLM generation failed: %s — falling back to keyword extraction", e)

        # Fallback to keyword-based extraction
        answer = self._fallback_generate(query, retrieval_results)
        confidence = sum(r.score for r in retrieval_results) / len(retrieval_results) if retrieval_results else 0.0

        return GeneratedResponse(
            answer=answer,
            citations=citations,
            domain_id=domain_id,
            confidence=confidence,
            refused=False,
        )


class ChatSession:
    """Interactive chat session with conversation history."""

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
    ) -> None:
        settings = get_settings()
        self.generator = CitationGroundedGenerator(
            provider=provider,
            model=model,
            system_language=settings["generation"]["system_language"],
        )
        self.retriever = self._load_retriever()
        self.classifier = self._load_classifier()
        self.history: list[dict[str, str]] = []

    def _load_retriever(self):
        from src.retrieval.indexer import HybridRetriever
        settings = get_settings()
        retriever = HybridRetriever(
            bm25_weight=settings["retrieval"]["hybrid"]["bm25_weight"],
            dense_weight=settings["retrieval"]["hybrid"]["dense_weight"],
            embedding_model=settings["retrieval"]["embedding_model"],
            persist_dir=settings["retrieval"]["chromadb"]["persist_dir"],
        )
        chunks_path = find_repo_root() / settings["paths"]["processed_dir"] / "all_chunks.json"
        if chunks_path.exists():
            chunks = load_json(chunks_path)
            if chunks:
                retriever.build_index(chunks)
        return retriever

    def _load_classifier(self):
        from src.domain_classifier.classifier import EmbeddingClassifier
        settings = get_settings()
        classifier = EmbeddingClassifier(settings["classifier"]["embedding_model"])
        classifier_path = find_repo_root() / "data" / "index" / "domain_classifier.pkl"
        if classifier_path.exists():
            classifier.load(classifier_path)
        return classifier

    def _keyword_domain(self, query: str) -> str:
        """Fast keyword-based domain classification."""
        q = query.lower()
        
        # Identity & Civil Registration
        if any(kw in q for kw in ['नागरिकता', 'nagarikta', 'national id', 'राष्ट्रिय परिचय',
                                   'परिचयपत्र', 'जन्म दर्ता', 'मृत्यु दर्ता', 'विवाह दर्ता']):
            return "identity_civil_registration"
        
        # Land & Property
        if any(kw in q for kw in ['जग्गा', 'जमिन', 'नामसारी', 'जग्गा नामसारी', 'land',
                                   'property', 'jagga', 'namasari', 'lalpurja', 'भूमि']):
            return "land_property"
        
        # Employment & Labour
        if any(kw in q for kw in ['सामाजिक सुरक्षा', 'ssf', 'वैदेशिक', 'bidesh rojgar',
                                   'foreign employment', 'श्रम स्वीकृति', 'dofe']):
            return "employment_labour"
        
        # Health & Social Welfare
        if any(kw in q for kw in ['स्वास्थ्य', 'health', 'swasthya', 'medical', 'बृद्धा',
                                   'भत्ता', 'बीमा', 'bima', 'insurance', 'health insurance']):
            return "health_social_welfare"
        
        return ""

    def reset(self) -> None:
        self.history = []

    def ask(self, query: str) -> GeneratedResponse:
        # 1. Keyword-based domain classification
        domain_id = self._keyword_domain(query)
        
        # 2. Retrieve results
        if self.retriever.bm25.index:
            all_results = self.retriever.search(query, top_k=10)
            
            if domain_id:
                # Prioritize same-domain results
                domain_results = [r for r in all_results if r.domain_id == domain_id]
                other_results = [r for r in all_results if r.domain_id != domain_id]
                retrieval_results = (domain_results[:3] + other_results[:2])[:5]
            else:
                retrieval_results = all_results[:5]
        else:
            retrieval_results = []
            domain_id = ""

        # 3. Generate answer
        response = self.generator.generate(query, retrieval_results, domain_id)
        
        self.history.append({"role": "user", "content": query})
        self.history.append({"role": "assistant", "content": response.answer})
        
        return response
