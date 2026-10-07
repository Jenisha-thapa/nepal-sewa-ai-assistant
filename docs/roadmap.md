# Roadmap — reconciled with the submitted proposal

This supersedes the domain list and eval metrics from the earlier general
architecture document with what's actually in
`Domain-Aware_RAG_Agent_Proposal_Final.docx`. See that doc's own roadmap
for the deeper "why" behind each architectural choice (A/B/C labeling,
technology trade-offs) — this file tracks *what's locked in* and *what's
still open*.

## Locked in (per proposal)

- 4 domains: identity_civil_registration, land_property, employment_labour, health_social_welfare
- Domain routing is a required, evaluated component (vs. keyword-matching baseline)
- Three retrieval configs required: BM25, dense, hybrid — compared per domain partition
- Structure-aware chunking (preserve title/section/FAQ-pair structure, not fixed-length only)
- Generation: plain Nepali, evidence-only, cites source + domain, refuses when evidence insufficient
- Eval set: 100 hand-annotated Nepali questions, 25/domain, gold source + gold domain label per question
- Metrics: domain classification (accuracy/precision/recall/F1), retrieval (P@K, R@K, MRR — no nDCG),
  generation (groundedness, relevance, citation correctness, clarity — human-evaluated subset),
  refusal rate on out-of-scope/insufficient-evidence questions
- No fine-tuning as a core requirement; standard laptops, no GPU cluster
- 6-month timeline: M1-2 planning/sourcing, M3-4 build, M5-6 eval/deploy

## Open decisions (surface these before Phase 4)

1. **English/mixed-language scope** — proposal's eval set is Nepali-only. Confirm whether
   English/mixed queries are still in scope; this changes the embedding model shortlist.
2. **Vector store** — FAISS vs. ChromaDB (both listed as acceptable in Resource Requirements).
   ChromaDB gives easier metadata filtering per domain partition out of the box; FAISS is
   what the NVIDIA course teaches directly. Either is defensible — pick one and justify it
   in the report rather than switching mid-project.
3. **LLM choice** — "a suitable accessible LLM API or lightweight open-weight model" is
   still unspecified. Needs a short bake-off on Nepali generation fluency before Phase 7.
4. **Citation completeness** — the proposal evaluates citation *correctness* but doesn't
   separately name citation *completeness* (are ALL claims covered, not just the cited ones
   accurate). Worth adding as a secondary metric — cheap to add once correctness scoring
   exists, and strengthens the "citation-grounded" claim in your abstract.
5. **Domain-classifier architecture** — proposal specifies "a lightweight classifier... using
   the same multilingual embeddings as the retrieval component" vs. a keyword baseline.
   Concrete choice (logistic regression / shallow network / nearest-centroid) still open.

## Immediate next steps (Month 1-2 work, in order)

1. Fill in the `# TODO` gaps in `config/domains.yaml` — two domains currently only have
   bare portal roots (dolma.gov.np, dofe.gov.np, ssf.gov.np). Browse each portal, identify
   the actual FAQ/procedure pages, add them as explicit sources.
2. Confirm official Nepali service names for each domain (also `# TODO` in the config).
3. Run `src/ingestion/fetch_sources.py` against the confirmed source list.
4. Manually spot-check fetched content for scanned/image-only PDFs — flag which sources
   need the OCR path before Phase 2 processing.
5. Resolve the open decisions above — at minimum #1 and #2, since they block Phase 4/7 choices.
6. Begin the 100-question eval set alongside sourcing, not after — annotators can draft
   questions per domain as documents come in, rather than waiting for the full corpus.
