# Project: Agricultural Advisory Assistant — RAG vs Fine-Tuning

Thesis: RAG supplies knowledge; fine-tuning supplies behavior. Prove it with a 2×2
(base/fine-tuned × RAG off/on), scored on one evaluation harness.

Crops in scope: **rice, wheat, tomato**.

Build order: data → corpus → RAG v1 → NLP layer (intent + NER/IE + topics) →
gold set → eval harness → SFT data → fine-tune → run 2×2 → guardrails/serve → writeup.

Full spec: `ag-advisory-build-guide-FINAL.md` (read before starting any phase not yet summarized here).

## Rules
- Safety first: no ungrounded dosages/chemicals; abstain when retrieved context is thin; every answer cites sources + carries a "verify locally" disclaimer.
- Keep train data (`data/finetune/`) and eval data (`eval/datasets/gold.jsonl`) strictly separate — assert zero overlap wherever both are built.
- Fine-tuning teaches format/citation-discipline/abstention, NOT facts.
- RAGAS evaluates the RAG arm only; use custom/seqeval/sklearn for intent, NER, abstention, dosage-hallucination.
- Every phase ships tests and updates its report in `reports/`.
- Secrets via env vars only (`.env`, never committed).

## Stack
transformers+peft+bitsandbytes+trl (QLoRA, 3B), FAISS/Chroma, BGE/E5 embeddings,
scikit-learn + spaCy + BERTopic (intent/NER/topics), RAGAS + seqeval + custom eval,
Anthropic API as LLM-judge, FastAPI + Streamlit, W&B.

## Status
See `PROGRESS.md` for current phase and what's done.
