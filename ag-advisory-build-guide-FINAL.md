# Agricultural Advisory Assistant — RAG vs Fine-Tuning, with Real NLP & Evaluation
### End-to-end build guide for Claude Code

A crop/agronomy advisory assistant that answers questions with **source-cited** advice — and, when its corpus can't answer, **says so** instead of inventing a pesticide dose. It's built to demonstrate the skills that matter for **both data-scientist and AI-engineer roles**, and to prove them with numbers.

Work top to bottom. Every phase has a **goal**, what to build, its **acceptance criteria**, and a **ready-to-paste Claude Code prompt**. Gate each phase (implement → test → check → commit) before the next.

---

## 0. What you're building

The system demonstrates **four** competencies, each measured:

- **RAG** — retrieval over a real agronomy corpus supplies the *facts* (thresholds, treatments, varieties), with citations.
- **Fine-tuning (LoRA/QLoRA)** — teaches a small open model the *behavior*: advisory format, citation discipline, domain tone, and safe **abstention**.
- **Core NLP modeling** — supervised text classification (intent), sequence labeling (agronomic NER + information extraction), and unsupervised text analysis (topic modeling): the "model and analyze text as data" skills DS roles test, not just LLM orchestration.
- **Rigorous evaluation** — a harness measuring all of it across a controlled experiment, reporting honest findings.

### The headline experiment — a 2×2 you evaluate identically

|  | No RAG | With RAG |
|---|---|---|
| **Base model** | control | does retrieval help? |
| **Fine-tuned** | did FT "learn facts"? (it hallucinates) | the combination |

That result table is the deliverable. Most candidates build one cell; you build the comparison.

### The thesis (state it explicitly in the README)

RAG and fine-tuning solve **different** problems: RAG supplies *knowledge*, fine-tuning supplies *behavior*. The common mistake — "fine-tune the model on our documents so it learns agronomy" — bakes in confident hallucination. You'll show that **fine-tuning for form + RAG for facts** beats it, with metrics.

### The pipeline

```
User question
  → Intent classifier (route; out-of-scope → abstain)
  → Agent orchestrator (plans, calls tools)
      → Retrieval (RAG)  →  NER/IE (structured filter + safety)
  → Grounded generation (cited advisory)  →  safety check  →  human view
```

---

## 1. Safety frame (read first — agriculture makes evaluation real)

Wrong output here can cause real harm: a hallucinated pesticide dose, an out-of-region recommendation, a missed toxicity warning. That raises the stakes *and* is your best portfolio story. Bake in from day one:

- **Abstention is a first-class behavior.** If retrieved context doesn't cover the crop/region/pest, the correct answer is "I don't have reliable information on that — consult your local KVK/extension officer," never a guessed dosage.
- **No un-grounded specifics.** Every dosage, chemical, or threshold in an answer must trace to a retrieved passage — enforced *and measured*.
- **Always cite, always disclaim.** Every answer carries sources and a "verify locally before applying" note.

Abstention accuracy and dosage-hallucination rate are metrics in §7, not afterthoughts.

---

## 2. Data — where to get it

"The data" is three distinct assets: (1) a **document corpus** for RAG, (2) **real farmer questions** for intent/topics/eval, and (3) **labeled sets** (gold eval, NER spans, fine-tuning examples) that you *construct* from (1) and (2). Building #3 is part of the project's value. The good news: for Indian agriculture, #1 and #2 are free government/university material.

### 2a. Document corpus (the RAG knowledge base)

- **TNAU Agritech Portal** — `https://agritech.tnau.ac.in/` — crop production & crop protection sections with downloadable per-crop pest/disease PDFs, IPM guides, and crop guides. Your best single source.
- **TNAU Crop Production Guide** — `https://agritech.tnau.ac.in/pdf/AGRICULTURE.pdf` — one comprehensive PDF: recommended varieties + pest/disease management.
- **ICAR — All Publications** — `https://icar.org.in/en/all-publications` — handbooks, field-crop textbooks, package-of-practices volumes.
- **State Agricultural University "Package of Practices"** — most states publish one; pick the one matching your crops/region.
- **FAO** crop & IPM guides — for international grounding if wanted.

Scope to **2–4 crops** (e.g., rice, wheat, tomato). Record each document's licence in `data/corpus/sources.csv` and cite it.

### 2b. Real farmer questions — the goldmine (solves the hardest problem)

Most RAG projects fabricate their eval questions. You don't have to.

- **Kisan Call Centre (KCC)** — Government of India open data: real farmer queries + Farm Tele Advisor answers, tagged by crop, query type, season, and district.
  - Catalog: `https://www.data.gov.in/catalog/district-wise-and-month-wise-queries-farmers-kisan-call-centre-kcc`
  - Cleaned Q&A mirror (faster start): `https://www.kaggle.com/datasets/daskoushik/farmers-call-query-data-qa/data`

Why it's ideal for *this* project: real questions in farmers' own words → realistic eval + topic-modeling input; the crop/query-type tags are ready-made **weak labels** for your intent classifier; the query→answer structure seeds both eval and fine-tuning examples.

### 2c. For the stretch CV goal (leaf photo → diagnosis)

**PlantVillage** (clean lab images) and **PlantDoc** (real field images) — both free on Kaggle/GitHub.

**Acceptance criteria for §2:** corpus PDFs for your crops downloaded to `data/raw/` with a `sources.csv`; a KCC slice pulled to `data/nlp/kcc_raw.*`.

**Claude Code prompt:**
> Write `src/data/fetch.py` that (a) pulls a slice of the Kisan Call Centre dataset (via the data.gov.in API or the Kaggle mirror) filtered to my chosen crops/states into `data/nlp/kcc_raw.parquet`, and (b) records a `data/corpus/sources.csv` template for the TNAU/ICAR PDFs I place in `data/raw/`. Print counts and a sample of KCC queries with their tags.

---

## 3. Corpus ingestion

**Goal:** clean, chunked, retrievable documents — with tables preserved.

> ⚠️ Agronomy PDFs are full of **dosage/schedule tables**, and those tables are the high-stakes facts. Naive text extraction shreds them. Handle tables explicitly (extract as structured rows; keep them intact as retrievable chunks).

**Acceptance criteria:** `data/corpus/` holds cleaned docs; tables preserved as structured chunks; every doc in `sources.csv`.

**Claude Code prompt:**
> Set up `src/ingest/`: parse the PDFs in `data/raw/` with a table-aware extractor, clean headers/footers, output cleaned docs to `data/corpus/`, and chunk so dosage/schedule tables stay intact as single chunks. Print docs, chunks, and table-chunk counts.

---

## 4. RAG pipeline v1 (the retrieval arm)

**Goal:** grounded, cited answers from the **base** model.

- **Embeddings:** BGE / E5 / GTE (multilingual variant if you'll do Hindi later).
- **Index:** FAISS or Chroma.
- **Retriever:** dense → add **hybrid (BM25 + dense)** (crop/pest/chemical names are exact-match-heavy); optional **cross-encoder reranker** (ablate in eval).
- **Generation:** answer *only* from retrieved chunks, structured + cited, abstain when context is thin.

**Acceptance criteria:** a question returns an answer plus the exact source chunks used; retrieved chunks are inspectable per query.

**Claude Code prompt:**
> Build `src/rag/`: embedding + FAISS index over `data/corpus/`, a retriever with dense and hybrid (BM25+dense) modes and an optional config-flagged cross-encoder reranker, and `generate(question)` that answers only from retrieved chunks in a structured cited format and abstains on weak retrieval. Add a CLI printing answer + sources.

---

## 5. NLP modeling layer — the data-science depth

Three components that show text *modeling & analysis* (not just LLM orchestration), each of which also improves the system and adds an eval dimension. Code in `src/nlp/`, data in `data/nlp/`.

### 5a. Query intent classification (supervised text classification)

**Goal:** classify each question into `pest_disease`, `nutrient_fertilizer`, `irrigation`, `variety_seed`, `weather_season`, `market_price`, `out_of_scope`.

**Why DS-relevant:** the canonical supervised-classification task. Build a **TF-IDF + linear** baseline *and* a **fine-tuned encoder** (DistilBERT / IndicBERT), report macro-F1, confusion matrix, error analysis. Seed labels from KCC's crop/query-type tags (clean the weak labels).

**Integration:** routes queries (out_of_scope → abstain; intent → narrows retrieval); yields analytics on what users ask.

**Acceptance criteria:** labeled query set split with no leakage; both models trained; `reports/intent_eval.md` with per-class F1 + confusion matrix; router uses it at inference.

**Claude Code prompt:**
> Build `src/nlp/intent.py`: derive a labeled query set in `data/nlp/intents.jsonl` from the KCC slice (7 classes incl. out_of_scope, using crop/query-type tags as weak labels I then clean), train a TF-IDF+linear baseline and a fine-tuned DistilBERT/IndicBERT classifier, evaluate both into `reports/intent_eval.md`, and wire the classifier into the router.

### 5b. Agronomic NER & information extraction (sequence labeling)

**Goal:** extract `Crop`, `Pest`, `Disease`, `Symptom`, `Chemical`, `Dosage`, `Nutrient`, `GrowthStage` from queries and documents.

**Why DS-relevant:** custom-domain token classification + IE; evaluate with entity-level P/R/F1. spaCy (gazetteers + statistical) to start; optionally fine-tune a token-classification transformer and compare.

**Integration (payoff):** (1) makes the **dosage-grounding safety check principled** — extract Dosage/Chemical entities from an answer, verify each is in retrieved context (vs brittle regex); (2) **structured/filtered retrieval** by crop+pest; (3) IE over the corpus builds a `crop → pest → treatment → dosage` knowledge base as its own tool.

**Acceptance criteria:** annotated sample; NER model with entity-level F1 in `reports/ner_eval.md`; §7 safety check consumes its output.

**Claude Code prompt:**
> Build `src/nlp/ner.py`: define the entity schema, build gazetteers from the corpus, annotate a sample into `data/nlp/ner.jsonl`, train a spaCy NER (optionally a transformer token-classifier), report entity-level F1 in `reports/ner_eval.md`, expose `extract(text)`, and refactor the dosage-grounding check to use extracted entities.

### 5c. Corpus & query topic modeling (unsupervised text analysis)

**Goal:** discover thematic structure of corpus and queries with **BERTopic** (or LDA).

**Why DS-relevant:** unsupervised NLP + "analyze a corpus → produce insight," the muscle DS interviews prize.

**Integration:** surfaces **coverage gaps** ("content over-indexes on pest management; irrigation is thin"), which explains *where the system abstains most* — a real finding for the writeup.

**Acceptance criteria:** `reports/topics.md` with topics, sizes, terms, and coverage-gap read vs the query distribution.

**Claude Code prompt:**
> Build `src/nlp/topics.py`: run BERTopic over corpus chunks and over the KCC query set, output topic labels/sizes/terms, cross-tabulate corpus vs query topics for coverage gaps, write `reports/topics.md`, and relate gaps to abstention rate by topic.

**Optional extensions:** multi-document **summarization** (extractive + abstractive, ROUGE + faithfulness); **multilingual** (language ID + cross-lingual retrieval for Hindi/regional).

---

## 6. Gold evaluation set (build before you fine-tune)

**Goal:** ~150–200 human-verified items — the foundation of every trustworthy number.

Three types: **answerable** (question + verified answer + source `source_ids`), **adversarial / out-of-corpus** (correct behavior = abstain), **safety-sensitive** (a dosage/chemical is the answer). Seed questions from **real KCC queries**; verify with your agronomy knowledge. Keep strictly separate from any fine-tuning data.

**Acceptance criteria:** `eval/datasets/gold.jsonl` exists, human-verified, all three types present, `source_ids` resolve to real chunks.

**Claude Code prompt:**
> Create `src/eval/build_gold.py`: draft candidate Q/A from the corpus + KCC queries (with source chunk IDs), plus adversarial out-of-corpus and safety-sensitive items, into `eval/datasets/gold.jsonl` for me to hand-verify via a small accept/edit/reject review CLI.

---

## 7. Evaluation harness ⭐ (the star)

The single most important idea: **the harness is the project; RAG and fine-tuning are things you measure.** And **RAGAS is one instrument in it, not the whole harness** — it evaluates only the RAG arm. Map each metric to the right tool:

| Metric | Measures | Tool |
|---|---|---|
| Faithfulness / groundedness | answer uses only retrieved facts | **RAGAS** |
| Answer relevancy | answer addresses the question | **RAGAS** |
| Context precision / recall | retriever pulls & ranks the right passages | **RAGAS** |
| recall@k, MRR | retrieval vs labeled `source_ids` (hard IR) | **custom** (against gold) |
| Intent accuracy | macro-F1, confusion matrix | **scikit-learn** `classification_report` |
| NER quality | entity-level P/R/F1 | **seqeval** / spaCy scorer |
| Abstention accuracy | correct "I don't know" on out-of-corpus | **custom** |
| Dosage-hallucination rate | every dosage/chemical traces to context | **custom** (uses §5b NER) |
| Advisory quality | actionable, correctly sequenced, safe tone | **LLM-as-judge** (G-Eval / Claude) + human spot-check |

Notes: use RAGAS's context metrics **and** the hard IR metrics — both together is more rigorous than either. **DeepEval** is a pytest-style alternative/complement worth naming (handy for CI). For the LLM judge, always **spot-check against your own labels** on a sample and report judge–human agreement — a mature signal.

Run every config through this harness. Output one row per config into `reports/eval_report.md`.

**Acceptance criteria:** one command scores a config and appends its row; you can quote concrete numbers ("recall@5 = 0.82; NER F1 = 0.88; 0 ungrounded dosages on 40 safety items").

**Claude Code prompt:**
> Build `src/eval/harness.py` taking a config (model + rag on/off + retriever variant) and scoring it on `gold.jsonl`: RAGAS (faithfulness, answer relevancy, context precision/recall), custom recall@k & MRR vs source_ids, abstention accuracy, and a dosage-hallucination check using the §5b NER. Integrate the intent (scikit-learn) and NER (seqeval) reports. Add an LLM-as-judge advisory-quality score (Anthropic API) with a human-review sample. Aggregate into `reports/eval_report.md`.

---

## 8. Fine-tuning data (teach behavior, not knowledge)

**Goal:** instruction examples teaching *form and safety*, not facts.

- **Answerable →** (question + retrieved context) → formatted, cited answer.
- **Out-of-corpus →** thin context → abstain + defer to local expert.
- **Safety →** quote the dosage *from context*, never beyond.

A few hundred–~1–2k high-quality examples suffice for LoRA. **Quality > quantity.** Strictly no overlap with `gold.jsonl`.

**Acceptance criteria:** `data/finetune/train.jsonl` (+ val split), format-consistent, all three case types, zero gold overlap.

**Claude Code prompt:**
> Create `src/finetune/build_sft_data.py` generating LoRA examples (answerable / abstention / safety) into `data/finetune/train.jsonl` with a val split; assert zero overlap with `eval/datasets/gold.jsonl`; print class balance.

---

## 9. Fine-tune the small model (LoRA/QLoRA)

- **Model:** 1–8B open — Llama-3.2-3B, Qwen2.5-3B/7B, or Phi-3.5-mini. Start at **3B** (QLoRA fits a free Colab/Kaggle **T4**).
- **Stack:** `transformers` + `peft` + `bitsandbytes` + `trl` (`SFTTrainer`).
- **Track** with W&B / MLflow. Save adapters; keep base frozen.

**Acceptance criteria:** training completes, loss logged, adapters in `models/adapters/`, eyeball on 5 held-out prompts shows better format/abstention vs base.

**Claude Code prompt:**
> Build `src/finetune/train.py`: QLoRA fine-tune of a 3B model on `data/finetune/train.jsonl` (transformers+peft+bitsandbytes+trl) with W&B logging, checkpoints, adapters to `models/adapters/`; add `predict.py` (base+adapter) and a Colab/Kaggle `notebooks/finetune.ipynb`.

---

## 10. Run the 2×2 and write the report

Score all four configs (base/FT × RAG on/off) through §7. **Report these honest findings** (they're the credibility):

- RAG ≫ no-RAG on factual correctness.
- Fine-tuned **without** RAG hallucinates confident specifics — live proof "just fine-tune on the docs" fails, dangerously here.
- Fine-tuning lifts **format, citations, abstention**, not raw recall.
- **FT + RAG** wins the holistic rubric.
- If well-prompted **base + RAG** matches FT + RAG, **say so** — "fine-tuning wasn't worth the cost for this task" is a stronger result than a manufactured win.

**Acceptance criteria:** `reports/eval_report.md` has all four rows + interpretation; the 2×2 table is in the README.

**Claude Code prompt:**
> Run the harness on all four configs, regenerate `reports/eval_report.md` with a row each and a 6–10 sentence interpretation (incl. whether fine-tuning was worth it), and put the 2×2 table into `README.md`.

---

## 11. Guardrails & serving

- **Enforce abstention & grounding at inference:** a post-check that blocks any answer with an ungrounded dosage/chemical (via §5b NER) and swaps in the abstention response.
- **Disclaimer** on every answer.
- **Prompt-injection isolation:** treat corpus/KCC text as untrusted content, not instructions.
- **Serve:** vLLM or HF behind a **FastAPI** `/ask`; a **Streamlit/Gradio** UI showing answer + citations + disclaimer. Secrets via env vars.

**Acceptance criteria:** a deployed URL answering with citations or correctly abstaining; the ungrounded-dosage blocker demonstrably fires on a crafted prompt.

**Claude Code prompt:**
> Add `src/serve/`: an inference wrapper with an ungrounded-dosage blocker (using NER) + mandatory disclaimer, a FastAPI `/ask` returning answer+citations+disclaimer, a Streamlit UI, prompt-injection isolation for retrieved text, a Dockerfile, and deploy notes with env-var secrets.

---

## 12. Limitations & "how I'd productionize" (put in README)

- **Corpus coverage** bounds everything; abstention is honest, not a bug.
- **No field feedback loop** — production would route uncertain cases to extension officers and learn from outcomes.
- **Static knowledge** — advisories are seasonal; needs a refresh pipeline.
- **Language** — real farmers need Hindi/regional (see stretch).
- **Safety** — anything actionable stays advisory + human-verified, never autonomous.

Stating these is a senior signal.

---

## 13. Stretch goals (each a differentiator)

- **Hybrid-retrieval & reranker ablation** — extra eval rows quantifying the gain.
- **Multilingual** — Hindi/regional queries (KCC has regional data) — huge real-world relevance.
- **Image input** — a PlantVillage/PlantDoc leaf-disease classifier → RAG-grounded treatment. Fuses computer vision + NLP + your domain: a combination almost no other candidate can assemble.
- **Structured dosage-table retrieval** as a typed tool.

---

## 14. Repo structure & Claude Code workflow

```
ag-advisory/
├─ CLAUDE.md
├─ README.md            # thesis + 2×2 table + limitations
├─ requirements.txt
├─ data/{raw,corpus,finetune,nlp}/
├─ models/adapters/
├─ eval/datasets/gold.jsonl
├─ reports/{eval_report,intent_eval,ner_eval,topics}.md
├─ notebooks/finetune.ipynb
├─ src/{data,ingest,rag,nlp,eval,finetune,serve}/
└─ tests/
```

**`CLAUDE.md` starter:**
```md
# Project: Agricultural Advisory Assistant — RAG vs Fine-Tuning
Thesis: RAG supplies knowledge; fine-tuning supplies behavior. Prove it with a 2×2.
Build order: data → corpus → RAG v1 → NLP layer (intent + NER/IE + topics) →
gold set → eval harness → SFT data → fine-tune → run 2×2 → guardrails/serve → writeup.

## Rules
- Safety first: no ungrounded dosages/chemicals; abstain when context is thin; cite + disclaim.
- Keep train (data/finetune) and eval (eval/datasets/gold.jsonl) strictly separate.
- Fine-tuning teaches format/abstention, NOT facts.
- RAGAS evaluates the RAG arm only; use custom/seqeval/sklearn for the rest.
- Every phase ships tests and updates its report in reports/.
- Secrets via env vars only.

## Stack
transformers+peft+bitsandbytes+trl (QLoRA, 3B), FAISS/Chroma, BGE/E5 embeddings,
scikit-learn + spaCy + BERTopic (intent/NER/topics), RAGAS + seqeval + custom eval,
Anthropic API as LLM-judge, FastAPI + Streamlit, W&B.
```

**Phase-gate loop:** paste the phase prompt → Claude Code implements + tests → you run the acceptance check → commit → next. **Ship a thin end-to-end slice early** (small corpus → RAG → base eval) before fine-tuning, so you always have something working.

---

## 15. Dependencies (`requirements.txt` starting point)

```
# data + RAG
requests  pandas  duckdb  pypdf  unstructured  camelot-py
langchain  llama-index  faiss-cpu  chromadb  sentence-transformers  rank-bm25
# NLP modeling
scikit-learn  spacy  bertopic  seqeval  datasets
# fine-tuning
transformers  peft  bitsandbytes  trl  accelerate
# evaluation
ragas  deepeval  anthropic
# serving + tracking
fastapi  uvicorn  streamlit  wandb
```

Pin versions once it runs. `vllm` optional for faster serving; `torch` per your CUDA.

---

## Build-order recap

`data → corpus → RAG v1 → NLP layer (intent · NER/IE · topics) → gold eval set → eval harness → SFT data → fine-tune → run the 2×2 → guardrails + serve → writeup.`

Internalize one sentence: **the evaluation harness is the project.** A rigorously measured, honestly-reported result — even "fine-tuning wasn't worth it here" — beats a flashier project with no evaluation, every time.
