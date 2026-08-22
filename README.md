# Agricultural Advisory Assistant — RAG vs Fine-Tuning

A crop/agronomy advisory assistant that answers questions with **source-cited** advice
and, when its corpus can't answer, **abstains** instead of inventing a pesticide dose.

Scoped to **rice, wheat, and tomato** (India, TNAU/ICAR sources).

## Thesis

RAG and fine-tuning solve **different** problems: RAG supplies *knowledge*,
fine-tuning supplies *behavior*. The common mistake — "fine-tune the model on our
documents so it learns agronomy" — bakes in confident hallucination. This project
shows that **fine-tuning for form + RAG for facts** beats it, with metrics.

## The headline experiment

|  | No RAG | With RAG |
|---|---|---|
| **Base model** | control | does retrieval help? |
| **Fine-tuned** | did FT "learn facts"? (hallucinates) | the combination |

_Results land here once §10 of the build guide is complete — see `reports/eval_report.md`._

## Status

Build in progress — see `PROGRESS.md` and `CLAUDE.md` for current phase.
Full phased spec: `ag-advisory-build-guide-FINAL.md`.

## Repo layout

```
ag-advisory/
├─ CLAUDE.md
├─ README.md
├─ requirements.txt
├─ data/{raw,corpus,finetune,nlp}/
├─ models/adapters/
├─ eval/datasets/gold.jsonl
├─ reports/{eval_report,intent_eval,ner_eval,topics}.md
├─ notebooks/finetune.ipynb
├─ src/{data,ingest,rag,nlp,eval,finetune,serve}/
└─ tests/
```

## Limitations

See §12 of the build guide — corpus coverage bounds everything, no field feedback
loop yet, knowledge is static/seasonal, English-only for now, and all output stays
advisory + human-verified, never autonomous.
