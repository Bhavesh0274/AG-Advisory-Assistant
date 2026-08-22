# Build Progress

Tracking phases from `ag-advisory-build-guide-FINAL.md`. Crops in scope: rice, wheat, tomato.

| Phase | Status | Notes |
|---|---|---|
| 0. Repo scaffold | done | structure, CLAUDE.md, README, requirements.txt, venv (Python 3.12), git init |
| 2. Data fetch (KCC + sources.csv) | in progress | |
| 3. Corpus ingestion | not started | |
| 4. RAG pipeline v1 | not started | |
| 5a. Intent classification | not started | |
| 5b. NER / IE | not started | |
| 5c. Topic modeling | not started | |
| 6. Gold eval set | not started | |
| 7. Eval harness | not started | |
| 8. SFT data | not started | |
| 9. Fine-tune (QLoRA) | not started | needs GPU/Colab |
| 10. Run 2×2 + report | not started | |
| 11. Guardrails & serving | not started | |

## Environment notes
- System Python is 3.14 (too new for this stack). Project venv (`.venv/`) uses Python 3.12.
- Kaggle API token, Anthropic API key, and W&B account not yet configured — will flag inline when a phase needs them.
