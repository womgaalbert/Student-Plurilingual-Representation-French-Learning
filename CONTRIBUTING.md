# Contributing

This is an open research project. Contributions are welcome, especially around:
- French NLP improvements (CamemBERT fine-tuning)
- Cameroonian language datasets
- Educational AI / multilingualism research

## Branching strategy
- `main`      — stable, tested code only
- `dev`       — integration branch
- `feature/*` — new features
- `exp/*`     — ML experiments (e.g. `exp/h1-lightgbm`)

## Before submitting a PR
1. Run `pytest tests/ -v` — all tests must pass
2. Hyperparameters must be in `params.yaml`, not hardcoded
3. Every training run must log to MLflow
4. Raw data must never be committed (see .gitignore)
