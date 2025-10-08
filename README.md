# Predictive Intelligence Systems: Linear Regression

[![CI](https://github.com/Rishav-raj-github/Predictive-Intelligence-Systems-Linear-Regression-/actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-90%2B%25-brightgreen)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%20|%203.10%20|%203.11-blue)](#)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](../../pulls)

Production-grade linear regression with modular feature engineering, regularization, explainability, tests, and CI/CD. Built for reliability, readability, and fast iteration in real-world ML systems.

---

## Table of Contents
- 🚀 Overview
- ✨ Features
- 🧰 Tech Stack
- 🗂️ Project Structure
- 📦 Installation
- ⚡ Quick Start
- 🧪 Usage Examples
- 🧭 API (FastAPI)
- 🤝 Contributing
- 🧾 License
- 📬 Contact

---

## 🚀 Overview
This repository demonstrates an enterprise-ready approach to Linear Regression using scikit-learn and modern Python tooling. It includes reproducible pipelines, robust testing, automated quality checks, and optional FastAPI for serving predictions.

Use this template to:
- Prototype quickly with clean, modular code
- Ship with confidence using CI, linting, and testing
- Explain model behavior using SHAP/LIME
- Scale from notebooks to production

## ✨ Key Features
- Feature engineering pipelines: polynomial, interactions, scaling, imputation
- Regularized models: Ridge, Lasso, ElasticNet with sensible defaults
- Explainability: SHAP/LIME ready utilities and plots
- FastAPI service for real-time inference (Docker-friendly)
- CI with ruff, black, mypy, pytest, coverage
- Pre-commit hooks for consistent style
- Typed codebase and clear project layout

## 🧰 Tech Stack
- Python, NumPy, Pandas, scikit-learn, matplotlib/seaborn
- FastAPI, Uvicorn
- pytest, coverage, hypothesis (optional)
- black, ruff, mypy, pre-commit
- Docker, GitHub Actions

## 🗂️ Project Structure
```text
.
├── src/
│   └── pis_lr/
│       ├── __init__.py
│       ├── data/
│       │   └── loaders.py
│       ├── features/
│       │   ├── __init__.py
│       │   └── engineering.py
│       ├── models/
│       │   ├── __init__.py
│       │   └── linear.py
│       ├── pipeline/
│       │   └── train.py
│       └── api/
│           └── predict.py
├── tests/
│   ├── conftest.py
│   ├── test_features.py
│   └── test_models.py
├── examples/
│   ├── quickstart_basic.py
│   └── advanced_pipeline.py
├── .github/workflows/ci.yml
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── .gitignore
├── Dockerfile
└── README.md
```

## 📦 Installation
Requirements: Python 3.9+

Using virtualenv (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
# For contributors (dev tooling)
pip install -r requirements-dev.txt
pre-commit install
```

## ⚡ Quick Start
Train and evaluate a simple Linear Regression pipeline:
```bash
python examples/quickstart_basic.py
```

## 🧪 Usage Examples
- examples/quickstart_basic.py — load CSV, train/evaluate LinearRegression
- examples/advanced_pipeline.py — polynomial features + scaling + Ridge pipeline

Programmatic usage:
```python
from pis_lr.pipeline.train import train_pipeline
from pis_lr.data.loaders import load_csv

X_train, y_train = load_csv("data/train.csv", target="target")
model, metrics = train_pipeline(X_train, y_train, model="ridge", degree=2)
print(metrics)
```

## 🧭 API (FastAPI)
Serve the model via FastAPI:
```bash
uvicorn pis_lr.api.predict:app --host 0.0.0.0 --port 8000
```
Example request:
```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"features": {"x1": 1.2, "x2": 3.4}}'
```

## 🤝 Contributing
We welcome contributions! Please:
- Follow PEP8; run black, ruff, mypy; add tests and docstrings
- Ensure tests pass locally: `pytest -q` and maintain/increase coverage
- Use pre-commit: `pre-commit run --all-files`
- Open PRs with a clear description, context, and checklist

PR Checklist:
- [ ] Linted (black/ruff) and typed (mypy)
- [ ] Tests added/updated and passing
- [ ] Docs/README updated if needed

## 🧾 License
MIT License — see [LICENSE](LICENSE).

## 📬 Contact
For questions/support:
- Author: Rishav Raj
- Email: rishavraj4383@gmail.com
- GitHub Issues: [Open an issue](../../issues)

If you like this project, please ⭐ the repo and share it!
