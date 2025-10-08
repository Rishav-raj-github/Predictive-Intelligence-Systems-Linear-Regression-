# Predictive Intelligence Systems: Linear Regression (2025)

[![CI](https://github.com/Rishav-raj-github/Predictive-Intelligence-Systems-Linear-Regression-/actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](#)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](../../pulls)

A production-oriented framework for building linear regression systems with feature engineering, explainability, tests, and CI/CD.

## Contents
- Overview
- Features
- Tech Stack
- Project Layout
- Installation
- Quick Start
- Examples
- Contributing
- License

## Overview
This repository demonstrates enterprise-grade linear regression with scalability, interpretability, and maintainability in mind. It includes modular components, testing, linting, and automation to help you ship reliable ML quickly.

## Key Features
- Feature engineering pipelines (polynomial, interactions, scaling, imputation)
- Regularized models (Ridge, Lasso, ElasticNet)
- Explainability with SHAP/LIME
- Realtime inference via FastAPI
- CI with linting (ruff), formatting (black), type checks (mypy), tests (pytest)
- Dockerized development and deployment

## Tech Stack
- Python, NumPy, Pandas, scikit-learn, matplotlib/seaborn
- FastAPI, Uvicorn
- pytest, coverage, hypothesis (optional)
- black, ruff, mypy, pre-commit
- Docker, GitHub Actions

## Project Layout
```
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

## Installation
- Python 3.9+
- Create and activate a virtualenv
- Install:
```
pip install -r requirements.txt
pip install -r requirements-dev.txt  # for contributors
```

## Quick Start
```
python examples/quickstart_basic.py
```

## Examples
- examples/quickstart_basic.py: load CSV, train/evaluate LinearRegression
- examples/advanced_pipeline.py: polynomial features + scaling + Ridge pipeline

## Contributing
- Follow PEP 8; run black, ruff, mypy; add tests and docstrings
- Use pre-commit hooks: `pre-commit install`
- Open PRs with clear description and passing CI

## License
MIT License. See LICENSE.
