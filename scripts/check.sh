#!/usr/bin/env bash

# Fail fast on errors, undefined variables and failed pipes.
set -euo pipefail

# Przechodzimy do katalogu głównego repozytorium niezależnie od miejsca uruchomienia.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Uruchom linting dla kodu źródłowego i testów.
echo "[check] ruff"
ruff check src/ tests/

# Uruchom statyczną analizę typów dla modułów Pythona.
echo "[check] mypy"
mypy src/

# Uruchom testy jednostkowe wraz z raportem pokrycia.
echo "[check] pytest + coverage"
pytest --cov=src/image_analysis --cov-report=term-missing
