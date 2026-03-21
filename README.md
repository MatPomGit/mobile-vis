# Szablon Projektu – Analiza Obrazu w Pythonie

> **Język opisów:** Polski  
> **Język kodu:** Angielski

Szablon repozytorium dla projektów dotyczących **analizy obrazu** zrealizowanych w Pythonie.
Zawiera gotową strukturę katalogów, konfigurację narzędzi jakości kodu oraz instrukcje dla agentów AI
(GitHub Copilot, Codex, Claude), które gwarantują spójność i najwyższą jakość kodu.

---

## 📋 Spis treści

1. [Opis projektu](#opis-projektu)
2. [Struktura repozytorium](#struktura-repozytorium)
3. [Wymagania systemowe](#wymagania-systemowe)
4. [Instalacja](#instalacja)
5. [Użycie](#użycie)
6. [Testowanie](#testowanie)
7. [Styl kodu i narzędzia jakości](#styl-kodu-i-narzędzia-jakości)
8. [Współpraca z agentami AI](#współpraca-z-agentami-ai)
9. [Wkład w projekt](#wkład-w-projekt)
10. [Licencja](#licencja)

---

## Opis projektu

Niniejszy szablon przeznaczony jest do szybkiego uruchomienia nowych projektów analizy obrazu
w Pythonie. Pokrywa typowy przepływ pracy obejmujący:

- wstępne przetwarzanie obrazów (preprocessing),
- wykrywanie obiektów (detection),
- klasyfikację (classification),
- narzędzia pomocnicze (utils).

Projekty oparte na tym szablonie używają polskojęzycznych opisów (README, dokumentacja, komentarze
do zadań), natomiast cały kod źródłowy, nazwy funkcji, zmiennych i docstringi pisane są **po
angielsku**, co ułatwia współpracę z narzędziami AI i społecznością open-source.

---

## Struktura repozytorium

```
repo-template/
├── .github/
│   ├── copilot-instructions.md   # Instrukcje dla GitHub Copilot
│   ├── PULL_REQUEST_TEMPLATE.md  # Szablon opisu Pull Request
│   └── ISSUE_TEMPLATE/
│       ├── bug_report.md         # Szablon zgłoszenia błędu
│       └── feature_request.md    # Szablon prośby o funkcjonalność
├── src/
│   └── image_analysis/
│       ├── __init__.py           # Eksporty publicznego API modułu
│       ├── preprocessing.py      # Wstępne przetwarzanie obrazów
│       ├── detection.py          # Wykrywanie obiektów
│       ├── classification.py     # Klasyfikacja obrazów
│       └── utils.py              # Funkcje pomocnicze
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py
│   ├── test_detection.py
│   └── test_classification.py
├── notebooks/
│   └── example_analysis.ipynb   # Przykładowy notatnik Jupyter
├── data/
│   ├── raw/                      # Surowe dane wejściowe (ignorowane przez git)
│   └── processed/                # Przetworzone dane (ignorowane przez git)
├── models/                       # Wytrenowane modele (ignorowane przez git)
├── docs/
│   └── index.md                  # Dokumentacja projektu
├── AGENTS.md                     # Instrukcje dla agentów Codex / OpenAI
├── CLAUDE.md                     # Instrukcje dla agenta Claude
├── pyproject.toml                # Konfiguracja projektu i narzędzi
├── requirements.txt              # Zależności projektu
└── .gitignore
```

---

## Wymagania systemowe

| Wymaganie     | Wersja minimalna |
|---------------|-----------------|
| Python        | 3.11             |
| pip           | 23.x             |
| Git           | 2.x              |

Zalecane środowisko: **virtualenv** lub **conda**.

---

## Instalacja

```bash
# Sklonuj repozytorium (lub użyj jako szablonu w GitHub)
git clone https://github.com/<uzytkownik>/<projekt>.git
cd <projekt>

# Utwórz i aktywuj wirtualne środowisko
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate    # Windows

# Zainstaluj zależności
pip install -e ".[dev]"
```

---

## Użycie

```python
from image_analysis.preprocessing import load_image, resize_image
from image_analysis.detection import detect_objects
from image_analysis.classification import classify_image

# Wczytaj i przygotuj obraz
image = load_image("data/raw/sample.jpg")
resized = resize_image(image, width=640, height=480)

# Wykryj obiekty
detections = detect_objects(resized)

# Sklasyfikuj obraz
label, confidence = classify_image(resized)
print(f"Etykieta: {label}, pewność: {confidence:.2%}")
```

Więcej przykładów: [`notebooks/example_analysis.ipynb`](notebooks/example_analysis.ipynb).

---

## Testowanie

```bash
# Uruchom wszystkie testy
pytest

# Testy z pokryciem kodu
pytest --cov=src/image_analysis --cov-report=term-missing

# Testy jednego modułu
pytest tests/test_preprocessing.py -v
```

---

## Styl kodu i narzędzia jakości

Projekt używa następujących narzędzi (skonfigurowanych w `pyproject.toml`):

| Narzędzie  | Zastosowanie                     |
|------------|----------------------------------|
| `ruff`     | Lintowanie i formatowanie kodu   |
| `mypy`     | Statyczna analiza typów          |
| `pytest`   | Testy jednostkowe                |
| `pre-commit` | Weryfikacja przed commitem     |

```bash
# Lintowanie
ruff check src/ tests/

# Formatowanie
ruff format src/ tests/

# Sprawdzanie typów
mypy src/

# Instalacja hooków pre-commit
pre-commit install
pre-commit run --all-files
```

---

## Współpraca z agentami AI

Repozytorium zawiera dedykowane instrukcje dla agentów AI:

| Plik                                  | Agent              |
|---------------------------------------|--------------------|
| `.github/copilot-instructions.md`     | GitHub Copilot     |
| `AGENTS.md`                           | Codex / OpenAI     |
| `CLAUDE.md`                           | Claude (Anthropic) |

Instrukcje obejmują:
- zasady stylu kodu (PEP 8, type hints, docstringi Google-style),
- wymagania testowe (pytest, pokrycie > 80 %),
- wzorce specyficzne dla analizy obrazu,
- reguły bezpieczeństwa i wydajności.

---

## Wkład w projekt

1. Utwórz branch z opisową nazwą: `feature/<nazwa>` lub `fix/<nazwa>`.
2. Wprowadź zmiany przestrzegając zasad zawartych w `AGENTS.md` / `CLAUDE.md`.
3. Uruchom testy i linter przed wystawieniem PR.
4. Wypełnij szablon Pull Request.

---

## Licencja

Projekt objęty licencją **Apache 2.0** – szczegóły w pliku [LICENSE](LICENSE).
