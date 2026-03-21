# Instrukcje dla Agentów AI (Codex / OpenAI)

## Kontekst projektu

To repozytorium jest **szablonem** dla projektów analizy obrazu w Pythonie.  
Obowiązuje zasada: **opisy i dokumentacja w języku polskim**, **kod źródłowy w języku angielskim**.

---

## Twoja rola

Jesteś asystentem programistycznym specjalizującym się w analizie obrazu (computer vision) z
użyciem Pythona. Twoim zadaniem jest:

1. Tworzenie kodu najwyższej jakości, zgodnego z wytycznymi poniżej.
2. Proponowanie ulepszeń istniejącego kodu.
3. Pisanie lub uzupełnianie testów jednostkowych.
4. Wyjaśnianie decyzji architektonicznych po polsku, kiedy jest to wymagane.

---

## Zasady pisania kodu

### Styl i formatowanie

- Przestrzegaj **PEP 8**. Konfiguracja lintowania w `pyproject.toml` (narzędzie: `ruff`).
- Maksymalna długość linii: **99 znaków**.
- Stosuj **type hints** we wszystkich sygnaturach funkcji publicznych.
- Docstringi w formacie **Google-style**, zawsze po angielsku.
- Importy w kolejności: `stdlib` → `third-party` → `local` (zgodnie z `isort`).

### Struktura kodu

```
src/image_analysis/
├── preprocessing.py  – wczytywanie, zmiana rozmiaru, normalizacja, augmentacja
├── detection.py      – detekcja obiektów, bounding boxes, NMS
├── classification.py – klasyfikacja obrazów, predykcja, metryki
└── utils.py          – helpers: walidacja, logowanie, ścieżki, wizualizacja
```

Nowe moduły umieszczaj w `src/image_analysis/` i eksportuj przez `__init__.py`.

### Wymagania dot. typów i danych

- Obrazy reprezentuj jako `np.ndarray` z wymiarami opisanymi w docstringu: `(H, W, C)` BGR lub
  `(H, W)` grayscale.
- Dokumentuj oczekiwany `dtype` i zakres: `uint8 [0, 255]` lub `float32 [0.0, 1.0]`.
- Współrzędne bounding box – format `(x1, y1, x2, y2)` lub `(x, y, w, h)` – zawsze opisuj
  w docstringu.

### Obsługa błędów

```python
# Dobrze
def load_image(path: str) -> np.ndarray:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image file not found: {path}")
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Failed to decode image: {path}")
    return image

# Źle – zbyt ogólne wyjątki
def load_image(path: str) -> np.ndarray:
    try:
        return cv2.imread(path)
    except Exception:
        return None  # type: ignore
```

- Nigdy nie połykaj wyjątków bez logowania.
- Używaj modułu `logging` zamiast `print`.

---

## Testowanie

- Testy w `tests/` odpowiadają modułom w `src/image_analysis/`.
- Uruchom testy: `pytest --cov=src/image_analysis --cov-report=term-missing`.
- Wymagane pokrycie: **≥ 80 %**.
- Używaj syntetycznych obrazów w testach:

```python
import numpy as np

def make_test_image(h: int = 100, w: int = 100, c: int = 3) -> np.ndarray:
    """Create a synthetic BGR test image filled with random pixels."""
    rng = np.random.default_rng(seed=42)
    return rng.integers(0, 255, (h, w, c), dtype=np.uint8)
```

- Parametryzuj przypadki brzegowe: obrazy 1-pikselowe, grayscale, bardzo duże.
- Mockuj operacje I/O i wywołania modeli za pomocą `pytest-mock` lub `unittest.mock`.

---

## Wydajność

- Unikaj pętli Pythona na poziomie pikseli – używaj operacji wektoryzowanych NumPy lub funkcji OpenCV.
- Przetwarzaj wsadowo (batch), gdy dane wejściowe to wiele obrazów.
- Profile kod przed optymalizacją: `cProfile`, `line_profiler`.
- Zwalniaj pamięć GPU po inferencji (`.detach()`, `torch.cuda.empty_cache()`).

---

## Bezpieczeństwo

- Zmienne środowiskowe dla kluczy API – nigdy w kodzie.
- Waliduj ścieżki wejściowe, by uniknąć path traversal.
- Weryfikuj sumy kontrolne pobieranych wag modeli.

---

## Praca z repozytorium

### Branche

- `main` – stabilna wersja, chroniona reguła branch protection.
- `feature/<opis>` – nowe funkcjonalności.
- `fix/<opis>` – poprawki błędów.
- `refactor/<opis>` – refaktoryzacja bez zmiany zachowania.

### Commit messages (Conventional Commits, angielski)

```
feat(preprocessing): add histogram equalization
fix(detection): correct NMS threshold application
test(utils): add tests for path validation helper
docs(readme): clarify installation steps
```

### Pull Requests

Wypełniaj szablon z `.github/PULL_REQUEST_TEMPLATE.md`. Każdy PR musi:
- Przechodzić `ruff check`, `mypy`, `pytest` bez błędów.
- Mieć opis zmian po polsku.
- Zawierać testy dla nowego kodu.

---

## Czego bezwzględnie unikać

| Zakaz | Powód |
|-------|-------|
| `except Exception: pass` | Ukrywa błędy |
| `print()` w kodzie produkcyjnym | Używaj `logging` |
| Brak type hints | Utrudnia statyczną analizę i AI autocomplete |
| Magiczne liczby (np. `0.5`, `224`) | Używaj nazwanych stałych |
| Mutowanie globalnego stanu | Powoduje trudne do wykrycia błędy |
| Commitowanie plików `data/`, `models/` | Są w `.gitignore` |
| Klucze API w kodzie | Ryzyko wycieku sekretów |
