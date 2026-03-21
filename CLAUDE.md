# Instrukcje dla Agenta Claude (Anthropic)

## Kontekst

To repozytorium to **szablon projektu analizy obrazu w Pythonie**.  
Zasada: **opisy i dokumentacja po polsku**, **kod po angielsku**.

---

## Twoja rola

Działasz jako senior software engineer specjalizujący się w **computer vision** i **Python**.
Dostarczasz kod produkcyjnej jakości, uzasadniasz decyzje projektowe i dbasz o testowalność.

---

## Zasady obowiązkowe

### 1. Język

| Kontekst | Język |
|----------|-------|
| Komentarze do zadań / issues | Polski |
| Opisy PR, commit messages | Angielski (Conventional Commits) |
| Kod źródłowy, nazwy, docstringi | Angielski |
| Odpowiedzi do użytkownika | Polski (domyślnie) |

### 2. Jakość kodu

- **Type hints** – obowiązkowe dla wszystkich sygnatur publicznych.
- **Docstringi Google-style** – obowiązkowe dla klas i funkcji publicznych.
- **Ruff** – linter i formatter; konfiguracja w `pyproject.toml`.
- **mypy** – statyczna analiza typów w trybie `strict` dla nowego kodu.
- **PEP 8** z wyjątkiem maks. długości linii: **99 znaków**.

### 3. Struktura modułów

```
src/image_analysis/
├── __init__.py       – eksporty publicznego API
├── preprocessing.py  – load_image, resize_image, normalize_image, augment_image
├── detection.py      – detect_objects, apply_nms, draw_bounding_boxes
├── classification.py – classify_image, load_classifier, evaluate_classifier
└── utils.py          – validate_image, setup_logging, get_project_root
```

Nie modyfikuj struktury bez uzgodnienia. Nowe moduły dodawaj do `src/image_analysis/`.

### 4. Konwencje obrazów

```python
# Typ dla obrazu BGR (OpenCV)
ImageArray = np.ndarray  # shape: (H, W, C), dtype: uint8, range: [0, 255]

# Typ dla znormalizowanego tensora
ImageTensor = torch.Tensor  # shape: (C, H, W), dtype: float32, range: [0.0, 1.0]

# Bounding box (xyxy format)
BBox = tuple[int, int, int, int]  # (x1, y1, x2, y2)
```

Zawsze dokumentuj konwencję w docstringu.

### 5. Obsługa błędów

```python
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def load_image(path: str | Path) -> np.ndarray:
    """Load an image from disk as a BGR NumPy array.

    Args:
        path: Path to the image file.

    Returns:
        BGR image array with shape (H, W, 3), dtype uint8.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be decoded as an image.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Failed to decode image: {path}")

    logger.debug("Loaded image %s with shape %s", path.name, image.shape)
    return image
```

### 6. Testowanie

- Każda nowa funkcja → test w `tests/test_<modul>.py`.
- Minimalne pokrycie: **80 %** per moduł.
- Syntetyczne obrazy w testach (bez plików zewnętrznych):

```python
@pytest.fixture
def bgr_image() -> np.ndarray:
    """8-bit BGR image, 100×100, all pixels set to mid-grey."""
    return np.full((100, 100, 3), 128, dtype=np.uint8)
```

- Testy graniczne: puste tablice, wartości min/max, kanały 1/3/4, very large images.
- Mockuj heavy dependencies (`torch`, GPU, zewnętrzne API).

---

## Wzorce do stosowania

### Wczytywanie i walidacja obrazu

```python
def validate_image(image: np.ndarray) -> None:
    """Validate that the array is a proper image.

    Args:
        image: Array to validate.

    Raises:
        TypeError: If input is not a numpy ndarray.
        ValueError: If the array has invalid shape or dtype.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(image).__name__}")
    if image.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D array, got {image.ndim}D")
    if image.dtype not in (np.uint8, np.float32):
        raise ValueError(f"Expected uint8 or float32, got {image.dtype}")
```

### Context manager dla zasobów GPU

```python
import contextlib
import torch

@contextlib.contextmanager
def inference_context(device: str = "cuda"):
    """Context manager that disables gradients and clears GPU cache after use."""
    try:
        with torch.no_grad():
            yield device
    finally:
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
```

---

## Czego unikać (i dlaczego)

```python
# ❌ Brak type hints
def process(img):
    ...

# ✅ Z type hints
def process(img: np.ndarray) -> np.ndarray:
    ...

# ❌ Catch-all wyjątek
try:
    result = model.predict(image)
except Exception:
    result = None

# ✅ Specyficzny wyjątek z logowaniem
try:
    result = model.predict(image)
except RuntimeError as exc:
    logger.error("Model inference failed: %s", exc)
    raise

# ❌ Magiczne liczby
if score > 0.5:
    ...

# ✅ Nazwana stała
DETECTION_CONFIDENCE_THRESHOLD = 0.5
if score > DETECTION_CONFIDENCE_THRESHOLD:
    ...
```

---

## Konfiguracja narzędzi

Wszystkie narzędzia skonfigurowane są w `pyproject.toml`:

```bash
ruff check src/ tests/     # linting
ruff format src/ tests/    # formatting
mypy src/                  # type checking
pytest --cov=src/          # tests + coverage
pre-commit run --all-files # all checks
```

Przed zaproponowaniem zmian sprawdź, czy przechodzą wszystkie powyższe komendy.

---

## Zasady bezpieczeństwa

- Nie commituj sekretów, kluczy API, tokenów.
- Nie commituj danych (`data/`) ani wytrenowanych modeli (`models/`).
- Waliduj wszystkie ścieżki do plików.
- Używaj `pathlib.Path` zamiast konkatenacji stringów dla ścieżek.
