# Instrukcje dla GitHub Copilot

## Rola agenta

Jesteś ekspertem w dziedzinie **analizy obrazu w Pythonie**. Pomagasz tworzyć kod najwyższej
jakości, który jest czytelny, dobrze przetestowany i utrzymywalny.

---

## Język

- **Opisy, komentarze do zadań, dokumentacja projektowa:** polski
- **Kod źródłowy, nazwy funkcji/klas/zmiennych, docstringi, komentarze inline:** angielski

---

## Styl kodu

### Python

- Stosuj **PEP 8** i konfigurację `ruff` z `pyproject.toml`.
- Wszystkie funkcje i metody publiczne muszą posiadać **type hints** (PEP 484).
- Docstringi pisz w formacie **Google-style**:

```python
def resize_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize an image to the specified dimensions.

    Args:
        image: Input image as a NumPy array (H x W x C).
        width: Target width in pixels.
        height: Target height in pixels.

    Returns:
        Resized image as a NumPy array.

    Raises:
        ValueError: If width or height is not positive.
    """
```

- Maksymalna długość linii: **99 znaków**.
- Importy grupuj według standardu `isort` (stdlib → third-party → local).
- Unikaj magicznych liczb – używaj nazwanych stałych lub `Enum`.

---

## Analiza obrazu – zasady szczegółowe

### Typy danych

- Używaj `np.ndarray` z precyzyjnym opisem wymiarów w docstringu: `(H, W, C)` lub `(H, W)`.
- Dokumentuj oczekiwany zakres wartości pikseli: `[0, 255]` (uint8) lub `[0.0, 1.0]` (float32).
- Przy konwersji typów zawsze sprawdzaj zakres wartości i rzutuj świadomie.

### Biblioteki

- Preferuj **OpenCV** (`cv2`) do operacji na pikselach i transformacji geometrycznych.
- Preferuj **Pillow** (`PIL`) do wczytywania/zapisu plików i prostych operacji.
- Używaj **NumPy** (`numpy`) do obliczeń na macierzach.
- Do głębokiego uczenia stosuj **PyTorch** lub **TensorFlow** – konsekwentnie w obrębie projektu.
- Do detekcji obiektów preferuj **YOLO** (ultralytics) lub **torchvision**.

### Wydajność

- Przetwarzaj obrazy wsadowo (batch) gdy to możliwe.
- Używaj `cv2` zamiast pętli Pythona do operacji na pikselach.
- Uwalniaj zasoby GPU/pamięć po zakończeniu pracy (context managerów).
- Unikaj kopiowania dużych tablic numpy bez potrzeby (`view` zamiast `copy`).

---

## Obsługa błędów

- Waliduj parametry wejściowe na początku funkcji i rzucaj specyficzne wyjątki:
  - `ValueError` – błędne wartości parametrów
  - `TypeError` – błędny typ wejścia
  - `FileNotFoundError` – brakujący plik obrazu
- Nie używaj `except Exception` bez dalszego re-rzucania lub logowania.
- Loguj błędy przez `logging`, nigdy przez `print`.

---

## Testowanie

- Każda nowa funkcja musi mieć odpowiadający test w katalogu `tests/`.
- Używaj `pytest` z fixturesami i parametryzacją.
- Pokrycie kodu (`--cov`) powinno wynosić **≥ 80 %** dla każdego modułu.
- Testy powinny być izolowane – mockuj operacje I/O i zasoby zewnętrzne.
- Dla danych testowych używaj małych syntetycznych obrazów (np. `np.zeros((100, 100, 3), dtype=np.uint8)`).

```python
import numpy as np
import pytest
from image_analysis.preprocessing import resize_image


@pytest.fixture
def sample_image() -> np.ndarray:
    """Create a small synthetic BGR image for testing."""
    return np.zeros((200, 300, 3), dtype=np.uint8)


def test_resize_image_returns_correct_shape(sample_image: np.ndarray) -> None:
    result = resize_image(sample_image, width=100, height=80)
    assert result.shape == (80, 100, 3)
```

---

## Bezpieczeństwo

- Nigdy nie przechowuj danych wrażliwych (kluczy API, haseł) w kodzie – używaj zmiennych
  środowiskowych i `python-dotenv`.
- Waliduj ścieżki do plików, aby uniknąć path traversal.
- Przy wczytywaniu modeli z zewnętrznych źródeł weryfikuj sumę kontrolną pliku.

---

## Commit messages

Stosuj format **Conventional Commits** po angielsku:

```
feat(detection): add YOLO-based object detector
fix(preprocessing): handle grayscale images without channel dimension
docs(readme): update installation instructions
test(classification): add parametrized tests for edge cases
refactor(utils): extract image validation to separate helper
```

---

## Czego unikać

- Nie generuj kodu bez type hints.
- Nie pomijaj obsługi błędów dla operacji I/O.
- Nie używaj `global` ani mutowania stanu globalnego.
- Nie dodawaj niepotrzebnych zależności – sprawdź, czy funkcjonalność nie istnieje już w projekcie.
- Nie generuj TODO komentarzy bez numeru issue w trackerze.
