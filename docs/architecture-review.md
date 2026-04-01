# Przegląd struktury programu i rekomendacje usprawnień

Data przeglądu: 2026-03-31

## 1. Co działa dobrze

1. **Wyraźny podział domenowy modułów** w `src/image_analysis/` (preprocessing, detection,
   classification, utilities + moduły markerów i MediaPipe), co ułatwia nawigację i rozwój.
2. **Spójne typowanie i docstringi** w głównych modułach Pythona.
3. **Konfiguracja jakości kodu** (`ruff`, `mypy`, `pytest`, coverage) jest przygotowana i spójna.
4. **Rozdzielenie warstwy Python i Android** w repozytorium jest czytelne i praktyczne.

## 2. Najważniejsze obszary do poprawy (priorytety)

## P0 (krytyczne)

### P0.1. Stabilność importów pakietu `image_analysis`

**Problem:** `src/image_analysis/__init__.py` importuje wiele modułów naraz. Jeśli jedna
zależność natywna (np. OpenCV) nie jest dostępna, import całego pakietu może zakończyć się
błędem już na etapie collection testów.

**Skutek:** awaria importów nawet dla tych testów/funkcji, które nie potrzebują wszystkich
modułów.

**Rekomendacja:**
- ograniczyć importy w `__init__.py` do najstabilniejszych elementów, albo
- zastosować podejście „lazy import” (np. przez `__getattr__` dla API publicznego), albo
- podzielić API na podpakiety (`image_analysis.core`, `image_analysis.markers`,
  `image_analysis.mediapipe`).

### P0.2. Niedopasowanie środowiska testowego do zależności OpenCV

**Problem:** testy nie przechodzą w środowisku bez biblioteki systemowej `libGL.so.1`.

**Rekomendacja:**
- dla CI i środowisk headless używać wyłącznie `opencv-python-headless` (już obecne w
  zależnościach),
- doprecyzować w dokumentacji wymagania systemowe dla lokalnego OpenCV,
- dodać prosty smoke-check importów do CI (`python -c "import cv2; print(cv2.__version__)"`).

## P1 (wysoki)

### P1.1. Ujednolicenie kontraktów danych obrazowych

W części modułów kontrakty wejścia/wyjścia są dobre, ale można je scentralizować:

- dodać jeden moduł typów (`types.py`) z aliasami:
  - `ImageU8 = NDArray[np.uint8]`
  - `ImageF32 = NDArray[np.float32]`
  - `BgrImageU8 = NDArray[np.uint8]  # shape (H, W, 3)`
- przenieść wspólne walidatory (`validate_bgr_image`, `validate_bbox`) do `utils.py`.

### P1.2. Dopracowanie architektury „stubów” produkcyjnych

`detection.py` i `classification.py` zawierają placeholdery. To jest poprawne dla szablonu,
ale warto dodać formalny wzorzec rozszerzeń:

- interfejs/protokół modelu (np. `Protocol`),
- rejestr backendów (`opencv`, `onnxruntime`, `torch`),
- jednolity format wyników i metryk.

### P1.3. Lepsza separacja katalogu `android/`

Android jest pełnoprawnym podprojektem. Warto doprecyzować workflow:

- osobna sekcja CI dla Android,
- w `README` i `docs` wyraźnie opisać, kiedy uruchamiamy testy Python, a kiedy build Android,
- rozważyć `docs/architecture/` z diagramem granic modułów Python ↔ Android.

## P2 (średni)

### P2.1. Wzmocnienie spójności testów

- dodać testy kontraktowe API publicznego (`tests/test_public_api.py`) sprawdzające, że
  eksporty z `__init__.py` są stabilne,
- dodać testy skrajnych przypadków dla bbox i NMS,
- dodać testy wydajnościowe jako `pytest -m performance` (opcjonalnie, poza CI domyślnym).

### P2.2. Planowany podział dokumentacji

Dla rosnącego repo przyda się:

- `docs/architecture.md` (opis modułów i zależności),
- `docs/testing.md` (matrix testów i środowisk),
- `docs/deployment.md` (lokalne uruchamianie + CI).

## 3. Proponowana kolejność wdrożeń (2 sprinty)

### Sprint 1
1. Ograniczyć ryzyko importów w `__init__.py`.
2. Ustabilizować środowisko testowe OpenCV (CI + dokumentacja).
3. Dodać test publicznego API i smoke-check importów.

### Sprint 2
1. Wprowadzić moduł typów (`types.py`) i ujednolicić walidatory.
2. Dodać prosty interfejs backendu klasyfikacji/detekcji.
3. Rozdzielić i doprecyzować dokumentację architektury Python/Android.

## 4. Quick wins (do wdrożenia od razu)

1. Dodać `tests/test_public_api.py`.
2. Dodać sekcję „Known environment issues” z `libGL.so.1` w dokumentacji.
3. Dodać `Makefile` lub skrypt `scripts/check.sh` z poleceniami:
   - `ruff check .`
   - `mypy src`
   - `pytest --cov=src/image_analysis --cov-report=term-missing`

## 5. Podsumowanie

Projekt ma bardzo dobry fundament jak na szablon CV + Android. Największa poprawa jakości w
krótkim czasie będzie wynikała z:

- utwardzenia importów i środowiska testowego,
- wyraźniejszego kontraktu API,
- doprecyzowania granic i workflow między Python i Android.
