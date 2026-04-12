# Szablon Projektu – Analiza Obrazu w Pythonie / Android

> **Język opisów:** Polski  
> **Język kodu:** Angielski

Repozytorium łączy bibliotekę **analizy obrazu w Pythonie** z aplikacją **Android MobileCV**.
Zawiera moduły computer vision, testy, benchmarki, dokumentację oraz mobilny klient do pracy na
obrazie z kamery telefonu przy użyciu OpenCV. Projekt zachowuje prostą zasadę: dokumentacja po
polsku, kod źródłowy po angielsku.

---

## 📋 Spis treści

1. [Opis projektu](#opis-projektu)
2. [Struktura repozytorium](#struktura-repozytorium)
3. [Aplikacja Android – MobileCV](#aplikacja-android--mobilecv)
4. [Wymagania systemowe](#wymagania-systemowe)
5. [Instalacja](#instalacja)
6. [Użycie](#użycie)
7. [Testowanie](#testowanie)
8. [Styl kodu i narzędzia jakości](#styl-kodu-i-narzędzia-jakości)
9. [Współpraca z agentami AI](#współpraca-z-agentami-ai)
10. [Wkład w projekt](#wkład-w-projekt)
11. [Roadmap i rozwój](#roadmap-i-rozwój)
12. [Licencja](#licencja)

> 📖 **Pełna dokumentacja** (instalacja krok po kroku, opis każdego modułu, przykłady użycia,
> instrukcja obsługi aplikacji Android, rozwiązywanie problemów) dostępna jest w
> [`docs/index.md`](docs/index.md).

---

## Opis projektu

Projekt obejmuje dwa uzupełniające się obszary:

- bibliotekę `image_analysis` w Pythonie,
- aplikację Android do demonstracji i uruchamiania wybranych funkcji CV na żywo.

Warstwa Python pokrywa typowy przepływ pracy obejmujący:

- wstępne przetwarzanie obrazów (preprocessing),
- wykrywanie obiektów (detection),
- wykrywanie markerów AprilTag,
- klasyfikację (classification),
- narzędzia pomocnicze (utils).

Repozytorium zawiera też rozszerzone moduły do hologramów, iris/holistic, QR, AprilTagów,
benchmarkingu, robot perception, markerów, odometrii, SLAM i YOLO oraz odpowiadające im testy. Artefakty budowania,
lokalne dane i pliki tymczasowe nie powinny być commitowane.

---

## Struktura repozytorium

```text
mobile-vis/
├── .github/                  # Workflow CI, szablony issue/PR, instrukcje Copilota
├── android/                  # Aplikacja Android MobileCV
├── benchmarks/               # Baseline i wyniki benchmarków
├── data/                     # Katalogi na dane wejściowe i przetworzone (ignorowane)
├── docs/                     # Dokumentacja projektu
├── models/                   # Modele lokalne i przykładowe wagi
├── notebooks/                # Notatniki Jupyter
├── scripts/                  # Skrypty smoke/perf i narzędzia pomocnicze
├── src/image_analysis/       # Biblioteka Python z modułami CV
├── tests/                    # Testy jednostkowe i integracyjne modułów Python
├── AGENTS.md                 # Instrukcje dla agentów Codex / OpenAI
├── CLAUDE.md                 # Instrukcje dla agenta Claude
├── pyproject.toml            # Konfiguracja pakietu i narzędzi jakości
└── README.md
```

Najważniejsze moduły w `src/image_analysis/`:

- `preprocessing.py` – wczytywanie, zmiana rozmiaru i normalizacja obrazów,
- `detection.py`, `yolo.py`, `rtmdet.py` – detekcja obiektów i eksport modeli,
- `april_tags.py`, `qr_detection.py`, `cctag.py` – detekcja markerów i kodów,
- `holistic.py`, `hologram.py`, `iris.py` – moduły MediaPipe i efekty wizualne,
- `benchmarking.py`, `robot_perception.py`, `planes.py`, `calibration.py` – pomiary, VO i geometria,
- `utils.py` – walidacja, logowanie i operacje pomocnicze.

---

## Aplikacja Android – MobileCV

Katalog `android/` zawiera pełną aplikację na Androida umożliwiającą:

- **podgląd na żywo** z kamery przedniej lub tylnej telefonu,
- **przełączanie kamer** przyciskiem FAB w dolnym pasku,
- **wybór filtra OpenCV** z poziomego paska chipów (pojedyncze zaznaczenie).

- **uproszczony workflow trybów detekcji i analizy**:
  - Detekcja: `MARKERS`, `POSE`, `YOLO`,
  - Odometria: jedna zakładka (`VISUAL_ODOMETRY`, `FULL_ODOMETRY`, `ODOMETRY_TRAJECTORY`, `ODOMETRY_MAP`),
  - SLAM: osobna zakładka (punkty odometrii + markery),
  - Dynamic ROI (Active Tracking): osobna zakładka.

### Dostępne filtry

| Filtr | Opis |
|-------|------|
| Original | Surowy obraz z kamery |
| Grayscale | Konwersja do skali szarości |
| Canny Edges | Wykrywanie krawędzi algorytmem Canny |
| Gaussian Blur | Rozmycie gaussowskie 15×15 |
| Threshold | Progowanie binarne przy wartości 127 |
| Sobel Edges | Gradient magnitudy (Sobel X + Y) |
| Laplacian | Krawędzie przez operator Laplace'a |
| Dilate | Dylacja morfologiczna (jądro 9×9) |
| Erode | Erozja morfologiczna (jądro 9×9) |

### Wymagania do budowania

| Wymaganie | Wersja |
|-----------|--------|
| Android Studio | Hedgehog (2023.1) lub nowszy |
| JDK | 17+ |
| Android SDK | API 24–34 |
| Gradle | 8.6 (pobierany automatycznie przez wrapper) |

### Budowanie i uruchamianie

```bash
cd android

# 1. Skopiuj i uzupełnij local.properties (ścieżka do Android SDK)
cp local.properties.example local.properties
# Edytuj local.properties i ustaw sdk.dir

# 2. Wygeneruj Gradle Wrapper (wymagane tylko przy pierwszym pobraniu)
#    – wymaga zainstalowanego Gradle lub Android Studio
gradle wrapper --gradle-version 8.6
chmod +x gradlew

# 3. Zbuduj APK (debug)
./gradlew assembleDebug

# 4. Zainstaluj na podłączonym urządzeniu / emulatorze
./gradlew installDebug
```

Alternatywnie: otwórz katalog `android/` bezpośrednio w **Android Studio** –
IDE wygeneruje wrapper automatycznie i zaoferuje przycisk *Run*.

### Architektura aplikacji

```
MainActivity ──► CameraX (ImageAnalysis)
                      │ ImageProxy (RGBA_8888)
                      ▼
               ImageProcessor.processFrame()
                      │ OpenCV Mat → filter → Mat
                      ▼
               Bitmap → ImageView (full-screen)
```

- **CameraX** dostarcza klatki w formacie RGBA_8888.
- **ImageProcessor** konwertuje je do macierzy BGRA (`Utils.bitmapToMat`),
  stosuje wybrany filtr OpenCV i zwraca przetworzone `Bitmap`.
- **MainActivity** wyświetla wynik w `ImageView` wypełniającym cały ekran.

### Automatyczne wersjonowanie na głównej gałęzi

Dla commitów pushowanych do gałęzi `main` działa workflow GitHub Actions, który:
1. uruchamia `./bump_version.sh patch`,
2. aktualizuje `android/build.gradle.kts` (`app_version_name` i `app_version_code`),
3. wykonuje commit techniczny z podbitym numerem wersji.

---

## Wymagania systemowe

### Python

| Wymaganie     | Wersja minimalna |
|---------------|-----------------|
| Python        | 3.11             |
| pip           | 23.x             |
| Git           | 2.x              |

Zalecane środowisko: **virtualenv** lub **conda**.

### Android

| Wymaganie        | Wersja minimalna |
|------------------|-----------------|
| Android Studio   | Hedgehog 2023.1  |
| JDK              | 17               |
| Android SDK      | API 24 (Android 7.0) |
| Urządzenie/emulator | API 24+       |

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
from image_analysis.april_tags import detect_april_tags
from image_analysis.preprocessing import load_image, resize_image
from image_analysis.detection import detect_objects
from image_analysis.classification import classify_image

# Wczytaj i przygotuj obraz
image = load_image("data/raw/sample.jpg")
resized = resize_image(image, width=640, height=480)

# Wykryj obiekty
detections = detect_objects(resized)

# Wykryj markery AprilTag
april_tags = detect_april_tags(resized)

# Sklasyfikuj obraz
label, confidence = classify_image(resized)
print(f"Etykieta: {label}, pewność: {confidence:.2%}, april tags: {len(april_tags)}")
```

Więcej przykładów: [`notebooks/example_analysis.ipynb`](notebooks/example_analysis.ipynb).

### Wersja aplikacji w CLI

Aktualną wersję aplikacji MobileCV (Android `versionName` + `versionCode`) oraz wersję pakietu
Python można wypisać z linii poleceń:

```bash
mobilecv-version version
```

Wynik zawiera dwie linie:
- `MobileCV Android version: ...`
- `image-analysis package version: ...`

---

## Import behavior

Pakiet `image_analysis` został zaprojektowany tak, aby `import image_analysis` działał także
w środowiskach bez ciężkich zależności opcjonalnych.

### Co importuje się eager (od razu)

Przy samym imporcie pakietu ładowane są tylko stabilne i lekkie elementy:

- `__version__`,
- podstawowe utility z `image_analysis.utils`
  (`get_project_root`, `setup_logging`, `validate_image`, `safe_makedirs`, `list_images`).

### Co importuje się lazy (na żądanie)

Cięższe moduły i ich symbole są ładowane dopiero przy pierwszym dostępie przez API pakietu
(`__getattr__`). Dotyczy to w szczególności:

- `image_analysis.yolo` (zależność opcjonalna: `ultralytics`),
- `image_analysis.rtmdet` (zależność opcjonalna: `mmdet`),
- `image_analysis.holistic` i `image_analysis.iris` (MediaPipe i powiązane komponenty).

Dzięki temu:

- prosty `import image_analysis` nie wymaga pełnego stosu CV,
- koszty startu procesu są mniejsze,
- moduły opcjonalne instalujesz tylko wtedy, gdy są potrzebne.

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

## Szybka diagnostyka

Poniższy zestaw komend pozwala lokalnie odtworzyć kluczowe kroki CI dla części Python i Android.

### Python (lint / type / test / smoke / metryki FPS)

```bash
# 1) Instalacja zależności deweloperskich
python -m pip install --upgrade pip
pip install -e ".[dev]"

# 2) Lint i typy
ruff check src/ tests/
mypy src/

# 3) Testy jednostkowe z pokryciem
pytest --cov=src/image_analysis --cov-report=term-missing

# 4) Smoke import modułów
python -c 'import importlib; [importlib.import_module(m) for m in [
"image_analysis",
"image_analysis.preprocessing",
"image_analysis.detection",
"image_analysis.classification",
"image_analysis.april_tags",
"image_analysis.qr_detection",
"image_analysis.utils",
]]; print("Smoke import OK")'

# 5) Metryki wydajności smoke (czas klatki / FPS)
python scripts/smoke_perf.py --iterations 150 --warmup 30

# 6) Benchmark VO + planes z porównaniem do baseline
PYTHONPATH=src python scripts/benchmark_vo_planes.py --strict
```

### Android (build debug / lint / testy instrumentacyjne)

```bash
cd android
chmod +x gradlew

# 1) Lint Android
./gradlew lintDebug --stacktrace

# 2) Build debug APK
./gradlew assembleDebug --stacktrace

# 3) (Opcjonalnie) testy instrumentacyjne na emulatorze/urządzeniu
./gradlew connectedDebugAndroidTest --stacktrace
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

## Roadmap i rozwój

Projekt jest aktywnie rozwijany. Zaplanowano **40 issues** obejmujących nowe funkcjonalności, 
optymalizacje i usprawnienia. Zobacz szczegóły:

- 📋 **[DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md)** – Pełna lista zaplanowanych issues z opisami
- 🗺️ **[docs/roadmap.md](docs/roadmap.md)** – Roadmap projektu i plan realizacji w fazach

**Przykładowe planowane funkcjonalności:**
- 🎯 Detekcja obiektów YOLO (Python)
- 📱 Real-time AprilTag/QR detection w aplikacji Android
- 🤖 REST API dla przetwarzania obrazów
- 👤 Face detection & landmarks
- ⚡ GPU acceleration i optymalizacje wydajności
- 🚀 CI/CD, automatyczne testy i publikacja na PyPI

Chcesz pomóc? Sprawdź listę issues lub zaproponuj własne pomysły w GitHub Discussions!

---

## Licencja

Projekt objęty licencją **Apache 2.0** – szczegóły w pliku [LICENSE](LICENSE).
