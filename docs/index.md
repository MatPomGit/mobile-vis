# Dokumentacja projektu – MobileVis / image_analysis

Niniejszy dokument zawiera obszerną instrukcję **instalacji**, **działania** oraz **używania**
projektu. Projekt składa się z dwóch części:

1. **Biblioteka Python `image_analysis`** – moduły do przetwarzania obrazu uruchamiane na
   komputerze (analiza, detekcja obiektów, AprilTagi, kody QR, klasyfikacja).
2. **Aplikacja Android MobileCV** – podgląd kamery na żywo z nakładanymi w czasie rzeczywistym
   filtrami OpenCV.

---

## Spis treści

1. [Wymagania wstępne](#1-wymagania-wstępne)
2. [Instalacja – Python](#2-instalacja--python)
3. [Instalacja – aplikacja Android](#3-instalacja--aplikacja-android)
4. [Architektura projektu](#4-architektura-projektu)
5. [Opis modułów Python](#5-opis-modułów-python)
   - [preprocessing](#51-moduł-preprocessing)
   - [detection](#52-moduł-detection)
   - [april_tags](#53-moduł-april_tags)
   - [qr_detection](#54-moduł-qr_detection)
   - [classification](#55-moduł-classification)
   - [utils](#56-moduł-utils)
6. [Użycie biblioteki Python – przykłady](#6-użycie-biblioteki-python--przykłady)
7. [Aplikacja Android MobileCV – działanie i obsługa](#7-aplikacja-android-mobilecv--działanie-i-obsługa)
8. [Testowanie](#8-testowanie)
9. [Narzędzia jakości kodu](#9-narzędzia-jakości-kodu)
10. [Rozszerzanie projektu](#10-rozszerzanie-projektu)
11. [Rozwiązywanie problemów](#11-rozwiązywanie-problemów)
12. [Known environment issues](#12-known-environment-issues)

---

## 1. Wymagania wstępne

### Python

| Składnik   | Minimalna wersja | Uwagi                                 |
|------------|-----------------|---------------------------------------|
| Python     | 3.11            | Wymagana obsługa `match`, PEP 673     |
| pip        | 23.x            | Instalacja zależności                 |
| Git        | 2.x             | Pobieranie repozytorium               |

Zalecane środowisko wirtualne: **`venv`** (wbudowany w Python) lub **conda**.

Biblioteki wymagane w czasie działania (instalowane automatycznie):

| Biblioteka                    | Wersja     | Zastosowanie                           |
|-------------------------------|-----------|----------------------------------------|
| `numpy`                       | ≥ 1.26    | Operacje na macierzach obrazów         |
| `opencv-python-headless`      | ≥ 4.9     | Przetwarzanie obrazu, ArUco, QR        |
| `Pillow`                      | ≥ 10.3    | Wczytywanie i zapis plików obrazów     |

Biblioteki deweloperskie (opcjonalne, instalowane przez `pip install -e ".[dev]"`):

| Biblioteka   | Wersja  | Zastosowanie                       |
|--------------|---------|------------------------------------|
| `pytest`     | ≥ 8.2   | Uruchamianie testów jednostkowych  |
| `pytest-cov` | ≥ 5.0   | Mierzenie pokrycia kodu            |
| `pytest-mock`| ≥ 3.14  | Mockowanie w testach               |
| `ruff`       | ≥ 0.4   | Lintowanie i formatowanie kodu     |
| `mypy`       | ≥ 1.10  | Statyczna analiza typów            |
| `pre-commit` | ≥ 3.7   | Automatyczne sprawdzanie przed commitem |

### Android

| Składnik               | Minimalna wersja           | Uwagi                             |
|------------------------|---------------------------|-----------------------------------|
| Android Studio         | Hedgehog 2023.1.1         | IDE + Gradle + emulator           |
| JDK                    | 17                        | Wymagany przez AGP 8.x            |
| Android SDK            | API 24 (Android 7.0)      | `minSdk` aplikacji                |
| Gradle Wrapper         | 8.6 (pobierany auto.)     | Zawarty w `android/gradlew`       |
| Urządzenie lub emulator| API 24+                   | Fizyczne urządzenie lub AVD       |

---

## 2. Instalacja – Python

### 2.1 Pobranie kodu

```bash
# Sklonuj repozytorium
git clone https://github.com/MatPomGit/mobile-vis.git
cd mobile-vis
```

### 2.2 Tworzenie wirtualnego środowiska

```bash
# Utwórz środowisko wirtualne
python -m venv .venv

# Aktywacja – Linux / macOS
source .venv/bin/activate

# Aktywacja – Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Aktywacja – Windows (cmd.exe)
.venv\Scripts\activate.bat
```

### 2.3 Instalacja zależności

**Tryb deweloperski** (zalecany – instaluje pakiet jako edytowalny oraz narzędzia deweloperskie):

```bash
pip install -e ".[dev]"
```

**Tylko zależności produkcyjne** (bez narzędzi deweloperskich):

```bash
pip install -e .
# lub bezpośrednio z requirements.txt:
pip install -r requirements.txt
```

### 2.4 Weryfikacja instalacji

```bash
python -c "import image_analysis; print('OK –', image_analysis.__file__)"
python -c "import cv2; print('OpenCV:', cv2.__version__)"
python -c "import numpy as np; print('NumPy:', np.__version__)"
```

Oczekiwany wynik:

```
OK – /path/to/mobile-vis/src/image_analysis/__init__.py
OpenCV: 4.x.x
NumPy: 1.x.x
```

### 2.5 Konfiguracja hooków pre-commit (opcjonalnie)

```bash
pre-commit install
```

Od tej chwili `ruff`, `mypy` i inne sprawdzenia będą uruchamiane automatycznie przed każdym
commitem.

### 2.6 Import behavior

Pakiet `image_analysis` wspiera model importu, który ogranicza zależności wymagane na starcie.

**Import eager (minimalny, stabilny):**

- `__version__`,
- podstawowe utility (`get_project_root`, `setup_logging`, `validate_image`,
  `safe_makedirs`, `list_images`).

**Import lazy (na pierwsze użycie):**

- moduły cięższe obliczeniowo i/lub zależne od opcjonalnych bibliotek,
  np. `yolo`, `rtmdet`, `holistic`, `iris`,
- ich publiczne symbole dostępne przez `image_analysis.<symbol>`.

W praktyce oznacza to, że sam `import image_analysis` działa poprawnie także wtedy, gdy
opcjonalne zależności (np. `ultralytics`, `mmdet`) nie są zainstalowane. Błąd pojawi się dopiero
w momencie użycia funkcji, która faktycznie wymaga brakującego pakietu.

---

## 3. Instalacja – aplikacja Android

### 3.1 Konfiguracja środowiska

1. Zainstaluj **Android Studio** (Hedgehog 2023.1.1 lub nowszy).
2. Przy pierwszym uruchomieniu Android Studio zaakceptuj umowy licencyjne i zainstaluj
   **Android SDK API 34** oraz **Build-Tools 34.x.x**.
3. Upewnij się, że zmienna `JAVA_HOME` wskazuje na JDK 17.

### 3.2 Konfiguracja local.properties

```bash
cd android

# Skopiuj przykładowy plik konfiguracyjny
cp local.properties.example local.properties
```

Otwórz `android/local.properties` i ustaw ścieżkę do Android SDK:

```properties
# Linux / macOS
sdk.dir=/home/<użytkownik>/Android/Sdk

# Windows
sdk.dir=C\:\\Users\\<użytkownik>\\AppData\\Local\\Android\\Sdk
```

> **Wskazówka:** Android Studio automatycznie generuje `local.properties` z poprawną ścieżką po
> otwarciu projektu przez IDE – nie musisz tego robić ręcznie.

### 3.3 Budowanie APK (wiersz poleceń)

```bash
cd android

# Nadaj uprawnienia wykonywania skryptowi Gradle Wrapper (Linux/macOS)
chmod +x gradlew

# Zbuduj wersję debug
./gradlew assembleDebug

# Windows
gradlew.bat assembleDebug
```

Plik APK zostanie wygenerowany w:
`android/app/build/outputs/apk/debug/app-debug.apk`

### 3.4 Instalacja na urządzeniu / emulatorze

```bash
# Zainstaluj i uruchom na podłączonym urządzeniu przez ADB
./gradlew installDebug

# lub zainstaluj ręcznie przez ADB
adb install android/app/build/outputs/apk/debug/app-debug.apk
```

### 3.5 Budowanie przez Android Studio (zalecane)

1. Otwórz Android Studio i wybierz **File → Open**.
2. Wskaż katalog `android/` (nie katalog główny repozytorium).
3. Poczekaj, aż Gradle zsynchronizuje projekt (pasek postępu na dole).
4. Podłącz urządzenie USB lub uruchom emulator (AVD Manager → Create Virtual Device).
5. Kliknij zielony przycisk **Run ▶** (Shift+F10).

> **Uwaga dotycząca OpenCV:** biblioteka OpenCV jest pobierana z Maven Central podczas budowania
> (`org.opencv:opencv:4.10.0`). Pierwsza synchronizacja Gradle wymaga dostępu do internetu i może
> zająć kilka minut.

---

## 4. Architektura projektu

### Python

```
src/image_analysis/
├── __init__.py          ← publiczne API (importy eksportowane)
├── preprocessing.py     ← wczytywanie, zmiana rozmiaru, normalizacja
├── detection.py         ← detekcja obiektów, NMS, rysowanie bboxów
├── april_tags.py        ← detekcja markerów AprilTag (ArUco)
├── qr_detection.py      ← detekcja i dekodowanie kodów QR
├── classification.py    ← klasyfikacja obrazów (stub do rozbudowy)
└── utils.py             ← walidacja obrazów, logowanie, ścieżki
```

Przepływ danych dla typowego zadania analizy obrazu:

```
Plik obrazu → load_image() → NDArray[uint8] (BGR, H×W×3)
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
             resize_image()   detect_objects()  detect_april_tags()
             normalize_image()  apply_nms()     detect_qr_codes()
                    │          draw_bounding_boxes()  draw_april_tags()
                    ▼               │               │
             NDArray[float32]       └───────┬───────┘
             [0.0, 1.0]                     ▼
                                   obraz z adnotacjami
```

### Android

```
MainActivity
    │
    ├── OpenCVLoader.initLocal()           ← inicjalizacja biblioteki OpenCV
    ├── CameraX ImageAnalysis              ← dostarczanie klatek RGBA_8888
    │       │ ImageProxy
    │       ▼
    ├── orientBitmap()                     ← obrót + odbicie dla kamery przedniej
    │       │ Bitmap (ARGB_8888)
    │       ▼
    ├── ImageProcessor.processFrame()      ← konwersja Bitmap→Mat→filtr→Mat→Bitmap
    │       │ Bitmap (ARGB_8888)
    │       ▼
    └── ImageView.setImageBitmap()         ← wyświetlanie na pełnym ekranie
```

---

## 5. Opis modułów Python

### 5.1 Moduł `preprocessing`

Odpowiada za wczytywanie obrazów z dysku i ich wstępne przygotowanie do dalszej analizy.

#### `load_image(path)`

Wczytuje plik obrazu (JPEG, PNG, BMP, TIFF i inne formaty obsługiwane przez OpenCV) jako tablicę
BGR NumPy.

```python
def load_image(path: str | Path) -> NDArray[np.uint8]:
    ...
```

| Parametr | Typ            | Opis                              |
|----------|----------------|-----------------------------------|
| `path`   | `str \| Path`  | Ścieżka do pliku obrazu           |

**Zwraca:** `NDArray[np.uint8]` – tablica o kształcie `(H, W, 3)`, format BGR, zakres `[0, 255]`.

**Wyjątki:**
- `FileNotFoundError` – plik nie istnieje
- `ValueError` – plik istnieje, ale nie można go zdekodować jako obraz

#### `resize_image(image, width, height)`

Zmienia rozmiar obrazu do podanych wymiarów. Stosuje `INTER_AREA` przy zmniejszaniu (wyższa
jakość) i `INTER_LINEAR` przy powiększaniu.

```python
def resize_image(
    image: NDArray[np.uint8] | NDArray[np.float32],
    width: int,
    height: int,
) -> NDArray[np.uint8] | NDArray[np.float32]:
    ...
```

| Parametr | Typ   | Opis                           |
|----------|-------|--------------------------------|
| `width`  | `int` | Docelowa szerokość w pikselach |
| `height` | `int` | Docelowa wysokość w pikselach  |

#### `normalize_image(image)`

Konwertuje obraz `uint8` `[0, 255]` na `float32` `[0.0, 1.0]` przez podzielenie przez 255.

```python
def normalize_image(image: NDArray[np.uint8]) -> NDArray[np.float32]:
    ...
```

---

### 5.2 Moduł `detection`

Zawiera narzędzia do wykrywania obiektów na obrazach. Funkcje `detect_objects` i `draw_bounding_boxes`
są gotowe do użycia po podłączeniu modelu; aktualnie `detect_objects` to stub zwracający pustą listę.

#### Klasa `Detection`

```python
@dataclass(frozen=True)
class Detection:
    label: str                          # nazwa klasy
    confidence: float                   # pewność w [0.0, 1.0]
    bbox: tuple[int, int, int, int]     # (x1, y1, x2, y2) w pikselach
```

#### `detect_objects(image, confidence_threshold)`

Wykrywa obiekty w obrazie i zwraca listę posortowaną malejąco po pewności predykcji.

| Stała                           | Domyślna wartość | Opis                      |
|---------------------------------|-----------------|---------------------------|
| `DETECTION_CONFIDENCE_THRESHOLD`| 0.5             | Minimalna pewność detekcji|
| `NMS_IOU_THRESHOLD`             | 0.45            | Próg IoU dla NMS          |

#### `apply_nms(detections, iou_threshold)`

Stosuje algorytm Non-Maximum Suppression (NMS) do usunięcia nakładających się bounding boxów.
Korzysta z `cv2.dnn.NMSBoxes`.

#### `draw_bounding_boxes(image, detections, color, thickness)`

Rysuje prostokąty i etykiety na kopii obrazu. Nie modyfikuje obrazu wejściowego.

---

### 5.3 Moduł `april_tags`

Wykrywa znaczniki AprilTag (fiducial markers) używając słowników ArUco z OpenCV.

#### Klasa `AprilTagDetection`

```python
@dataclass(frozen=True)
class AprilTagDetection:
    tag_id: int                             # ID znacznika (integer)
    family: str                             # rodzina, np. "tag36h11"
    corners: NDArray[np.float32]            # narożniki (4, 2) w pikselach
    center: tuple[float, float]             # środek (x, y)
    bbox: tuple[int, int, int, int]         # (x1, y1, x2, y2) wyrównany do osi
```

#### Obsługiwane rodziny AprilTag

| Rodzina      | Stała OpenCV                       | Uwagi                              |
|--------------|------------------------------------|------------------------------------|
| `tag16h5`    | `cv2.aruco.DICT_APRILTAG_16h5`    | Małe znaczniki, mniej odporne      |
| `tag25h9`    | `cv2.aruco.DICT_APRILTAG_25h9`    | Dobry balans                       |
| `tag36h10`   | `cv2.aruco.DICT_APRILTAG_36h10`   | Dobry balans                       |
| `tag36h11`   | `cv2.aruco.DICT_APRILTAG_36h11`   | **Domyślna** – najlepsza odporność |

#### `detect_april_tags(image, family)`

Przyjmuje obraz w skali szarości `(H, W)` lub BGR `(H, W, 3)`. Zwraca listę wykrytych znaczników
posortowaną rosnąco po `tag_id`.

#### `draw_april_tags(image, detections, color, thickness)`

Rysuje obrys znacznika (wielokąt), punkt centralny i etykietę `id=<n>` na kopii obrazu.

---

### 5.4 Moduł `qr_detection`

Wykrywa i dekoduje kody QR w obrazach przy użyciu `cv2.QRCodeDetector`.

#### Klasa `QRCode`

```python
@dataclass(frozen=True)
class QRCode:
    data: str                               # zdekodowana treść kodu QR
    bbox: tuple[int, int, int, int]         # (x1, y1, x2, y2)
    polygon: list[tuple[int, int]]          # narożniki wielokąta QR kodu
```

#### `detect_qr_codes(image)`

Przyjmuje obraz BGR `(H, W, 3)` lub w skali szarości `(H, W)` z dtype `uint8`. Zwraca listę
zdekodowanych kodów QR; pomija kody wykryte, ale niemożliwe do zdekodowania.

#### `draw_qr_codes(image, qr_codes, color, thickness)`

Rysuje wielokąt (lub prostokąt dla uproszczonych detekcji) oraz zdekodowany tekst nad każdym
znalezionym kodem QR.

---

### 5.5 Moduł `classification`

Zapewnia cienką warstwę abstrakcji nad modelem klasyfikatora. Aktualnie zawiera **implementacje
stub** (`classify_image` zawsze zwraca `("unknown", 0.0)`), które należy zastąpić właściwą
inferencją modelu (PyTorch, TensorFlow itp.).

#### `classify_image(image, model, confidence_threshold)`

```python
def classify_image(
    image: NDArray[np.uint8] | NDArray[np.float32],
    model: object | None = None,
    confidence_threshold: float = 0.5,
) -> tuple[str, float]:
    ...
```

Zwraca krotkę `(etykieta, pewność)`. Jeśli pewność najlepszej predykcji jest poniżej progu,
zwraca `("unknown", 0.0)`.

#### `load_classifier(model_path)`

Wczytuje model z pliku. Aktualnie zgłasza `NotImplementedError` – do zaimplementowania po
wyborze frameworka.

#### `evaluate_classifier(predictions, ground_truth)`

Oblicza metryki dla partii predykcji:
- `accuracy` – dokładność (ułamek poprawnych predykcji)
- `avg_confidence` – średnia pewność predykcji

---

### 5.6 Moduł `utils`

Ogólne funkcje pomocnicze używane przez pozostałe moduły.

#### `validate_image(image)`

Waliduje, czy obiekt jest prawidłową tablicą obrazu NumPy. Akceptowane kształty:
`(H, W)`, `(H, W, 1)`, `(H, W, 3)`, `(H, W, 4)`. Akceptowane typy: `uint8`, `float32`.

#### `get_project_root()`

Zwraca bezwzględną ścieżkę do katalogu głównego projektu (rodzic katalogu `src/`). Wynik jest
buforowany po pierwszym wywołaniu.

#### `setup_logging(level)`

Konfiguruje root logger z prostym handlerem konsolowym w formacie:
```
2024-01-15T10:30:00 | INFO     | image_analysis.preprocessing | Loaded image 'sample.jpg'
```

#### `safe_makedirs(directory)`

Tworzy katalog (i wszystkich jego rodziców) jeśli nie istnieje. Odpowiednik `mkdir -p`.

#### `list_images(directory, extensions)`

Zwraca posortowaną listę ścieżek do plików obrazów w podanym katalogu.
Domyślne rozszerzenia: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`.

---

## 6. Użycie biblioteki Python – przykłady

### 6.1 Wczytanie i zmiana rozmiaru obrazu

```python
from image_analysis.preprocessing import load_image, resize_image, normalize_image

# Wczytaj obraz
image = load_image("data/raw/photo.jpg")
print(f"Oryginalny rozmiar: {image.shape}")  # (H, W, 3)

# Zmień rozmiar do 640×480
resized = resize_image(image, width=640, height=480)

# Znormalizuj do zakresu [0.0, 1.0]
normalized = normalize_image(resized)
print(f"Dtype po normalizacji: {normalized.dtype}")  # float32
```

### 6.2 Detekcja obiektów

```python
from image_analysis.preprocessing import load_image
from image_analysis.detection import detect_objects, apply_nms, draw_bounding_boxes
import cv2

image = load_image("data/raw/scene.jpg")

# Wykryj obiekty (po podłączeniu modelu)
raw_detections = detect_objects(image, confidence_threshold=0.4)

# Zastosuj Non-Maximum Suppression
filtered = apply_nms(raw_detections, iou_threshold=0.45)

# Narysuj wyniki
annotated = draw_bounding_boxes(image, filtered)
cv2.imwrite("data/processed/scene_detected.jpg", annotated)

print(f"Wykryto {len(filtered)} obiektów:")
for det in filtered:
    print(f"  [{det.label}] pewność={det.confidence:.2%}, bbox={det.bbox}")
```

### 6.3 Detekcja znaczników AprilTag

```python
from image_analysis.preprocessing import load_image
from image_analysis.april_tags import detect_april_tags, draw_april_tags
import cv2

image = load_image("data/raw/markers.jpg")

# Wykryj znaczniki AprilTag rodziny tag36h11 (domyślna)
tags = detect_april_tags(image)

print(f"Znaleziono {len(tags)} znaczników AprilTag:")
for tag in tags:
    print(f"  ID={tag.tag_id}, środek={tag.center}, bbox={tag.bbox}")

# Narysuj adnotacje
annotated = draw_april_tags(image, tags, color=(0, 255, 255), thickness=2)
cv2.imwrite("data/processed/markers_annotated.jpg", annotated)
```

### 6.4 Detekcja i dekodowanie kodów QR

```python
from image_analysis.preprocessing import load_image
from image_analysis.qr_detection import detect_qr_codes, draw_qr_codes
import cv2

image = load_image("data/raw/qrcodes.jpg")

# Wykryj i zdekoduj kody QR
codes = detect_qr_codes(image)

print(f"Znaleziono {len(codes)} kodów QR:")
for qr in codes:
    print(f"  Treść: '{qr.data}', bbox={qr.bbox}")

# Narysuj adnotacje
annotated = draw_qr_codes(image, codes, color=(0, 255, 0), thickness=2)
cv2.imwrite("data/processed/qrcodes_annotated.jpg", annotated)
```

### 6.5 Przetwarzanie wsadowe katalogu obrazów

```python
import logging
from image_analysis.utils import setup_logging, list_images, safe_makedirs
from image_analysis.preprocessing import load_image, resize_image
from image_analysis.april_tags import detect_april_tags, draw_april_tags
import cv2

setup_logging(logging.INFO)
logger = logging.getLogger(__name__)

input_dir = "data/raw"
output_dir = safe_makedirs("data/processed/april_tags")

for image_path in list_images(input_dir):
    logger.info("Przetwarzam: %s", image_path.name)
    image = load_image(image_path)
    resized = resize_image(image, width=800, height=600)
    tags = detect_april_tags(resized)

    if tags:
        annotated = draw_april_tags(resized, tags)
        out_path = output_dir / image_path.name
        cv2.imwrite(str(out_path), annotated)
        logger.info("  Zapisano %d tagów do %s", len(tags), out_path.name)
    else:
        logger.info("  Brak tagów")
```

### 6.6 Konfigurowanie logowania

```python
import logging
from image_analysis.utils import setup_logging

# Domyślne poziomy: DEBUG, INFO, WARNING, ERROR, CRITICAL
setup_logging(logging.DEBUG)   # wszystkie komunikaty
setup_logging(logging.WARNING) # tylko ostrzeżenia i błędy
```

---

## 7. Aplikacja Android MobileCV – działanie i obsługa

### 7.1 Pierwsze uruchomienie

Po zainstalowaniu APK i pierwszym uruchomieniu aplikacja:

1. Wyświetla systemowy dialog z prośbą o **uprawnienie do kamery**.
2. Jeśli uprawnienie zostało przyznane – uruchamia podgląd kamery.
3. Jeśli uprawnienie zostało odrzucone – wyświetla komunikat i zamknij aplikację.

> Jeśli przypadkowo odmówiłeś uprawnień, przejdź do **Ustawienia → Aplikacje → MobileCV →
> Uprawnienia → Kamera** i włącz dostęp.

### 7.2 Interfejs użytkownika

```
┌─────────────────────────────────────────────┐
│                                             │
│          Podgląd kamery (pełny ekran)       │
│                                             │
│  [Etykieta aktywnego filtra]               │
│                                             │
├─────────────────────────────────────────────┤
│ ○ Original  ○ Grayscale  ○ Canny  ○ Blur … │  ← paski chipów (przewijane)
├─────────────────────────────────────────────┤
│                                     [📷⟳]  │  ← FAB przełączania kamer
└─────────────────────────────────────────────┘
```

| Element UI             | Opis                                                              |
|------------------------|-------------------------------------------------------------------|
| Podgląd pełnoekranowy  | Przetworzona klatka z kamery wyświetlana w `ImageView`            |
| Etykieta filtra        | Nazwa aktywnego filtra w lewym górnym rogu ekranu                 |
| Paski chipów           | Poziomy, przewijalny pasek z jednorazowym wyborem filtra          |
| FAB (prawy dół)        | Przełącza między kamerą tylną a przednią                          |

### 7.3 Aktualny workflow trybów Android

| Grupa | Zakładki | Zawartość |
|------|----------|-----------|
| **Detekcja** | `MARKERS`, `POSE`, `YOLO` | Markery wizualne, MediaPipe oraz detekcja/segmentacja/poza YOLO |
| **Odometria** | `ODOMETRY` | Jeden przebieg: `VISUAL_ODOMETRY` → `FULL_ODOMETRY` → `ODOMETRY_TRAJECTORY` → `ODOMETRY_MAP` |
| **SLAM** | `SLAM` | Połączenie punktów odometrii i markerów (`SLAM_MARKERS`) |
| **Dynamic ROI** | `ACTIVE_TRACKING` | Stabilizowane śledzenie obiektu/ROI (`YOLO_KALMAN`, `MARKER_UKF`) |

Pozostałe tryby (`Filtry`, `Krawędzie`, `Morfologia`, `Efekty`, `Kalibracja`) pozostają bez zmian i służą do szybkich eksperymentów oraz diagnostyki obrazu.

### 7.4 Przełączanie filtrów

- Dotknij chipa z nazwą filtra w poziomym pasku na dole ekranu.
- Aktywny chip zostaje zaznaczony, a etykieta nad paskiem aktualizuje się.
- Zmiana jest natychmiastowa – kolejna klatka z kamery zostanie przetworzona nowym filtrem.

### 7.5 Przełączanie kamer

- Naciśnij przycisk FAB (ikona kamery z symbolem odświeżania) w prawym dolnym rogu.
- Aplikacja ponownie wiąże use-case CameraX z nowym obiektywem.
- Obraz z kamery przedniej jest automatycznie odbijany lustrzanie (tak jak w lustrze).

### 7.6 Architektura przetwarzania klatek

```
CameraX ImageAnalysis
    └─ OUTPUT_IMAGE_FORMAT_RGBA_8888
    └─ setTargetResolution(640×480)
    └─ STRATEGY_KEEP_ONLY_LATEST      ← odrzucaj klatki przy przeciążeniu
           │ ImageProxy
           ▼ (wątek analysisExecutor – jeden wątek)
    imageProxy.toBitmap()             → Bitmap ARGB_8888
    orientBitmap(rotation, lensFacing) → Bitmap (obrót + odbicie)
    ImageProcessor.processFrame()     → Bitmap ARGB_8888
           │ (wątek główny UI)
    ImageView.setImageBitmap()        → wyświetlenie
```

**Kluczowe decyzje projektowe:**
- `STRATEGY_KEEP_ONLY_LATEST` zapobiega budowaniu kolejki klatek – jeśli przetwarzanie jest
  wolniejsze niż dostarczanie klatek, pośrednie klatki są pomijane.
- Jeden wątek (`newSingleThreadExecutor`) eliminuje problemy ze współbieżnym dostępem do
  `ImageProcessor` (który nie jest bezpieczny wątkowo).
- `Utils.bitmapToMat` konwertuje ARGB_8888 → BGRA Mat (OpenCV używa BGR, nie RGB).

---

## 8. Testowanie

### 8.1 Uruchomienie testów

```bash
# Wszystkie testy z krótkim podsumowaniem
pytest

# Z raportowaniem pokrycia kodu w konsoli
pytest --cov=src/image_analysis --cov-report=term-missing

# Tylko jeden moduł
pytest tests/test_preprocessing.py -v

# Tylko jeden moduł z filtrem nazwy testu
pytest tests/test_april_tags.py -v -k "grayscale"

# Pełny raport HTML pokrycia kodu
pytest --cov=src/image_analysis --cov-report=html
# Otwórz: htmlcov/index.html
```

### 8.2 Wymagane pokrycie kodu

Konfiguracja w `pyproject.toml` wymaga ≥ 80 % pokrycia dla każdego modułu:

```toml
[tool.coverage.report]
fail_under = 80
```

### 8.3 Pisanie testów – konwencje

```python
import numpy as np
import pytest
from image_analysis.preprocessing import load_image, resize_image


@pytest.fixture
def bgr_image() -> np.ndarray:
    """Syntetyczny obraz BGR 100×100 do testów."""
    return np.zeros((100, 100, 3), dtype=np.uint8)


def test_resize_image_shape(bgr_image: np.ndarray) -> None:
    result = resize_image(bgr_image, width=50, height=40)
    assert result.shape == (40, 50, 3)


@pytest.mark.parametrize("width,height", [(1, 1), (320, 240), (1920, 1080)])
def test_resize_image_various_sizes(bgr_image: np.ndarray, width: int, height: int) -> None:
    result = resize_image(bgr_image, width=width, height=height)
    assert result.shape == (height, width, 3)
```

Zasady:
- Używaj małych syntetycznych obrazów (np. `np.zeros((100, 100, 3), dtype=np.uint8)`).
- Mockuj operacje I/O za pomocą `pytest-mock` lub `unittest.mock`.
- Parametryzuj przypadki brzegowe: 1-pikselowe obrazy, grayscale, różne typy danych.

### 8.4 Szybka diagnostyka

Poniższe komendy pomagają szybko sprawdzić, czy lokalne środowisko zachowuje się tak samo
jak workflow CI dla Pythona i Androida.

#### Python

```bash
python -m pip install --upgrade pip
pip install -e ".[dev]"

ruff check src/ tests/
mypy src/
pytest --cov=src/image_analysis --cov-report=term-missing

python -c 'import importlib; [importlib.import_module(m) for m in [
"image_analysis",
"image_analysis.preprocessing",
"image_analysis.detection",
"image_analysis.classification",
"image_analysis.april_tags",
"image_analysis.qr_detection",
"image_analysis.utils",
]]; print("Smoke import OK")'

python scripts/smoke_perf.py --iterations 150 --warmup 30
```

Wynik `scripts/smoke_perf.py` zawiera metryki:
- `FRAME_TIME_MS_AVG` – średni czas przetworzenia klatki,
- `FRAME_TIME_MS_P95` – 95 percentyl czasu klatki,
- `FPS_AVG` – średni FPS dla scenariusza smoke.

#### Android

```bash
cd android
chmod +x gradlew

./gradlew lintDebug --stacktrace
./gradlew assembleDebug --stacktrace

# Opcjonalnie (wymagany emulator lub urządzenie):
./gradlew connectedDebugAndroidTest --stacktrace
```

---

## 9. Narzędzia jakości kodu

### 9.1 Ruff – lintowanie i formatowanie

```bash
# Sprawdź styl kodu
ruff check src/ tests/

# Napraw automatycznie
ruff check --fix src/ tests/

# Formatuj kod (podobnie do Black)
ruff format src/ tests/
```

Konfiguracja w `pyproject.toml`:
- Długość linii: **99 znaków**
- Cudzysłów: **podwójny** (`"`)
- Docelowa wersja: **Python 3.11**
- Aktywne zestawy reguł: `E`, `W`, `F`, `I`, `N`, `UP`, `ANN`, `B`, `C4`, `SIM`, `RUF`

### 9.2 Mypy – statyczna analiza typów

```bash
mypy src/
```

Tryb `strict = true` – wymagane type hints we wszystkich funkcjach publicznych.

### 9.3 Pre-commit – automatyczne sprawdzanie

```bash
# Instalacja hooków (jednorazowo po sklonowaniu)
pre-commit install

# Ręczne uruchomienie wszystkich hooków
pre-commit run --all-files
```

---

## 10. Rozszerzanie projektu

### 10.1 Dodanie nowego modułu Python

1. Utwórz plik `src/image_analysis/<nowy_modul>.py`.
2. Napisz funkcje z type hints i docstringami Google-style.
3. Wyeksportuj publiczne funkcje i klasy przez `src/image_analysis/__init__.py`:
   ```python
   from .nowy_modul import NowaDetekcja, wykryj_cos
   __all__ = [..., "NowaDetekcja", "wykryj_cos"]
   ```
4. Napisz testy w `tests/test_<nowy_modul>.py`.
5. Sprawdź jakość kodu: `ruff check src/ tests/ && mypy src/`.

### 10.2 Podłączenie modelu detekcji obiektów

W pliku `src/image_analysis/detection.py` zastąp stub w `detect_objects`:

```python
# Przykład integracji z YOLO (ultralytics)
from ultralytics import YOLO

_model = YOLO("models/yolov8n.pt")

def detect_objects(
    image: NDArray[np.uint8],
    confidence_threshold: float = DETECTION_CONFIDENCE_THRESHOLD,
) -> list[Detection]:
    _validate_bgr_image(image)
    results = _model(image, conf=confidence_threshold)[0]
    detections = [
        Detection(
            label=_model.names[int(box.cls)],
            confidence=float(box.conf),
            bbox=(int(box.xyxy[0][0]), int(box.xyxy[0][1]),
                  int(box.xyxy[0][2]), int(box.xyxy[0][3])),
        )
        for box in results.boxes
    ]
    detections.sort(key=lambda d: d.confidence, reverse=True)
    return detections
```

### 10.3 Dodanie nowego filtra do aplikacji Android

1. Dodaj nowy wpis w `OpenCvFilter.kt`:
   ```kotlin
   SEPIA("Sepia"),
   ```
2. Dodaj implementację w `ImageProcessor.kt`:
   ```kotlin
   OpenCvFilter.SEPIA -> applySepia(src)
   ```
3. Napisz metodę prywatną `applySepia(src: Mat): Mat`.
4. Zbuduj i uruchom aplikację.

---

## 11. Rozwiązywanie problemów

### Python

**Problem:** `ModuleNotFoundError: No module named 'image_analysis'`

Przyczyna: pakiet nie jest zainstalowany lub wirtualne środowisko jest nieaktywne.

```bash
# Aktywuj środowisko i zainstaluj ponownie
source .venv/bin/activate
pip install -e ".[dev]"
```

---

**Problem:** `cv2.error` lub `ImportError` przy imporcie `cv2`

Przyczyna: OpenCV nie jest zainstalowany lub zainstalowana jest wersja bez wymaganych modułów.

```bash
pip uninstall opencv-python opencv-python-headless opencv-contrib-python
pip install opencv-python-headless>=4.9
```

---

**Problem:** `ruff check` zgłasza błędy ANN (brakujące type hints)

Rozwiązanie: dodaj type hints do wszystkich funkcji publicznych. W testach adnotacje są
wyłączone (patrz konfiguracja `per-file-ignores` w `pyproject.toml`).

---

**Problem:** `mypy` zgłasza `error: Cannot find implementation or library stub for module named 'cv2'`

Rozwiązanie: upewnij się, że `ignore_missing_imports = true` jest ustawione w `pyproject.toml`
(domyślnie tak jest). Alternatywnie zainstaluj typy `opencv-stubs`.

---

### Android

**Problem:** `Gradle sync failed – SDK location not found`

Rozwiązanie: utwórz plik `android/local.properties` i ustaw `sdk.dir`.

```bash
# Linux / macOS
echo "sdk.dir=/home/<użytkownik>/Android/Sdk" > android/local.properties
```

---

**Problem:** `OpenCV initialisation failed` (toast po uruchomieniu aplikacji)

Możliwe przyczyny:
- Zbudowano APK dla architektury ABI nieobsługiwanej przez bibliotekę OpenCV.
- Plik `.so` nie został dołączony do APK.

Rozwiązanie: upewnij się, że `libs.versions.toml` odwołuje się do `org.opencv:opencv:4.10.0`
(wersja z Maven Central zawiera skompilowane biblioteki `.so` dla `arm64-v8a`, `armeabi-v7a`,
`x86`, `x86_64`).

---

**Problem:** Aplikacja prosi o uprawnienie do kamery wielokrotnie lub nie otwiera kamery

Rozwiązanie:
1. Przejdź do **Ustawienia → Aplikacje → MobileCV → Uprawnienia → Kamera** i włącz uprawnienie.
2. Upewnij się, że w `AndroidManifest.xml` jest wpis:
   ```xml
   <uses-permission android:name="android.permission.CAMERA" />
   ```

---

**Problem:** Podgląd kamery jest obrócony lub odwrócony

Zachowanie aplikacji:
- Kamera tylna: klatka jest obracana o kąt podany przez `imageProxy.imageInfo.rotationDegrees`.
- Kamera przednia: dodatkowo stosowane jest odbicie lustrzane (skala X = -1), aby podgląd
  wyglądał jak lustro.

Jeśli obraz nadal jest nieprawidłowo zorientowany, sprawdź, czy urządzenie prawidłowo raportuje
kąt obrotu sensora.

---

**Problem:** Niska wydajność / opóźniony podgląd

Strategia `STRATEGY_KEEP_ONLY_LATEST` w `ImageAnalysis` oznacza, że jeśli `ImageProcessor`
jest zbyt wolny, klatki są pomijane. Aby poprawić wydajność:
- Zmniejsz docelową rozdzielczość: `setTargetResolution(Size(320, 240))`.
- Upewnij się, że aplikacja jest uruchamiana na urządzeniu fizycznym, a nie emulatorze.

---

## 12. Known environment issues

- **OpenCV/headless:** w środowiskach bez serwera graficznego (`CI`, kontenery) funkcje okienkowe
  OpenCV (`imshow`, `waitKey`) nie są wspierane i mogą kończyć się błędem.
- **`cv2` import w kontenerze:** część obrazów bazowych nie ma kompletu bibliotek systemowych
  potrzebnych przez OpenCV — w takim przypadku import `cv2` może się nie powieść mimo poprawnej
  instalacji pakietu Python.
- **Android Emulator:** emulowana kamera i akceleracja GPU mogą być niestabilne, co wpływa na FPS
  i jakość wyników modułów realtime; rekomendowane są testy na urządzeniu fizycznym.

---

*Dokumentacja wygenerowana dla wersji `0.1.0` projektu MobileVis.*


