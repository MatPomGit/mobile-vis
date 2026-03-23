# Plan rozwoju aplikacji MobileVis

> **Data utworzenia:** 2026-03-23  
> **Cel:** Lista zagadnień (issues) planowanych do realizacji w projekcie MobileVis

---

## Spis treści

1. [Funkcjonalności Python](#1-funkcjonalności-python)
2. [Funkcjonalności Android](#2-funkcjonalności-android)
3. [Integracja Python-Android](#3-integracja-python-android)
4. [Dokumentacja i przykłady](#4-dokumentacja-i-przykłady)
5. [Testy i jakość kodu](#5-testy-i-jakość-kodu)
6. [Infrastruktura i CI/CD](#6-infrastruktura-i-cicd)
7. [Optymalizacja i wydajność](#7-optymalizacja-i-wydajność)
8. [Inne usprawnienia](#8-inne-usprawnienia)

---

## 1. Funkcjonalności Python

### Issue #1: Dodanie detekcji obiektów YOLO
**Priorytet:** Wysoki  
**Etykiety:** `enhancement`, `python`, `detection`

**Opis:**  
Zaimplementować detekcję obiektów przy użyciu modeli YOLO (YOLOv8/YOLOv9) z biblioteki `ultralytics`.

**Zakres prac:**
- Dodanie nowego modułu `src/image_analysis/yolo_detection.py`
- Funkcja `detect_objects_yolo()` zwracająca bounding boxy, klasy i confidence scores
- Funkcja `draw_yolo_detections()` do wizualizacji wykrytych obiektów
- Wsparcie dla własnych modeli trenowanych przez użytkownika
- Testy jednostkowe w `tests/test_yolo_detection.py`
- Dokumentacja i przykład użycia w `notebooks/`

**Zależności:**
- Instalacja `ultralytics>=8.0`
- Pobranie pretrenowanych modeli (yolov8n.pt, yolov8s.pt)

---

### Issue #2: Segmentacja obrazu (Semantic/Instance Segmentation)
**Priorytet:** Średni  
**Etykiety:** `enhancement`, `python`, `segmentation`

**Opis:**  
Dodanie funkcjonalności segmentacji semantycznej i instancyjnej obrazów.

**Zakres prac:**
- Nowy moduł `src/image_analysis/segmentation.py`
- Integracja z modelami SAM (Segment Anything Model) lub DeepLabV3
- Funkcje `segment_image()`, `draw_segmentation_mask()`
- Wsparcie dla różnych backendów (PyTorch, ONNX)
- Testy i dokumentacja

**Wymagania:**
- PyTorch >= 2.0 lub ONNX Runtime
- Modele segmentacyjne pretrenowane na COCO/ADE20K

---

### Issue #3: Augmentacja obrazów (Data Augmentation)
**Priorytet:** Średni  
**Etykiety:** `enhancement`, `python`, `preprocessing`

**Opis:**  
Rozszerzenie modułu `preprocessing.py` o funkcje augmentacji danych dla treningu modeli.

**Zakres prac:**
- Dodanie funkcji: `random_flip()`, `random_rotation()`, `random_crop()`, `color_jitter()`
- Integracja z biblioteką `albumentations` (opcjonalnie)
- Pipeline augmentacji: `create_augmentation_pipeline()`
- Przykłady w notebooku Jupyter
- Testy jednostkowe

---

### Issue #4: Wykrywanie twarzy i punktów charakterystycznych (Face Detection & Landmarks)
**Priorytet:** Średni  
**Etykiety:** `enhancement`, `python`, `detection`

**Opis:**  
Implementacja detekcji twarzy i punktów charakterystycznych (oczy, nos, usta) przy użyciu MediaPipe lub dlib.

**Zakres prac:**
- Nowy moduł `src/image_analysis/face_detection.py`
- Funkcje: `detect_faces()`, `detect_facial_landmarks()`, `draw_landmarks()`
- Wsparcie dla wielu twarzy w jednym obrazie
- Dataclass `FaceDetection` z bounding boxem i landmarks
- Testy i dokumentacja

**Biblioteki:**
- `mediapipe>=0.10` lub `dlib>=19.24`

---

### Issue #5: Wykrywanie barcode (1D/2D)
**Priorytet:** Niski  
**Etykiety:** `enhancement`, `python`, `detection`

**Opis:**  
Dodanie wsparcia dla wykrywania i dekodowania kodów kreskowych (EAN-13, Code128, DataMatrix, itp.).

**Zakres prac:**
- Rozszerzenie modułu `qr_detection.py` lub nowy moduł `barcode_detection.py`
- Integracja z biblioteką `pyzbar`
- Funkcje: `detect_barcodes()`, `draw_barcodes()`
- Testy dla różnych typów kodów kreskowych

---

### Issue #6: Histogram i korekcja kolorów (Color Correction)
**Priorytet:** Niski  
**Etykiety:** `enhancement`, `python`, `preprocessing`

**Opis:**  
Dodanie funkcji do analizy i korekcji histogramu kolorów obrazu.

**Zakres prac:**
- Funkcje w `preprocessing.py`: `histogram_equalization()`, `clahe()`, `white_balance()`
- Analiza i wizualizacja histogramów RGB/HSV
- Automatyczna korekcja balansu bieli
- Testy i przykłady

---

### Issue #7: Eksport wyników do JSON/CSV
**Priorytet:** Niski  
**Etykiety:** `enhancement`, `python`, `utils`

**Opis:**  
Umożliwienie eksportu wyników detekcji (AprilTag, QR, obiekty) do formatów JSON i CSV.

**Zakres prac:**
- Dodanie funkcji w `utils.py`: `export_detections_to_json()`, `export_detections_to_csv()`
- Wsparcie dla wszystkich typów detekcji (AprilTag, QR, YOLO, face)
- Czytelny format JSON z metadanymi (timestamp, image path, detections)
- Testy serializacji/deserializacji

---

## 2. Funkcjonalności Android

### Issue #8: Dodanie filtrów AprilTag i QR do aplikacji Android
**Priorytet:** Wysoki  
**Etykiety:** `enhancement`, `android`, `detection`

**Opis:**  
Integracja wykrywania AprilTag i kodów QR bezpośrednio w aplikacji mobilnej z wizualizacją na żywo.

**Zakres prac:**
- Rozszerzenie `OpenCvFilter.kt` o opcje: `APRILTAG_DETECTION`, `QR_DETECTION`
- Implementacja w `ImageProcessor.kt` używając OpenCV ArUco i QRCodeDetector
- Rysowanie bounding boxów i etykiet na podglądzie kamery
- Optymalizacja wydajności dla real-time processing (downsampling, ograniczenie FPS)
- Testy jednostkowe (Mockito/Robolectric)

---

### Issue #9: Zapisywanie zdjęć z nałożonymi filtrami
**Priorytet:** Średni  
**Etykiety:** `enhancement`, `android`, `ui`

**Opis:**  
Dodanie przycisku do zapisywania aktualnej klatki z zastosowanym filtrem do galerii telefonu.

**Zakres prac:**
- Przycisk FAB "Zrób zdjęcie" w `MainActivity.kt`
- Funkcja `saveImageToGallery()` z obsługą MediaStore API (Android 10+)
- Wyświetlanie Snackbar z potwierdzeniem zapisu
- Uprawnienia WRITE_EXTERNAL_STORAGE (Android 9 i starsze)
- Dodanie ikony kamery w zasobach `res/drawable/`

---

### Issue #10: Wybór rozdzielczości kamery
**Priorytet:** Niski  
**Etykiety:** `enhancement`, `android`, `ui`

**Opis:**  
Umożliwienie użytkownikowi wyboru rozdzielczości podglądu kamery (480p, 720p, 1080p).

**Zakres prac:**
- Menu ustawień w `MainActivity.kt` (opcje ActionBar lub Settings Activity)
- Zmiana konfiguracji `ImageAnalysis` w CameraX
- Zapisywanie preferencji użytkownika w `SharedPreferences`
- Restart kamery po zmianie rozdzielczości

---

### Issue #11: Filtr detekcji krawędzi w kolorze (Color Edge Detection)
**Priorytet:** Niski  
**Etykiety:** `enhancement`, `android`, `filter`

**Opis:**  
Dodanie zaawansowanego filtra detekcji krawędzi z zachowaniem informacji o kolorze.

**Zakres prac:**
- Implementacja w `ImageProcessor.kt`: detekcja krawędzi dla każdego kanału RGB osobno
- Połączenie wyników w kolorowy obraz krawędzi
- Nowa opcja w `OpenCvFilter.kt`: `COLOR_EDGES`
- Możliwość regulacji progów Canny dla każdego kanału

---

### Issue #12: Tryb nocny i dostosowanie jasności
**Priorytet:** Niski  
**Etykiety:** `enhancement`, `android`, `ui`

**Opis:**  
Dodanie możliwości regulacji jasności podglądu kamery oraz trybu nocnego UI aplikacji.

**Zakres prac:**
- SeekBar do regulacji jasności w `MainActivity.kt`
- Aplikowanie korekcji gamma/ekspozycji do klatek w `ImageProcessor.kt`
- Material Design 3 Dark Theme w `res/values/themes.xml`
- Przełącznik Day/Night mode w menu

---

### Issue #13: Wskaźnik FPS i informacje diagnostyczne
**Priorytet:** Niski  
**Etykiety:** `enhancement`, `android`, `ui`

**Opis:**  
Wyświetlanie liczby klatek na sekundę (FPS) oraz czasu przetwarzania na ekranie.

**Zakres prac:**
- TextView w `activity_main.xml` do wyświetlania FPS
- Pomiar czasu przetwarzania każdej klatki w `ImageProcessor.kt`
- Obliczanie średniego FPS (moving average)
- Możliwość włączenia/wyłączenia wyświetlania (opcja w menu)

---

### Issue #14: Obsługa gestów (pinch-to-zoom)
**Priorytet:** Niski  
**Etykiety:** `enhancement`, `android`, `ui`

**Opis:**  
Dodanie obsługi gestów pinch-to-zoom do powiększania/pomniejszania podglądu kamery.

**Zakres prac:**
- Implementacja `ScaleGestureDetector` w `MainActivity.kt`
- Kontrola zoomu przez CameraX `Camera.cameraControl.setZoomRatio()`
- Płynna animacja zmiany zoomu
- Ograniczenia min/max zoom na podstawie możliwości kamery

---

## 3. Integracja Python-Android

### Issue #15: Biblioteka wspólna dla Python i Android (Shared C++ Core)
**Priorytet:** Wysoki  
**Etykiety:** `enhancement`, `integration`, `c++`

**Opis:**  
Stworzenie wspólnego rdzenia w C++ używanego zarówno przez moduły Python (pybind11), jak i Android (JNI).

**Zakres prac:**
- Katalog `cpp/` z kodem źródłowym C++ dla kluczowych algorytmów
- Binding dla Python przez `pybind11`
- Binding dla Android przez JNI/CMake
- Implementacja przynajmniej jednego filtra w C++ (np. Canny Edge)
- Testy jednostkowe dla C++ (Google Test)
- Dokumentacja budowania

**Zalety:**
- Zwiększona wydajność
- Spójność algorytmów między platformami
- Łatwiejsze utrzymanie kodu

---

### Issue #16: REST API dla przetwarzania obrazów
**Priorytet:** Średni  
**Etykiety:** `enhancement`, `python`, `backend`

**Opis:**  
Stworzenie REST API (FastAPI/Flask) umożliwiającego wysyłanie obrazów z aplikacji Android do serwera Python do przetworzenia.

**Zakres prac:**
- Nowy katalog `backend/` z aplikacją FastAPI
- Endpointy: `/detect/apriltag`, `/detect/qr`, `/detect/yolo`, `/classify`
- Upload obrazu, przetworzenie, zwrot JSON z wynikami
- Opcjonalnie: zwrot obrazu z nałożonymi adnotacjami
- Dokumentacja Swagger/OpenAPI
- Docker image dla łatwego wdrożenia

---

### Issue #17: Aplikacja Android jako klient API
**Priorytet:** Średni  
**Etykiety:** `enhancement`, `android`, `integration`

**Opis:**  
Rozszerzenie aplikacji Android o możliwość wysyłania zdjęć do serwera Python API do przetworzenia.

**Zakres prac:**
- Integracja Retrofit/OkHttp w Android do komunikacji z API
- Nowa aktywność `ServerProcessingActivity.kt`
- Upload zdjęcia na serwer, odbieranie i wyświetlanie wyników
- Obsługa błędów sieciowych (retry, timeout)
- Konfiguracja URL serwera w ustawieniach aplikacji

---

## 4. Dokumentacja i przykłady

### Issue #18: Przykłady Jupyter Notebooks dla wszystkich modułów
**Priorytet:** Średni  
**Etykiety:** `documentation`, `examples`

**Opis:**  
Stworzenie kompleksowych notebooków Jupyter demonstrujących użycie każdego modułu biblioteki.

**Zakres prac:**
- `notebooks/01_preprocessing.ipynb` - wczytywanie, resize, normalizacja
- `notebooks/02_apriltag_detection.ipynb` - detekcja i wizualizacja AprilTag
- `notebooks/03_qr_detection.ipynb` - detekcja i dekodowanie QR
- `notebooks/04_object_detection.ipynb` - YOLO detection (jeśli zaimplementowane)
- `notebooks/05_classification.ipynb` - klasyfikacja obrazów
- `notebooks/06_full_pipeline.ipynb` - kompletny pipeline przetwarzania
- Wizualizacje wyników, wykresy, metryki

---

### Issue #19: Video tutorial - obsługa aplikacji Android
**Priorytet:** Niski  
**Etykiety:** `documentation`, `android`

**Opis:**  
Nagranie krótkiego video (2-3 min) pokazującego instalację i użycie aplikacji Android MobileCV.

**Zakres prac:**
- Screencast z demonstracją głównych funkcji
- Prezentacja wszystkich filtrów OpenCV
- Przełączanie kamery przedniej/tylnej
- Nagranie w rozdzielczości 1080p, 30fps
- Upload na YouTube / Vimeo
- Link w README.md i docs/index.md

---

### Issue #20: Generowanie dokumentacji API (Sphinx)
**Priorytet:** Średni  
**Etykiety:** `documentation`, `python`

**Opis:**  
Automatyczne generowanie dokumentacji API dla biblioteki Python przy użyciu Sphinx.

**Zakres prac:**
- Konfiguracja Sphinx w katalogu `docs_api/`
- Generowanie HTML z docstringów modułów
- Tematyka: Read the Docs lub Furo
- Automatyczne budowanie przez GitHub Actions
- Publikacja na GitHub Pages

---

### Issue #21: Dokumentacja architektury Android (UML)
**Priorytet:** Niski  
**Etykiety:** `documentation`, `android`

**Opis:**  
Stworzenie diagramów UML (klasy, sekwencji) dokumentujących architekturę aplikacji Android.

**Zakres prac:**
- Diagram klas dla `MainActivity`, `ImageProcessor`, `OpenCvFilter`
- Diagram sekwencji dla przepływu przetwarzania klatki
- Użycie PlantUML lub Mermaid
- Osadzenie diagramów w `docs/android_architecture.md`

---

## 5. Testy i jakość kodu

### Issue #22: Zwiększenie pokrycia testów do 90%+
**Priorytet:** Wysoki  
**Etykiety:** `testing`, `python`

**Opis:**  
Uzupełnienie testów jednostkowych dla modułów Python, aby osiągnąć pokrycie > 90%.

**Zakres prac:**
- Dodanie testów dla edge cases (puste obrazy, błędne formaty, bardzo duże obrazy)
- Testy parametryzowane dla różnych typów wejść
- Mockowanie zależności zewnętrznych (OpenCV, file I/O)
- Raport coverage z oznaczeniem brakujących linii
- Integracja z Codecov lub Coveralls

---

### Issue #23: Testy wydajnościowe (Performance Benchmarks)
**Priorytet:** Średni  
**Etykiety:** `testing`, `performance`

**Opis:**  
Stworzenie suite testów wydajnościowych mierzących czas wykonania kluczowych operacji.

**Zakres prac:**
- Użycie `pytest-benchmark` do pomiaru czasu
- Benchmarki dla: `load_image()`, `resize_image()`, `detect_april_tags()`, `detect_qr_codes()`
- Porównanie wydajności dla różnych rozmiarów obrazów (480p, 720p, 1080p, 4K)
- Wykresy i raporty HTML
- Testy regresji wydajności w CI/CD

---

### Issue #24: Testy integracyjne dla Android (Espresso)
**Priorytet:** Średni  
**Etykiety:** `testing`, `android`

**Opis:**  
Dodanie testów UI dla aplikacji Android przy użyciu Espresso/UI Automator.

**Zakres prac:**
- Katalog `android/app/src/androidTest/`
- Testy: uruchomienie aplikacji, przełączanie filtrów, przełączanie kamery
- Testy interakcji z UI (kliknięcia, swipe)
- Uruchamianie na emulatorze w CI/CD
- Minimum 5 testów scenariuszowych

---

### Issue #25: Pre-commit hooks dla Kotlin/Java
**Priorytet:** Niski  
**Etykiety:** `tooling`, `android`

**Opis:**  
Dodanie pre-commit hooks dla kodu Android (Kotlin/Java) sprawdzających formatowanie i linting.

**Zakres prac:**
- Konfiguracja ktlint dla Kotlin
- Dodanie do `.pre-commit-config.yaml`
- Automatyczne formatowanie przed commitem
- Sprawdzanie w GitHub Actions CI
- Dokumentacja w README

---

## 6. Infrastruktura i CI/CD

### Issue #26: GitHub Actions workflow dla Python (lint, test, coverage)
**Priorytet:** Wysoki  
**Etykiety:** `ci/cd`, `python`

**Opis:**  
Utworzenie workflow GitHub Actions automatycznie uruchamiającego testy i lintery dla kodu Python.

**Zakres prac:**
- Plik `.github/workflows/python-ci.yml`
- Jobs: lint (ruff, mypy), test (pytest), coverage (pytest-cov)
- Matrix testing dla Python 3.11, 3.12, 3.13
- Upload coverage do Codecov
- Badge w README.md pokazujący status CI

---

### Issue #27: GitHub Actions workflow dla Android (build, test)
**Priorytet:** Wysoki  
**Etykiety:** `ci/cd`, `android`

**Opis:**  
Utworzenie workflow GitHub Actions do budowania aplikacji Android i uruchamiania testów.

**Zakres prac:**
- Plik `.github/workflows/android-ci.yml`
- Jobs: build APK (debug i release), unit tests, instrumentation tests
- Cache Gradle dependencies dla szybszego buildu
- Artefakty: upload APK jako GitHub artifact
- Badge w README.md

---

### Issue #28: Automatyczne releases (Semantic Versioning)
**Priorytet:** Średni  
**Etykiety:** `ci/cd`, `automation`

**Opis:**  
Automatyzacja tworzenia release'ów zgodnie z Semantic Versioning na podstawie commit messages.

**Zakres prac:**
- Integracja z `semantic-release` lub `release-please`
- Automatyczne generowanie CHANGELOG.md
- Tagowanie wersji w Git
- Publikacja release na GitHub z artifacts (APK, wheel package)
- Dokumentacja dla developerów

---

### Issue #29: Docker image dla środowiska Python
**Priorytet:** Niski  
**Etykiety:** `infrastructure`, `python`

**Opis:**  
Stworzenie Dockerfile dla łatwego uruchomienia środowiska Python z zainstalowanymi zależnościami.

**Zakres prac:**
- `Dockerfile` bazujący na `python:3.11-slim`
- Instalacja wszystkich dependencies z `requirements.txt`
- Expose portu dla Jupyter Notebook (8888)
- Docker Compose z konfiguracją dla Jupyter i ewentualnie API backend
- Dokumentacja użycia

---

### Issue #30: Publikacja pakietu Python na PyPI
**Priorytet:** Niski  
**Etykiety:** `infrastructure`, `python`

**Opis:**  
Przygotowanie i publikacja biblioteki `image_analysis` jako pakietu na PyPI.

**Zakres prac:**
- Konfiguracja `setup.py` lub rozszerzenie `pyproject.toml`
- Testowanie buildu wheel: `python -m build`
- Publikacja na TestPyPI do weryfikacji
- Publikacja na PyPI
- Automatyzacja publikacji przez GitHub Actions (on release tag)

---

## 7. Optymalizacja i wydajność

### Issue #31: Optymalizacja przetwarzania obrazów w Python (NumPy vectorization)
**Priorytet:** Średni  
**Etykiety:** `performance`, `python`

**Opis:**  
Refaktoryzacja kodu w celu maksymalnego wykorzystania operacji wektoryzowanych NumPy i unikania pętli Python.

**Zakres prac:**
- Analiza profilowania: `cProfile`, `line_profiler`
- Identyfikacja bottlenecks
- Refaktoryzacja kodu w `preprocessing.py`, `detection.py`
- Benchmarki przed/po optymalizacji
- Dokumentacja osiągniętych speed-upów

---

### Issue #32: GPU acceleration dla Python (CUDA/OpenCL)
**Priorytet:** Niski  
**Etykiety:** `performance`, `python`, `gpu`

**Opis:**  
Dodanie wsparcia dla akceleracji GPU w operacjach obrazu przy użyciu `cv2.cuda` lub PyTorch.

**Zakres prac:**
- Wrapper funkcji z automatycznym wykrywaniem dostępności GPU
- Fallback do CPU jeśli GPU niedostępne
- Implementacja GPU dla: resize, Canny, threshold
- Benchmarki CPU vs GPU
- Dokumentacja wymagań (CUDA Toolkit, cuDNN)

---

### Issue #33: Optymalizacja zużycia pamięci w Android
**Priorytet:** Średni  
**Etykiety:** `performance`, `android`

**Opis:**  
Redukcja zużycia pamięci RAM aplikacji Android podczas przetwarzania klatek high-resolution.

**Zakres prac:**
- Profiling przez Android Profiler (Memory, CPU)
- Recykling bitmap przez `BitmapPool`
- Optymalizacja rozmiaru bufora w `ImageAnalysis`
- Downsampling klatek przed przetwarzaniem dla ciężkich filtrów
- Mierzenie before/after memory footprint

---

### Issue #34: Lazy loading modeli ML w Python
**Priorytet:** Niski  
**Etykiety:** `performance`, `python`

**Opis:**  
Implementacja leniwego ładowania modeli ML tylko gdy są faktycznie używane.

**Zakres prac:**
- Refaktoryzacja `classification.py`: model ładowany dopiero przy pierwszym wywołaniu
- Singleton pattern dla modeli
- Cache modeli w pamięci
- Zmniejszenie czasu startu aplikacji
- Testy jednostkowe

---

## 8. Inne usprawnienia

### Issue #35: Wsparcie dla wielu języków w aplikacji Android (i18n)
**Priorytet:** Niski  
**Etykiety:** `enhancement`, `android`, `i18n`

**Opis:**  
Dodanie tłumaczeń aplikacji na języki: angielski, polski, niemiecki.

**Zakres prac:**
- Pliki `res/values-en/strings.xml`, `res/values-pl/strings.xml`, `res/values-de/strings.xml`
- Tłumaczenie wszystkich tekstów UI
- Testowanie layoutów dla różnych długości tekstów
- Automatyczna detekcja języka systemu

---

### Issue #36: Tryb batch processing w Python CLI
**Priorytet:** Niski  
**Etykiety:** `enhancement`, `python`, `cli`

**Opis:**  
Stworzenie CLI (Command Line Interface) do przetwarzania wielu obrazów w trybie batch.

**Zakres prac:**
- Moduł `src/image_analysis/cli.py` z biblioteką `click` lub `argparse`
- Komendy: `detect-apriltag`, `detect-qr`, `classify`, `apply-filter`
- Obsługa wildcards dla wielu plików: `*.jpg`
- Progress bar (tqdm)
- Eksport wyników do JSON/CSV
- Instalacja jako entry point w `pyproject.toml`

---

### Issue #37: Logo i ikona aplikacji Android
**Priorytet:** Niski  
**Etykiety:** `design`, `android`

**Opis:**  
Zaprojektowanie i wdrożenie profesjonalnego logo aplikacji oraz ikony.

**Zakres prac:**
- Design logo w kilku wariantach (full color, monochrome, adaptive)
- Ikona aplikacji w różnych rozmiarach (mdpi, hdpi, xhdpi, xxhdpi, xxxhdpi)
- Adaptive icon dla Android 8.0+ (`res/mipmap-anydpi-v26/`)
- Splash screen z logo (Android 12+ SplashScreen API)
- Zasoby w formacie wektorowym (SVG/PDF) + PNG

---

### Issue #38: Dark mode dla dokumentacji (Sphinx/MkDocs)
**Priorytet:** Niski  
**Etykiety:** `documentation`, `design`

**Opis:**  
Dodanie trybu ciemnego do dokumentacji technicznej projektu.

**Zakres prac:**
- Wybór tematu Sphinx z wsparciem dark mode (Furo, Material)
- Konfiguracja przełącznika light/dark
- Dostosowanie kolorów kodu source code
- Testowanie czytelności w obu trybach

---

### Issue #39: GitHub Discussions - Q&A i roadmap
**Priorytet:** Niski  
**Etykiety:** `community`

**Opis:**  
Uruchomienie GitHub Discussions dla społeczności użytkowników i developerów.

**Zakres prac:**
- Włączenie Discussions w ustawieniach repo
- Utworzenie kategorii: Q&A, Ideas, Show and tell, General
- Pin roadmap discussion z planowanymi features
- Zachęcenie użytkowników do zadawania pytań
- Link w README.md

---

### Issue #40: Contributing guidelines (CONTRIBUTING.md)
**Priorytet:** Średni  
**Etykiety:** `documentation`, `community`

**Opis:**  
Stworzenie dokumentu CONTRIBUTING.md wyjaśniającego jak współtworzyć projekt.

**Zakres prac:**
- Instrukcje setup środowiska developerskiego
- Zasady tworzenia Pull Requests
- Code review process
- Commit message conventions (Conventional Commits)
- Linki do Code of Conduct
- Informacje o licencji wkładu

---

## Podsumowanie

**Łącznie:** 40 issues  
**Kategorie:**
- Funkcjonalności Python: 7 issues
- Funkcjonalności Android: 7 issues
- Integracja Python-Android: 3 issues
- Dokumentacja i przykłady: 4 issues
- Testy i jakość kodu: 4 issues
- Infrastruktura i CI/CD: 5 issues
- Optymalizacja i wydajność: 4 issues
- Inne usprawnienia: 6 issues

**Priorytety:**
- Wysoki: 6 issues
- Średni: 14 issues
- Niski: 20 issues

---

## Kolejność realizacji (rekomendowana)

### Faza 1: Podstawy (Sprint 1-2)
1. Issue #26: GitHub Actions workflow dla Python
2. Issue #27: GitHub Actions workflow dla Android
3. Issue #22: Zwiększenie pokrycia testów do 90%+
4. Issue #40: Contributing guidelines

### Faza 2: Kluczowe funkcjonalności (Sprint 3-5)
5. Issue #1: Dodanie detekcji obiektów YOLO
6. Issue #8: Dodanie filtrów AprilTag i QR do aplikacji Android
7. Issue #18: Przykłady Jupyter Notebooks dla wszystkich modułów
8. Issue #20: Generowanie dokumentacji API (Sphinx)

### Faza 3: Rozszerzenia (Sprint 6-8)
9. Issue #4: Wykrywanie twarzy i punktów charakterystycznych
10. Issue #9: Zapisywanie zdjęć z nałożonymi filtrami
11. Issue #16: REST API dla przetwarzania obrazów
12. Issue #23: Testy wydajnościowe

### Faza 4: Optymalizacja i dopracowanie (Sprint 9-10)
13. Issue #31: Optymalizacja przetwarzania obrazów
14. Issue #33: Optymalizacja zużycia pamięci w Android
15. Issue #28: Automatyczne releases
16. Issue #30: Publikacja pakietu Python na PyPI

### Faza 5: Community i long-term (ciągłe)
17. Pozostałe issues według potrzeb i feedback użytkowników

---

**Uwaga:** Lista ta jest propozycją i może być modyfikowana na podstawie feedback'u użytkowników, dostępnych zasobów oraz priorytetów biznesowych. Każde issue powinno być utworzone osobno w GitHub Issues z odpowiednimi etykietami i opisem.
