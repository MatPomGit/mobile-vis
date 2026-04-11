# Spis zaplanowanych issues

> **Wygenerowano:** 2026-03-23  
> **Źródło:** DEVELOPMENT_PLAN.md

Szybki spis wszystkich 40 zaplanowanych issues dla projektu MobileVis.

---

## Kategoria: Funkcjonalności Python (7 issues)

| # | Tytuł | Priorytet | Etykiety |
|---|-------|-----------|----------|
| 1 | Dodanie detekcji obiektów YOLO | Wysoki | `enhancement`, `python`, `detection` |
| 2 | Segmentacja obrazu (Semantic/Instance Segmentation) | Średni | `enhancement`, `python`, `segmentation` |
| 3 | Augmentacja obrazów (Data Augmentation) | Średni | `enhancement`, `python`, `preprocessing` |
| 4 | Wykrywanie twarzy i punktów charakterystycznych | Średni | `enhancement`, `python`, `detection` |
| 5 | Wykrywanie barcode (1D/2D) | Niski | `enhancement`, `python`, `detection` |
| 6 | Histogram i korekcja kolorów | Niski | `enhancement`, `python`, `preprocessing` |
| 7 | Eksport wyników do JSON/CSV | Niski | `enhancement`, `python`, `utils` |

---

## Kategoria: Funkcjonalności Android (7 issues)

| # | Tytuł | Priorytet | Etykiety |
|---|-------|-----------|----------|
| 8 | Dodanie filtrów AprilTag i QR do aplikacji Android | Wysoki | `enhancement`, `android`, `detection` |
| 9 | Zapisywanie zdjęć z nałożonymi filtrami | Średni | `enhancement`, `android`, `ui` |
| 10 | Wybór rozdzielczości kamery | Niski | `enhancement`, `android`, `ui` |
| 11 | Filtr detekcji krawędzi w kolorze | Niski | `enhancement`, `android`, `filter` |
| 12 | Tryb nocny i dostosowanie jasności | Niski | `enhancement`, `android`, `ui` |
| 13 | Wskaźnik FPS i informacje diagnostyczne | Niski | `enhancement`, `android`, `ui` |
| 14 | Obsługa gestów (pinch-to-zoom) | Niski | `enhancement`, `android`, `ui` |

---

## Kategoria: Integracja Python-Android (3 issues)

| # | Tytuł | Priorytet | Etykiety |
|---|-------|-----------|----------|
| 15 | Biblioteka wspólna dla Python i Android (Shared C++ Core) | Wysoki | `enhancement`, `integration`, `c++` |
| 16 | REST API dla przetwarzania obrazów | Średni | `enhancement`, `python`, `backend` |
| 17 | Aplikacja Android jako klient API | Średni | `enhancement`, `android`, `integration` |

---

## Kategoria: Dokumentacja i przykłady (4 issues)

| # | Tytuł | Priorytet | Etykiety |
|---|-------|-----------|----------|
| 18 | Przykłady Jupyter Notebooks dla wszystkich modułów | Średni | `documentation`, `examples` |
| 19 | Video tutorial - obsługa aplikacji Android | Niski | `documentation`, `android` |
| 20 | Generowanie dokumentacji API (Sphinx) | Średni | `documentation`, `python` |
| 21 | Dokumentacja architektury Android (UML) | Niski | `documentation`, `android` |

---

## Kategoria: Testy i jakość kodu (4 issues)

| # | Tytuł | Priorytet | Etykiety |
|---|-------|-----------|----------|
| 22 | Zwiększenie pokrycia testów do 90%+ | Wysoki | `testing`, `python` |
| 23 | Testy wydajnościowe (Performance Benchmarks) | Średni | `testing`, `performance` |
| 24 | Testy integracyjne dla Android (Espresso) | Średni | `testing`, `android` |
| 25 | Pre-commit hooks dla Kotlin/Java | Niski | `tooling`, `android` |

---

## Kategoria: Infrastruktura i CI/CD (5 issues)

| # | Tytuł | Priorytet | Etykiety |
|---|-------|-----------|----------|
| 26 | GitHub Actions workflow dla Python | Wysoki | `ci/cd`, `python` |
| 27 | GitHub Actions workflow dla Android | Wysoki | `ci/cd`, `android` |
| 28 | Automatyczne releases (Semantic Versioning) | Średni | `ci/cd`, `automation` |
| 29 | Docker image dla środowiska Python | Niski | `infrastructure`, `python` |
| 30 | Publikacja pakietu Python na PyPI | Niski | `infrastructure`, `python` |

---

## Kategoria: Optymalizacja i wydajność (4 issues)

| # | Tytuł | Priorytet | Etykiety |
|---|-------|-----------|----------|
| 31 | Optymalizacja przetwarzania obrazów (NumPy vectorization) | Średni | `performance`, `python` |
| 32 | GPU acceleration dla Python (CUDA/OpenCL) | Niski | `performance`, `python`, `gpu` |
| 33 | Optymalizacja zużycia pamięci w Android | Średni | `performance`, `android` |
| 34 | Lazy loading modeli ML w Python | Niski | `performance`, `python` |

---

## Kategoria: Inne usprawnienia (6 issues)

| # | Tytuł | Priorytet | Etykiety |
|---|-------|-----------|----------|
| 35 | Wsparcie dla wielu języków w aplikacji Android (i18n) | Niski | `enhancement`, `android`, `i18n` |
| 36 | Tryb batch processing w Python CLI | Niski | `enhancement`, `python`, `cli` |
| 37 | Logo i ikona aplikacji Android | Niski | `design`, `android` |
| 38 | Dark mode dla dokumentacji (Sphinx/MkDocs) | Niski | `documentation`, `design` |
| 39 | GitHub Discussions - Q&A i roadmap | Niski | `community` |
| 40 | Contributing guidelines (CONTRIBUTING.md) | Średni | `documentation`, `community` |

---

## Podsumowanie według priorytetu

| Priorytet | Liczba issues | % całości |
|-----------|---------------|-----------|
| Wysoki ⚡ | 6 | 15% |
| Średni 🎯 | 14 | 35% |
| Niski 🌍 | 20 | 50% |
| **RAZEM** | **40** | **100%** |

---

## Etykiety (labels)

Do utworzenia w repozytorium GitHub:

### Typ
- `enhancement` - nowa funkcjonalność
- `bug` - naprawa błędu
- `documentation` - dokumentacja
- `testing` - testy

### Platforma
- `python` - kod Python
- `android` - aplikacja Android
- `integration` - integracja platform

### Obszar
- `detection` - detekcja obiektów/znaczników
- `preprocessing` - przetwarzanie wstępne
- `segmentation` - segmentacja
- `ui` - interfejs użytkownika
- `performance` - wydajność
- `ci/cd` - continuous integration/deployment
- `infrastructure` - infrastruktura
- `backend` - backend/API
- `community` - społeczność
- `design` - design/UX
- `c++` - kod C++
- `gpu` - akceleracja GPU

### Priorytet (opcjonalnie)
- `priority: high`
- `priority: medium`
- `priority: low`

---

## QA checklist (Android modele AI)

- [ ] Zweryfikowano inicjalizację i pobieranie modeli tylko dla: **MediaPipe**, **YOLO**, **TFLite**.
- [ ] Potwierdzono brak odwołań do usuniętych silników (RTMDet, Mobilint) w UI i opisach.

---

**Użycie:**  
Ten dokument może służyć jako checklist przy tworzeniu issues w GitHub. 
Każdy wiersz odpowiada jednemu issue do utworzenia.

Zobacz pełne opisy w [DEVELOPMENT_PLAN.md](../DEVELOPMENT_PLAN.md).
