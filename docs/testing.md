# Strategia testowania i walidacji

## Cel

Ten dokument opisuje, **co uruchamiamy lokalnie**, **co uruchamia CI** oraz jakie są
**progi akceptacji** dla pull requestów.

## Macierz uruchomień

| Obszar | Lokalnie (deweloper) | CI (`python-ci.yml`) | CI (`android-ci.yml`) |
|---|---|---|---|
| Lint Python | `ruff check` | `ruff check` | — |
| Typowanie Python | `mypy src` | `mypy src` | — |
| Testy Python + coverage | `pytest --cov=src/image_analysis --cov-report=term-missing` | `pytest --cov=src/image_analysis --cov-report=term-missing` | — |
| Build Android (`android/app`) | `cd android && ./gradlew :app:assembleDebug` | — | `./gradlew :app:assembleDebug` |
| Testy jednostkowe Android (`android/app`) | `cd android && ./gradlew :app:testDebugUnitTest` | — | `./gradlew :app:testDebugUnitTest` |

## Progi akceptacji PR

### Python

- `ruff check` musi zakończyć się bez błędów.
- `mypy src` musi zakończyć się bez błędów.
- `pytest --cov=src/image_analysis --cov-report=term-missing` musi przejść.
- Pokrycie testami dla `src/image_analysis` musi być **>= 80%**.

### Android

- Build modułu `android/app` (`:app:assembleDebug`) musi przejść.
- Testy jednostkowe modułu `android/app` (`:app:testDebugUnitTest`) muszą przejść.

## Wymagane status checks dla branch `main`

Dla branch protection na `main` wymagane są następujące status checks:

- `python-quality`
- `android-build-test`

Konfiguracja jest utrzymywana w pliku `.github/settings.yml` (branch protection as code).
Aby ustawienia były automatycznie stosowane, repozytorium musi mieć aktywną aplikację
**GitHub Settings** (Probot).
