# Wkład w projekt (CONTRIBUTING)

Dziękujemy za chęć współtworzenia projektu `mobile-vis`.
Poniżej znajduje się minimalny workflow wymagany przed wysłaniem Pull Requesta.

## 1. Nazewnictwo branchy

Używaj krótkich, opisowych nazw:

- `feature/<opis>` – nowa funkcjonalność,
- `fix/<opis>` – poprawka błędu,
- `refactor/<opis>` – refaktoryzacja bez zmiany zachowania,
- `docs/<opis>` – zmiany dokumentacji.

Przykłady:

- `feature/add-check-script`
- `fix/odometry-null-pointer`
- `docs/update-pr-template`

## 2. Konwencja commitów

Obowiązuje Conventional Commits (treść commit message po angielsku):

- `feat(preprocessing): add histogram equalization`
- `fix(detection): correct NMS threshold application`
- `docs(readme): add known environment issues`
- `chore(ci): update lint script`

## 3. Lokalne checki przed PR

Uruchom zestaw walidacyjny:

```bash
./scripts/check.sh
```

Skrypt wykonuje standardowe checki projektu:

1. `ruff check src/ tests/`
2. `mypy src/`
3. `pytest --cov=src/image_analysis --cov-report=term-missing`

## 4. Zasady Pull Request

Każdy PR powinien:

- zawierać **opis po polsku** (cel, zakres i ryzyko zmian),
- mieć uruchomione i zaliczone lokalne checki (`ruff`, `mypy`, `pytest + coverage`),
- zawierać testy dla nowego kodu lub uzasadnienie, dlaczego testy nie były potrzebne,
- opisywać wpływ zmian na:
  - część **Python** (`src/image_analysis`),
  - część **Android** (`android/`) – jeśli dotyczy.

Przed utworzeniem PR upewnij się, że branch jest aktualny względem `main`.
