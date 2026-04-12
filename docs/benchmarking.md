# Benchmark VO + Plane Detection

Dokument opisuje zestaw scen testowych i metryki jakości dla dwóch bloków:

- **VO (Visual Odometry)**,
- **detekcja i śledzenie płaszczyzn**.

## Sceny testowe

Nightly benchmark uruchamia 4 scenariusze syntetyczne:

1. `translation_motion` – dominujący ruch translacyjny,
2. `rotational_motion` – dominujący ruch obrotowy,
3. `low_texture` – scena o słabej teksturze,
4. `variable_lighting` – dynamicznie zmieniające się oświetlenie.

Konfiguracje scen znajdują się w `src/image_analysis/benchmarking.py` (funkcja
`default_benchmark_scenarios`).

Tryb **PR-lite** (dla pull requestów) uruchamia skrócony zestaw scen:

1. `translation_motion_pr_lite`,
2. `low_texture_pr_lite`.

Konfiguracja PR-lite znajduje się w tej samej lokalizacji (funkcja
`default_pr_lite_benchmark_scenarios`).

## Metryki VO

- `track_length` – średnia długość skutecznie utrzymanych torów cech,
- `inlier_ratio` – udział inlierów po estymacji geometrii,
- `drift_per_meter` – dryf trajektorii na metr,
- `reprojection_error` – średni błąd reprojekcji (px).

## Metryki płaszczyzn

- `overlap_region_iou` – IoU/overlap region wykrytych obszarów płaszczyzn,
- `normal_error_deg` – błąd kąta normalnej płaszczyzny (stopnie),
- `temporal_stability` – stabilność estymacji między klatkami.

## Baseline i alarm regresji

- Baseline zapisany jest w `benchmarks/vo_planes_baseline.json`.
- Baseline dla trybu PR-lite zapisany jest w `benchmarks/vo_planes_pr_lite_baseline.json`.
- Każdy run generuje `benchmarks/vo_planes_latest.json`.
- Nightly publikuje także trend w `benchmarks/vo_planes_trend.json` oraz `benchmark_trend.md`.
- Porównanie do baseline używa progów alarmowych zdefiniowanych w jednym miejscu:
  `DEFAULT_ALARM_THRESHOLDS` (`src/image_analysis/benchmarking.py`) i udostępnianych przez
  `default_alarm_thresholds()`.

### Progi akceptacji regresji (`DEFAULT_ALARM_THRESHOLDS`)

| Metryka | Typ progu | Wartość | Interpretacja |
| --- | --- | ---: | --- |
| `vo.track_length_mean` | `min_delta` | `-15.0` | Spadek > 15 klatek średnio = alarm |
| `vo.inlier_ratio_mean` | `min_delta` | `-0.03` | Spadek udziału inlierów > 3 pp = alarm |
| `vo.drift_per_meter_mean` | `max_delta` | `0.0025` | Wzrost dryfu > 0.0025/m = alarm |
| `vo.reprojection_error_mean` | `max_delta` | `0.2` | Wzrost błędu reprojekcji > 0.2 px = alarm |
| `planes.overlap_region_iou_mean` | `min_delta` | `-0.03` | Spadek IoU > 3 pp = alarm |
| `planes.normal_error_deg_mean` | `max_delta` | `0.5` | Wzrost błędu normalnej > 0.5° = alarm |
| `planes.temporal_stability_mean` | `min_delta` | `-0.03` | Spadek stabilności > 3 pp = alarm |

Jeżeli metryka przekroczy próg alarmowy, benchmark oznacza regresję
(`REGRESSION_STATUS=detected`).

## Uruchomienie lokalne

```bash
PYTHONPATH=src python scripts/benchmark_vo_planes.py
```

Aktualizacja baseline (po świadomej akceptacji poprawnego nowego poziomu):

```bash
PYTHONPATH=src python scripts/benchmark_vo_planes.py --write-baseline
```

Tryb ścisły (kod wyjścia != 0 przy regresjach):

```bash
PYTHONPATH=src python scripts/benchmark_vo_planes.py --strict
```

Tryb PR-lite (krótszy benchmark pod PR):

```bash
PYTHONPATH=src python scripts/benchmark_vo_planes.py --scenario-set pr-lite
```

## Aktualizacja baseline – checklista reviewerska

Baseline aktualizujemy **wyłącznie świadomie** i po przejściu checklisty:

1. Czy zmiana metryk wynika z intencjonalnej zmiany algorytmu, a nie z błędu implementacji?
2. Czy porównano wyniki Nightly i PR-lite oraz upewniono się, że trend nie pokazuje degradacji
   w metrykach krytycznych?
3. Czy opis PR zawiera uzasadnienie biznesowe/techniczne aktualizacji baseline?
4. Czy reviewer potwierdził, że regresja nie narusza założonych progów jakości produktu?
5. Czy baseline dla właściwego trybu (`vo_planes_baseline.json` lub
   `vo_planes_pr_lite_baseline.json`) został zaktualizowany w osobnym, czytelnym commicie?

**Kiedy wolno zaktualizować baseline:**
- po świadomej poprawce modelu/algorytmu, która zmienia rozkład metryk i została zaakceptowana,
- po zmianie scenariuszy benchmarku (np. reprezentatywniejsze dane syntetyczne),
- po korekcie błędu w samym benchmarku, jeśli wcześniej baseline był obciążony artefaktem pomiaru.

**Kiedy nie wolno aktualizować baseline:**
- aby „zamaskować” niezrozumiałą regresję,
- gdy nie ma analizy przyczyny odchyleń,
- gdy reviewer zgłasza wątpliwości co do stabilności wyników.

## CI (nightly)

Workflow `.github/workflows/nightly-benchmark.yml` uruchamia benchmark cyklicznie
(co noc), publikuje artefakty, trend metryk i krótkie podsumowanie w job summary.

Workflow `.github/workflows/pr-lite-benchmark.yml` uruchamia benchmark PR-lite przy zmianach:

- `src/image_analysis/benchmarking.py`,
- `src/image_analysis/planes.py`,
- `src/image_analysis/robot_perception.py`.
