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
- Każdy run generuje `benchmarks/vo_planes_latest.json`.
- Porównanie do baseline używa progów alarmowych z `default_alarm_thresholds()`:
  - metryki „im więcej tym lepiej” używają `min_delta`,
  - metryki „im mniej tym lepiej” używają `max_delta`.

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

## CI (nightly)

Workflow `.github/workflows/nightly-benchmark.yml` uruchamia benchmark cyklicznie
(co noc) i publikuje artefakty + podsumowanie metryk.
