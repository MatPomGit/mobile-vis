# Dokumentacja projektu

Witaj w dokumentacji projektu analizy obrazu.

## Moduły

### `preprocessing`

Moduł odpowiedzialny za wczytywanie i wstępne przetwarzanie obrazów:

- `load_image(path)` – wczytuje obraz z dysku jako tablicę BGR uint8
- `resize_image(image, width, height)` – zmienia rozmiar obrazu
- `normalize_image(image)` – normalizuje wartości pikseli do zakresu [0.0, 1.0]

### `detection`

Moduł do wykrywania obiektów na obrazach:

- `detect_objects(image, confidence_threshold)` – zwraca listę wykrytych obiektów
- `apply_nms(detections, iou_threshold)` – filtruje nakładające się bounding boxy (NMS)
- `draw_bounding_boxes(image, detections)` – rysuje bounding boxy na kopii obrazu

### `classification`

Moduł do klasyfikacji obrazów:

- `classify_image(image, model, confidence_threshold)` – zwraca etykietę i pewność predykcji
- `load_classifier(model_path)` – wczytuje model klasyfikatora
- `evaluate_classifier(predictions, ground_truth)` – oblicza metryki jakości klasyfikatora

### `utils`

Funkcje pomocnicze:

- `validate_image(image)` – waliduje czy tablica jest poprawnym obrazem
- `get_project_root()` – zwraca ścieżkę do katalogu głównego projektu
- `setup_logging(level)` – konfiguruje logger
- `safe_makedirs(directory)` – tworzy katalog (z rodzicami) jeśli nie istnieje
- `list_images(directory, extensions)` – lista plików obrazów w katalogu

## Rozszerzanie szablonu

1. Dodaj nowy moduł do `src/image_analysis/`.
2. Zaimportuj i wyeksportuj funkcje publiczne przez `__init__.py`.
3. Napisz testy w `tests/test_<modul>.py`.
4. Zaktualizuj tę dokumentację.
