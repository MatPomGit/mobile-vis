# Roadmap rozwoju projektu MobileVis

> **Ostatnia aktualizacja:** 2026-03-23  
> **Status:** Planowanie  
> **Pełny plan:** Zobacz [DEVELOPMENT_PLAN.md](../DEVELOPMENT_PLAN.md)

---

## 🎯 Wizja projektu

MobileVis ma stać się kompleksową platformą do analizy obrazu łączącą:
- Zaawansowane algorytmy przetwarzania obrazu w Pythonie
- Wydajną aplikację mobilną na Androida z real-time processing
- Łatwą integrację i rozszerzalność dla badaczy i developerów

---

## 📊 Podsumowanie planu

### Statystyki
- **Łączna liczba zaplanowanych issues:** 40
- **Priorytet wysoki:** 6 issues
- **Priorytet średni:** 14 issues
- **Priorytet niski:** 20 issues

### Kategorie rozwoju

| Kategoria | Liczba issues | Przykładowe zagadnienia |
|-----------|---------------|-------------------------|
| 🐍 Funkcjonalności Python | 7 | YOLO detection, segmentacja, face detection |
| 📱 Funkcjonalności Android | 7 | AprilTag/QR w Android, zapisywanie zdjęć, FPS counter |
| 🔗 Integracja | 3 | Wspólny rdzeń C++, REST API, klient API |
| 📚 Dokumentacja | 4 | Jupyter notebooks, Sphinx docs, video tutorial |
| ✅ Testy i jakość | 4 | 90%+ coverage, benchmarks, Espresso tests |
| 🚀 CI/CD | 5 | GitHub Actions, semantic release, Docker, PyPI |
| ⚡ Wydajność | 4 | GPU acceleration, optymalizacja pamięci |
| 🎨 Usprawnienia | 6 | i18n, CLI, logo, CONTRIBUTING.md |

---

## 🗓️ Plan realizacji

### Faza 1: Podstawy (Sprint 1-2) ⚡ PRIORYTET
**Cel:** Stabilna infrastruktura i proces developerski

- [x] Dokumentacja planu rozwoju
- [ ] #26: GitHub Actions workflow dla Python (CI/CD)
- [ ] #27: GitHub Actions workflow dla Android (CI/CD)
- [ ] #22: Zwiększenie pokrycia testów do 90%+
- [ ] #40: Contributing guidelines (CONTRIBUTING.md)

**Rezultat:** Automatyczne testy, linting, stabilny proces review

---

### Faza 2: Kluczowe funkcjonalności (Sprint 3-5) 🎯
**Cel:** Rozszerzenie możliwości detekcji i dokumentacji

- [ ] #1: Detekcja obiektów YOLO (Python)
- [ ] #8: Filtry AprilTag i QR w aplikacji Android
- [ ] #18: Przykłady Jupyter Notebooks dla wszystkich modułów
- [ ] #20: Generowanie dokumentacji API (Sphinx)

**Rezultat:** YOLO w Python, live detection w Android, pełna dokumentacja

---

### Faza 3: Rozszerzenia (Sprint 6-8) 🚀
**Cel:** Nowe algorytmy i integracja

- [ ] #4: Wykrywanie twarzy i punktów charakterystycznych
- [ ] #9: Zapisywanie zdjęć z filtrami (Android)
- [ ] #16: REST API dla przetwarzania obrazów
- [ ] #23: Testy wydajnościowe (benchmarks)

**Rezultat:** Face detection, API backend, metryki wydajności

---

### Faza 4: Optymalizacja (Sprint 9-10) ⚡
**Cel:** Zwiększenie wydajności i polish

- [ ] #31: Optymalizacja przetwarzania (NumPy vectorization)
- [ ] #33: Optymalizacja pamięci w Android
- [ ] #28: Automatyczne releases (semantic versioning)
- [ ] #30: Publikacja pakietu na PyPI

**Rezultat:** Szybsze przetwarzanie, mniejsze zużycie RAM, łatwe wdrożenia

---

### Faza 5: Community & Long-term (ciągłe) 🌍
**Cel:** Budowanie społeczności i ekosystemu

- [ ] #35: Wsparcie dla wielu języków (i18n)
- [ ] #37: Logo i ikona aplikacji
- [ ] #39: GitHub Discussions
- [ ] Pozostałe issues według feedback użytkowników

**Rezultat:** Aktywna społeczność, profesjonalny branding

---

## 🔥 Top 10 priorytetowych issues

1. **#26** - GitHub Actions dla Python (CI/CD) ⚡
2. **#27** - GitHub Actions dla Android (CI/CD) ⚡
3. **#22** - Pokrycie testów 90%+ ⚡
4. **#1** - YOLO Object Detection 🎯
5. **#8** - AprilTag/QR detection w Android 🎯
6. **#18** - Jupyter Notebooks examples 📚
7. **#20** - Sphinx API Documentation 📚
8. **#40** - CONTRIBUTING.md 🤝
9. **#16** - REST API Backend 🔗
10. **#23** - Performance Benchmarks ⚡

---

## 📈 Metryki sukcesu

### Infrastruktura
- ✅ CI/CD pipeline dla Python i Android
- ✅ Automatyczne testy przy każdym PR
- ✅ Coverage > 90% dla kodu Python
- ✅ Dokumentacja API dostępna online

### Funkcjonalności
- ✅ YOLO detection w Python
- ✅ AprilTag + QR real-time w Android
- ✅ REST API dla image processing
- ✅ Face detection & landmarks

### Community
- ✅ Pakiet na PyPI (łatwa instalacja: `pip install mobile-vis`)
- ✅ > 10 przykładów w Jupyter Notebooks
- ✅ CONTRIBUTING.md i aktywne Discussions
- ✅ > 100 GitHub stars

---

## 🤝 Jak pomóc?

1. **Zgłoś issue** - znajdź bug lub zaproponuj funkcjonalność
2. **Kod** - wybierz issue z listy i prześlij PR
3. **Dokumentacja** - popraw docs, dodaj przykłady
4. **Testowanie** - przetestuj na różnych urządzeniach
5. **Feedback** - podziel się doświadczeniem użytkowania

Zobacz szczegóły w `CONTRIBUTING.md` (planowany w issue #40)

---

## 📞 Kontakt

- **Issues:** [GitHub Issues](https://github.com/MatPomGit/mobile-vis/issues)
- **Dyskusje:** [GitHub Discussions](https://github.com/MatPomGit/mobile-vis/discussions) (wkrótce)
- **Email:** (do uzupełnienia)

---

## 📝 Historia zmian roadmapy

| Data | Zmiana | Autor |
|------|--------|-------|
| 2026-03-23 | Utworzenie roadmap i planu 40 issues | GitHub Copilot |

---

**Legenda:**
- ⚡ Wysoki priorytet
- 🎯 Średni priorytet
- 🌍 Niski priorytet / Long-term
- [x] Zrealizowane
- [ ] Do realizacji
