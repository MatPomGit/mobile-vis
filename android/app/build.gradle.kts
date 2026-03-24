import { useEffect } from 'react';
 
// Przypadek 1: Wykonaj RAZ przy montowaniu komponentu (inicjalizacja)
useEffect(() => {
  loadWeatherData(); // wywolaj async funkcje
}, []);  // [] = tablica zaleznosci PUSTA = efekt nie zalezy od nic = wykonaj raz
         // Odpowiednik: void initState() w Flutter
 
// Przypadek 2: Wykonaj przy KAZDEJ zmianie city
useEffect(() => {
  if (city) fetchWeather(city);
}, [city]);  // [city] = uruchom efekt gdy city sie zmieni
             // Odpowiednik: didUpdateWidget lub watch() w Compose
 
// Przypadek 3: Cleanup (sprzatanie po sobie)
useEffect(() => {
  const subscription = locationService.subscribe(onLocation);
  // Funkcja cleanup - wywolywana PRZED nastepnym efektem lub odmontowaniem:
  return () => subscription.unsubscribe();  // Odpowiednik: dispose() w Dart
}, []);
 
// Przypadek 4: Pobieranie danych - pelny wzorzec
const [weather, setWeather] = useState<WeatherData | null>(null);
const [loading, setLoading] = useState(false);
const [error, setError]     = useState<string | null>(null);
 
useEffect(() => {
  let cancelled = false;  // zapobieganie race condition
  const load = async () => {
    setLoading(true);
    try {
      const data = await weatherService.fetch(city);
      if (!cancelled) setWeather(data);
    } catch (e) {
      if (!cancelled) setError(String(e));
    } finally {
      if (!cancelled) setLoading(false);
    }
  };
  load();
  return () => { cancelled = true; };  // anuluj jezeli komponent odmontowany
}, [city]);
