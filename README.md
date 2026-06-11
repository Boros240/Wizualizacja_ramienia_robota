# Symulacja ramienia robota - Teach & Play

Projekt przedstawia symulację prostego ramienia robota w PyBullet. Program pozwala sterować końcówką robota, chwytać kostkę, nagrywać ruchy w trybie Teach & Play, odtwarzać je oraz zapisywać/wczytywać program ruchu z pliku JSON.

## Uruchomienie

```bash
python logic.py
```

Wymagane biblioteki:

- `pybullet`
- `numpy`
- `pygame`

## Sterowanie

### Suwaki PyBullet

- `Ramię Cel X/Y/Z` - pozycja celu dla końcówki ramienia, używana przez kinematykę odwrotną.
- `Start Kostki X/Y/Z` - pozycja, do której można przestawić kostkę przyciskiem `USTAW KOSTKE NA STARCIE`.

### Klawiatura

- Strzałki lewo/prawo - obrót bazy.
- Strzałki góra/dół - ruch pierwszego ramienia.
- `Z` / `X` - ruch drugiego ramienia.
- `C` / `V` - ruch przegubu.
- Spacja - włączenie/wyłączenie chwytaka.

## Teach & Play

1. Kliknij `NAGRYWAJ (START/STOP)`.
2. Steruj robotem suwakami lub klawiaturą.
3. Użyj spacji, aby złapać albo puścić kostkę.
4. Kliknij ponownie `NAGRYWAJ (START/STOP)`, aby zakończyć zapis.
5. Kliknij `ODTWORZ SEKWENCJE`.

Program zapisuje klatki zawierające:

- kąty stawów,
- stan chwytaka,
- pozycję kostki.

## Import i eksport JSON

Przyciski:

- `EKSPORTUJ JSON` - zapisuje aktualną sekwencję do pliku `teach_play_sequence.json`.
- `IMPORTUJ JSON` - wczytuje sekwencję z pliku `teach_play_sequence.json`.

Przykładowy format:

```json
{
  "version": 1,
  "format": "teach_and_play_sequence",
  "waypoint_count": 1,
  "waypoints": [
    {
      "angles": [0.0, 0.2, -0.1, 0.0],
      "gripper": false,
      "cube_pos": [0.3, 0.0, 0.2]
    }
  ]
}
```

## Tryb punktów A/B

Tryb A/B generuje automatyczny program przenoszenia kostki:

1. Ustaw kostkę w miejscu startowym.
2. Kliknij `ZAPISZ A (KOSTKA)` - punkt A zostanie pobrany z aktualnej pozycji kostki.
3. Ustaw suwakami `Ramię Cel X/Y/Z` miejsce docelowe.
4. Kliknij `ZAPISZ B (CEL)` - punkt B zostanie pobrany z aktualnego celu suwaków.
5. Kliknij `WYKONAJ A -> B`.

Robot wykona sekwencję:

1. dojazd nad punkt A,
2. zjazd do punktu A,
3. chwycenie kostki,
4. podniesienie kostki,
5. przejazd nad punkt B,
6. zjazd do punktu B,
7. puszczenie kostki,
8. odjazd w górę.

Wygenerowany program A/B jest zwykłą sekwencją Teach & Play, więc można go od razu wyeksportować do JSON i później zaimportować.

## Struktura programu

- `logic.py` - główna pętla symulacji, obsługa UI, klawiatury, chwytaka i scenariusza A/B.
- `robot_controller.py` - kontroler Teach & Play: nagrywanie, odtwarzanie, import/eksport JSON oraz generowanie programu A/B.
- `simple_arm.urdf` - model ramienia robota.
- `cube.urdf` - model kostki.

## Elementy istotne dla oceny projektu

- Wizualizacja: własny model URDF ramienia, materiały i kolory.
- Ergonomia: sterowanie suwakami, klawiaturą i przyciskami PyBullet.
- Programowe sterowanie: tryb Teach & Play, odtwarzanie sekwencji, import/eksport JSON, automatyczny program A/B.
- Fizyka i kolizje: PyBullet, bryły kolizyjne URDF, chwytanie przez constraint oraz filtrowanie kolizji podczas chwytu.
- Dodatki: obsługa dźwięku silnika, jeśli dostępny jest plik `motor.wav`.
