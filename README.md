# Svenska lejonet

Detta repo innehåller kod för att träna det svenska lejonet.

## Instruktioner

Klona repot och installera beroenden:
```
git clone https://devops.ra.se/DataLab/Datalab/_git/swedish-lion
cd swedish-lion
uv sync
```

Alla träningsinställningar finns i `params.py`.
Uppdatera den om det behövs och checka in eventuella ändringar.
Förbered datan med:
```
python3 data.py
```
Det tar en stund (~1h) att skapa träningssetet första gången man kör skriptet.

Starta sedan träningen:
```
python3 train.py
```

Träningen kommer att loggas till Mlflow under experimentet [`swedish-lion`](https://mlflow.ra.se/#/experiments/28).


## Repots struktur

- `augment.py`: augmentering av träningsdatan
- `data.py`: kod för att skapa LMDB:n utifrån GT
- `generate_splits.py`: skriptet som användes för att generera de initiala splitsen
- `gt.py`: lite hjälpfunktioner för att konvertera GT i PageXML till rader
- `params.py`: alla hyperparametrar
- `tracking.py`: uppsättning av MLflow + kontroll att repot är rent innan körning
- `train.py`: själva träningsskriptet


## Data

Träningen utgår från GT i form av Page XML och tillhörande bilder.
Variabeln `DATA_PATH` i `params.py` innehåller sökvägen till datan.
Skriptet letar rekursivt under `DATA_PATH` efter kataloger med följande struktur:
```
dir
├── image_0.jpg
├── image_1.jpg
├── image_2.jpg
├── ...
├── image_N.jpg
└── page
    ├── image_0.xml
    ├── image_1.xml
    ├── image_2.xml
    ├── ...
    ├── image_N.xml
    └── $TEST_SPLIT
```
Filen `$TEST_SPLIT` innehåller en lista på vilka sidor som ingår i testsetet.
Det är alltid minst en per katalog.
Själva namnet på splitfilen ges i `params.py`, just nu är den satt till `test0050` (dvs en split på 5%).
Dessa filer är genererade med `generate_splits.py`, men tanken är att de inte ska omgenereras vid ny körning.

Varje matchande bild- och PageXML-par klipps till rader och sparas i en LMDB.
Träningsexemplena skrivs som (bild, text)-tuplar till LMDBn:

|key|value|
|---|----|
|page_0_line_0_key|(image, text)|
|page_0_line_1_key|(image, text)|
|page_0_line_2_key|(image, text)|
|page_1_line_0_key|(image,text)|
|...|...|
|page_*n*_line\_*m*_key|(image,text)|

LMDB:n innehåller också ett index över nycklarna (page_0_line_0_key osv).
Eftersom vi vill kunna splitta datasetet på radnivå behöver vi hålla koll på vilken sida en viss rad tillhör.
Därför finns det nycklar både för sidor och för rader.
Sidnycklarna finns under `__keys__`, och radnycklarna finns under respektive sidnyckel:

|key|value|
|---|-----|
|`__keys__`|[page_0_key, ..., page_*n*_key]
|page_0_key|[page_0_line_0_key, ..., page_0_line_*m*\_key]
|page_*n*_key|[page_*n*\_line_0_key, ..., page_*n*\_line_*m*\_key]

### Augmentering

Träningsexemplena augmenteras under träning enligt `augment.py`.
