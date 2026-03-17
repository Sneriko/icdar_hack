# Svenska lejonet

Detta repo innehåller kod för att träna det svenska lejonet.

## Struktur

Alla hyperparametrar finns i `params.py`.

## Data

Träningen utgår från GT i form av Page XML och tillhörande bilder.
De laddas från filsystemet (lokalt eller från NFS), klipps till rader och sparas i en LMDB.
Om man utgår från GT på NFS:en tar det runt en timme att skapa LMDB:n (jag har inte testat från LakeFS än).

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

All kod för datahantering finns i `data.py`.
