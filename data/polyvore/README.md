# Polyvore Outfits

## Indice

- [File](#file)
- [Dettaglio dei file: responsabilità, input e output](#dettaglio-dei-file-responsabilità-input-e-output)
- [Origine e accesso](#origine-e-accesso)
  - [Priorità delle sorgenti](#priorità-delle-sorgenti)
- [Varianti e split](#varianti-e-split)
- [Risorse del dataset](#risorse-del-dataset)
- [Identificatori e collegamenti](#identificatori-e-collegamenti)
- [Composizione di un item](#composizione-di-un-item)
- [Compatibility Prediction](#compatibility-prediction)
- [Retrieval e Fill In The Blank](#retrieval-e-fill-in-the-blank)
- [Risorse scaricate per task](#risorse-scaricate-per-task)
- [Validazione e limiti](#validazione-e-limiti)

## File

| File | Responsabilità |
|---|---|
| `__init__.py` | Espone l’API pubblica del sottopackage Polyvore. |
| `download.py` | Cerca risorse locali/cache e scarica soltanto quelle mancanti. |
| `source.py` | Implementa `DatasetSource` e nasconde dettagli Polyvore ai workflow. |
| `rows.py` | Definisce il contratto strutturale minimo delle righe item. |
| `catalog.py` | Collega ogni `item_id` a immagine, descrizione e categoria. |
| `item_dataset.py` | Restituisce singoli articoli. |
| `compatibility_dataset.py` | Interpreta outfit con label binaria. |
| `retrieval_dataset.py` | Interpreta domande FITB con positivo e distrattori. |

## Dettaglio dei file: responsabilità, input e output

Gli esempi seguenti usano valori ridotti, ma rispettano il formato delle
risorse reali.


#### Esempio di input e output

| Input | Output |
|---|---|
| Importazione di `PolyvoreCatalog` da `data.polyvore`. | La classe pubblica definita in `catalog.py`. |
| Importazione di varianti, split o dataset. | I relativi tipi senza dover conoscere il file interno. |

### `download.py`

Conosce la sorgente Hugging Face e decide quali risorse servono in base a task,
variante e split. Scarica soltanto il necessario, riusa la cache e verifica che
le righe degli item contengano almeno `item_id` e immagine e che i file richiesti
esistano.

Non interpreta label o outfit, non applica transform e non costruisce dataset o
batch. Il suo risultato è un contenitore di risorse già localizzate e
verificate, che verrà passato agli altri componenti.

#### Esempio di input e output

```text
Input
  task:       compatibility
  variant:    disjoint
  split:      train
  token:      token Hugging Face configurato

Output
  PolyvoreResources
  item_rows:          righe con item_id e immagine
  metadata_path:      polyvore_item_metadata.json
  outfits_path:       disjoint/train.json
  compatibility_path: disjoint/compatibility_train.txt
  retrieval_path:     assente, perché non serve al task scelto
```

Il risultato contiene dati e percorsi verificati, non un `Dataset` o un batch.

### `catalog.py`

È il punto di collegamento tra le diverse sorgenti che descrivono un articolo.
Indicizza le righe immagine tramite `item_id`, legge i metadata e applica la
transform scelta. Produce così un `FashionItem` uniforme, indipendentemente da
come l’immagine era rappresentata nella sorgente.

Costruisce inoltre la mappa tra token `set_id_index` e `item_id` usando i file
degli outfit. Conserva soltanto descrizione e categoria degli item presenti
nello split, evitando di mantenere in memoria tutto il metadata globale.

Non conosce label compatibility o domande retrieval: queste vengono
interpretate dai rispettivi dataset.

#### Esempio di input e output

```text
Input
  item_id: "132621870"
  riga immagine: image associata a "132621870"
  metadata: title="white shirt", semantic_category="tops"
  transform: immagine PIL → Tensor [3, 224, 224]

Output
  FashionItem
  item_id:      "132621870"
  image:        Tensor float32 [3, 224, 224]
  description:  "white shirt"
  category:     "tops"
```

Lo stesso file traduce anche un token come `199244701_1` nell’`item_id`
corrispondente usando il file degli outfit.

### `item_dataset.py`

Espone il catalogo come sequenza di singoli articoli. Riceve un indice e
restituisce il `FashionItem` corrispondente; opzionalmente può limitare la vista
a una lista specifica di `item_id`.

Serve per elaborare ogni prodotto indipendentemente dagli outfit, per esempio
durante analisi del catalogo o precomputazione degli embedding. Non usa file di
compatibility o retrieval e non crea batch.

#### Esempio di input e output

```text
Input
  catalogo con item [i1, i2, i3]
  indice richiesto: 1

Output
  FashionItem relativo a i2
```

Una sequenza opzionale di ID può limitare o riordinare gli articoli esposti dal
dataset.

### `compatibility_dataset.py`

Interpreta le annotazioni `compatibility_*.txt`. Per ogni riga legge la label,
risolve i token degli articoli attraverso la mappa degli outfit e chiede al
catalogo i relativi `FashionItem`.

Valida label, token e appartenenza degli item allo split prima del training.
Restituisce un solo `CompatibilityExample` alla volta; raggruppamento, label
floating point e batch sono responsabilità di `collate.py`.

#### Esempio di input e output

```text
Input
  annotazione: 1 199244701_1 199244701_2 199244701_3
  mappa token: 199244701_1→i1, 199244701_2→i2, 199244701_3→i3

Output
  CompatibilityExample
  example_id: "compatibility:<numero-riga>"
  outfit:     [FashionItem(i1), FashionItem(i2), FashionItem(i3)]
  label:      1
```

### `retrieval_dataset.py`

Interpreta le annotazioni Fill In The Blank. Legge l’outfit parziale, la
posizione rimossa e i candidati; individua il positivo tramite `set_id` e
`blank_position`, quindi risolve tutti i token attraverso catalogo e mappa degli
outfit.

Restituisce un `RetrievalExample` con query, positivo e negativi distinti. Non
esegue padding, non genera nuovi negativi e non elimina duplicati o falsi
negativi presenti nel benchmark ufficiale: queste decisioni appartengono al
futuro training CIR.

#### Esempio di input e output

```text
Input
  question:       [199244701_2, 199244701_3]
  blank_position: 1
  answers:        [207312192_1, 199244701_1, 224593384_1]

Output
  RetrievalExample
  partial_outfit: item delle posizioni 2 e 3
  positive_item:  item indicato da 199244701_1
  negative_items: item indicati dagli altri due token
```

Il positivo viene riconosciuto tramite `set_id` e `blank_position`, non tramite
la posizione occupata nell’elenco `answers`.

## Origine e accesso

La sorgente usata è il dataset Hugging Face
[`mvasil/polyvore-outfits`](https://huggingface.co/datasets/mvasil/polyvore-outfits).
Le righe Hugging Face forniscono immagini e `item_id`; file separati forniscono
struttura degli outfit, testo e annotazioni dei task.

Il repository è gated: occorre accettare le condizioni indicate nella dataset
card e configurare un token Hugging Face.

### Priorità delle sorgenti

Ogni risorsa viene cercata in quest'ordine:

1. cartella `datasets/polyvore-outfits/`, o percorso `--dataset-root`;
2. cache Hugging Face locale, predefinita o scelta con `--cache-dir`;
3. repository Hugging Face, soltanto quando la risorsa manca nelle prime due.

La cartella locale può contenere anche solo parte del dataset. I parquet
seguono `data/<variant>/<split>.parquet`; metadata e annotazioni mantengono i
percorsi del repository `mvasil/polyvore-outfits`.

Per scaricare entrambe le varianti e tutti gli split nella cartella locale:

```powershell
python -m scripts.download_polyvore
```

## Varianti e split

| Variante | Separazione |
|---|---|
| `nondisjoint` | Gli outfit sono separati, ma lo stesso item può comparire in split differenti. |
| `disjoint` | Gli item di training, validation e test non si sovrappongono. |

La variante disjoint misura meglio la generalizzazione verso articoli mai visti.
La scelta non modifica il formato degli esempi o dei batch.

Entrambe le varianti prevedono gli split `train`, `validation` e `test`. Nei
nomi dei file originali, lo split `validation` usa l’abbreviazione `valid`.
Training serve per ottimizzare, validation per scegliere configurazioni e
checkpoint, test per la valutazione finale.

## Risorse del dataset

| Risorsa | Contenuto | Collegamento principale |
|---|---|---|
| Parquet esposti come righe Hugging Face | Una coppia `item_id` e immagine per articolo. | `item_id` |
| `<variant>/<split>.json` | Outfit con `set_id`, posizione e `item_id` degli articoli. | `set_id` + `index` |
| `polyvore_item_metadata.json` | Titolo, descrizione, nome URL e categoria degli articoli. | `item_id` |
| `compatibility_<split>.txt` | Outfit compatibili o incompatibili con label `1/0`. | token `set_id_index` |
| `fill_in_blank_<split>.json` | Outfit incompleti e possibili completamenti. | token `set_id_index` |

Le immagini non si trovano nel file dei metadata. Allo stesso modo, i token
nelle annotazioni non contengono né pixel né testo: sono riferimenti da
risolvere attraverso la struttura degli outfit.

## Identificatori e collegamenti

Polyvore usa tre identificatori distinti:

| Identificatore | Significato |
|---|---|
| `set_id` | Identifica un outfit. |
| `index` | Indica la posizione originale di un articolo nell’outfit. |
| `item_id` | Identifica il prodotto reale e collega immagine e metadata. |

Un outfit nel file dello split ha concettualmente questa forma:

```json
{
  "set_id": "199244701",
  "items": [
    {"index": 1, "item_id": "132621870"},
    {"index": 2, "item_id": "153967122"}
  ]
}
```

Le annotazioni usano un token ottenuto unendo `set_id` e `index`. Per esempio,
`199244701_1` viene risolto nell’`item_id` `132621870`. A quel punto il catalogo
recupera l’immagine dalle righe Hugging Face e il testo dai metadata.

Il token identifica una posizione in uno specifico outfit; `item_id` identifica
invece l’articolo reale. Token diversi possono quindi riferirsi allo stesso
prodotto.

## Composizione di un item

Il catalogo costruisce ogni articolo combinando tre sorgenti:

| Campo prodotto | Sorgente |
|---|---|
| Immagine | Campo `image` della riga Hugging Face, poi conversione RGB e transform. |
| Descrizione | `title` o `url_name`, eventualmente uniti a `description`. |
| Categoria | `semantic_category`; in assenza, `category_id`. |

Se il testo manca completamente viene usata la descrizione `fashion item`; se
manca la categoria viene usato `unknown`. Del metadata globale vengono
conservati soltanto descrizione e categoria degli item presenti nello split,
riducendo l’uso di memoria.

Le immagini possono essere già decodificate, rappresentate come byte o riferite
da un percorso locale fornito da Hugging Face. Non vengono scaricate da URL
esterni durante l’iterazione del dataset.

## Compatibility Prediction

Ogni riga `compatibility_*.txt` contiene prima la label e poi i token degli
articoli:

```text
1 199244701_1 199244701_2 199244701_3
0 219713029_1 223118810_1 224078562_3
```

- `1` rappresenta un outfit compatibile;
- `0` rappresenta una combinazione negativa già preparata dal benchmark.

Il dataset traduce ogni token in `item_id`, recupera i relativi articoli dal
catalogo e restituisce un outfit di lunghezza variabile con la label. La collate
mantiene gli ID degli item e produce label floating point con forma `[batch, 1]`,
compatibile con l’output del modello CP e con la sua loss.

## Retrieval e Fill In The Blank

Le annotazioni `fill_in_blank_*.json` descrivono un outfit al quale è stato
rimosso un articolo:

```json
{
  "question": ["199244701_2", "199244701_3"],
  "blank_position": 1,
  "answers": ["207312192_1", "199244701_1", "224593384_1"]
}
```

`question` contiene l’outfit parziale, `blank_position` indica la posizione
rimossa e `answers` contiene il completamento corretto insieme ai distrattori.
La risposta corretta non è necessariamente la prima: è il token con lo stesso
`set_id` della domanda e con indice uguale a `blank_position`.

Il dataset restituisce separatamente outfit parziale, item positivo e item
negativi. Conserva inoltre ID e categoria del target, informazioni utili per un
futuro task CIR.

I candidati ufficiali vengono preservati senza deduplicazione. Alcune domande
reali contengono token differenti associati allo stesso `item_id`, incluso
qualche falso negativo. Un’eventuale politica di pulizia deve essere decisa dal
training CIR, non applicata silenziosamente dal dataset.

## Risorse scaricate per task

| Task | Risorse richieste |
|---|---|
| Item | Righe immagine dello split e metadata globali. |
| Compatibility | Risorse item, struttura degli outfit e file compatibility. |
| Retrieval | Risorse item, struttura degli outfit e file fill-in-the-blank. |

Questa selezione evita di caricare o scaricare annotazioni non necessarie.
Risoluzione e verifica avvengono prima della creazione del dataset; l’accesso a un singolo
esempio non effettua operazioni di rete.

## Validazione e limiti

Il caricamento controlla che immagini e `item_id` siano presenti, che i file
richiesti esistano, che i token siano risolvibili e che gli articoli appartengano
allo split selezionato. Label, domande FITB e posizione del positivo vengono
validate prima del training, così gli errori di struttura emergono subito.

Il catalogo descrive i dati grezzi e trasformati, ma non salva embedding. Il job
`scripts/precompute_embeddings.py` crea la cache FashionCLIP fuori dal dataset.
Campionamento di nuovi negativi e politiche di bilanciamento restano nei
training loop dei task.
