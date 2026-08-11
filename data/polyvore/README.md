# Polyvore Outfits

## Indice

- [File](#file)
- [Origine e accesso](#origine-e-accesso)
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
| `download.py` | Scarica e verifica le sole risorse richieste dal task. |
| `catalog.py` | Collega ogni `item_id` a immagine, descrizione e categoria. |
| `item_dataset.py` | Restituisce singoli articoli. |
| `compatibility_dataset.py` | Interpreta outfit con label binaria. |
| `retrieval_dataset.py` | Interpreta domande FITB con positivo e distrattori. |

## Origine e accesso

La sorgente usata è il dataset Hugging Face
[`mvasil/polyvore-outfits`](https://huggingface.co/datasets/mvasil/polyvore-outfits).
Le righe Hugging Face forniscono immagini e `item_id`; file separati forniscono
struttura degli outfit, testo e annotazioni dei task.

Il repository è gated: occorre accettare le condizioni indicate nella dataset
card e configurare un token Hugging Face. I file vengono conservati nella cache
Hugging Face, oppure nella directory di cache scelta dall’utente; non vengono
copiati automaticamente nel repository del progetto.

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

Questa selezione evita di scaricare annotazioni non necessarie. Download e
verifica avvengono prima della creazione del dataset; l’accesso a un singolo
esempio non effettua operazioni di rete.

## Validazione e limiti

Il caricamento controlla che immagini e `item_id` siano presenti, che i file
richiesti esistano, che i token siano risolvibili e che gli articoli appartengano
allo split selezionato. Label, domande FITB e posizione del positivo vengono
validate prima del training, così gli errori di struttura emergono subito.

Il catalogo descrive i dati grezzi e trasformati, ma non salva embedding. Le
cache di rappresentazioni, il campionamento di nuovi negativi e le politiche di
bilanciamento appartengono rispettivamente alla precomputazione e ai training
loop dei task.
