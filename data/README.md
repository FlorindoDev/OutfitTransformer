# Dati e batching

Il package `data` legge gli outfit, applica il preprocessing delle immagini e
costruisce batch con padding e maschere.

Torna al [README principale](../README.md) oppure consulta
l'[architettura comune](../model/common/README.md).

## Indice

- [Cosa fanno `transforms.py` e `batch.py`](#cosa-fanno-transformspy-e-batchpy)
- [Dataset fornito: Polyvore Outfits](#dataset-fornito-polyvore-outfits)
  - [Cosa contiene](#cosa-contiene)
  - [Esempio reale: un outfit e i suoi item](#esempio-reale-un-outfit-e-i-suoi-item)
  - [Da dove vengono presi immagini e testo](#da-dove-vengono-presi-immagini-e-testo)
  - [Esempio reale: una domanda positiva di compatibility](#esempio-reale-una-domanda-positiva-di-compatibility)
  - [Esempio esplicativo: una domanda negativa](#esempio-esplicativo-una-domanda-negativa)
  - [Varianti del dataset](#varianti-del-dataset)
  - [Uso rispettivo degli split](#uso-rispettivo-degli-split)
- [Come prepariamo Polyvore per Compatibility Prediction](#come-prepariamo-polyvore-per-compatibility-prediction)
  - [Forma di un singolo esempio](#forma-di-un-singolo-esempio)
  - [Forma di un batch](#forma-di-un-batch)
  - [Dal batch agli input del modello](#dal-batch-agli-input-del-modello)
- [Caricamento da Hugging Face](#caricamento-da-hugging-face)
- [Loader generico alternativo](#loader-generico-alternativo)
- [DataLoader](#dataloader)
- [Padding mask](#padding-mask)
  - [Uso prima del Transformer](#uso-prima-del-transformer)
  - [Uso nel Transformer](#uso-nel-transformer)
- [File](#file)

## Cosa fanno `transforms.py` e `batch.py`

`transforms.py` definisce il preprocessing comune delle immagini. La funzione
`build_image_transform()` costruisce una trasformazione che:

1. ridimensiona ogni immagine a `224 × 224`;
2. la converte da immagine PIL a tensore PyTorch `[3,224,224]`;
3. la normalizza con media e deviazione standard ImageNet.

In questo modo i loader diversi producono immagini con la stessa forma e la
stessa scala numerica, pronte per l'encoder visuale del modello.

`batch.py` definisce le strutture dati usate tra dataset, `DataLoader` e
modello:

| Oggetto | Ruolo |
|---|---|
| `OutfitExample` | Un singolo outfit: ID, immagini degli item e descrizioni testuali |
| `OutfitBatch` | Un batch di outfit già padded, con immagini, testi, ID e maschera |
| `collate_outfits()` | Funzione passata al `DataLoader` per unire più outfit in un batch rettangolare |

Il punto chiave è che gli outfit non hanno tutti lo stesso numero di capi.
`collate_outfits()` trova l'outfit più lungo del batch, aggiunge immagini di
padding a zero agli outfit più corti e crea `padding_mask`:

- `False` indica una posizione reale;
- `True` indica una posizione di padding da ignorare.

Questa maschera viene poi usata dal Transformer del modello come
`src_key_padding_mask`, così l'attenzione considera solo gli item reali e non
impara informazioni artificiali dalle posizioni vuote.

## Dataset fornito: Polyvore Outfits

Il dataset usato è
[`mvasil/polyvore-outfits`](https://huggingface.co/datasets/mvasil/polyvore-outfits).
Contiene outfit e item di moda. Ogni item ha un `item_id`: quello è la chiave
che collega immagine, testo e outfit.

Il repository Hugging Face è gated: prima del download bisogna accettare le
condizioni di accesso indicate nella dataset card e autenticarsi.

### La struttura rilevante del repository è:

```text
polyvore-outfits/
  data/
    nondisjoint/
      train.parquet
      validation.parquet
      test.parquet
    disjoint/
      train.parquet
      validation.parquet
      test.parquet
  nondisjoint/
    compatibility_train.txt
    compatibility_valid.txt
    compatibility_test.txt
    train.json
    valid.json
    test.json
    fill_in_blank_*.json
  disjoint/
    compatibility_train.txt
    compatibility_valid.txt
    compatibility_test.txt
    train.json
    valid.json
    test.json
    fill_in_blank_*.json
  polyvore_item_metadata.json
```

I Parquet del repack Hugging Face contengono soprattutto righe `item_id` +
`image`: sono la sorgente delle immagini. I file `train.json`, `valid.json` e
`test.json` contengono la struttura degli outfit e permettono di tradurre i
token `set_id_index` nei relativi `item_id`. `polyvore_item_metadata.json`
fornisce descrizioni e categorie. I JSON `fill_in_blank_*.json` appartengono
al task FITB e non vengono usati nel training CP implementato qui.

### Cosa contiene

I file che usiamo sono questi:

| File | Cosa contiene | Chiave usata |
|---|---|---|
| `data/<variant>/<split>.parquet` | Immagini degli item | `item_id` |
| `<variant>/<split>.json` | Outfit: quali item stanno nello stesso outfit | `set_id` + `index` |
| `<variant>/compatibility_<split>.txt` | Domande CP e label `0/1` | token `set_id_index` |
| `polyvore_item_metadata.json` | Testo dell'item: descrizione, titolo, nome, categoria | `item_id` |

Quindi: il dataset è collegato tramite `item_id`. Con lo stesso `item_id` il
loader prende l'immagine dal Parquet e il testo da
`polyvore_item_metadata.json`.

Attenzione: `polyvore_item_metadata.json` sta nella radice del dataset, ma non
contiene i pixel delle immagini. Contiene il testo e le categorie. Le immagini
stanno nei Parquet.

### Esempio reale: un outfit e i suoi item

Questo è un outfit. `set_id` identifica l'outfit; ogni elemento in `items`
collega una posizione dell'outfit (`index`) a un prodotto reale (`item_id`):

```json
{
  "set_id": "199244701",
  "items": [
    {"index": 1, "item_id": "132621870"},
    {"index": 2, "item_id": "153967122"},
    {"index": 3, "item_id": "171169800"},
    {"index": 4, "item_id": "162799044"},
    {"index": 5, "item_id": "172538912"},
    {"index": 6, "item_id": "172312529"}
  ]
}
```

Qui ci sono tre identificatori diversi, che non vanno confusi:

| Valore | Che cosa identifica |
|---|---|
| `199244701` | L'intero outfit (`set_id`) |
| `1` | La posizione del primo item dentro quell'outfit (`index`) |
| `132621870` | Il prodotto reale in quella posizione (`item_id`) |

Il token usato dai file di compatibility si costruisce con `set_id` e
`index`, non con `item_id`:

```text
199244701_1
set_id    index
```

Il loader lo traduce così:

```text
token 199244701_1
  -> nel file train.json trova item_id 132621870
  -> nel Parquet trova l'immagine di 132621870
  -> in polyvore_item_metadata.json trova il testo di 132621870
```

Il token `199244701_1` non contiene né immagine né descrizione. È solo un
indirizzo: dice quale item cercare.

### Da dove vengono presi immagini e testo

Nel caricamento Hugging Face standard, le immagini vengono dai Parquet:

```text
data/nondisjoint/train.parquet
  item_id = 132621870
  image   = immagine del prodotto
```

Il testo viene invece dal file metadata alla radice:

```text
polyvore_item_metadata.json
  "132621870": {
    "description": "...",
    "title": "...",
    "url_name": "...",
    "semantic_category": "..."
  }
```

Quindi, per un item, il loader fa sempre lo stesso ragionamento:

```text
item_id 132621870
  -> immagine: data/nondisjoint/train.parquet
  -> descrizione: polyvore_item_metadata.json
```

Quindi durante il training non viene fatto download live delle immagini da
Polyvore o da URL esterni. Se nel campo `image` ci fosse solo una stringa
`http://...` o `https://...`, il loader non la usa come sorgente diretta:
si aspetta un'immagine incorporata nel dataset oppure un file locale.

Il loader supporta anche dataset locali. In quel caso l'immagine può arrivare
da un file chiamato con l'`item_id`:

```text
image_root = data/polyvore_images
item_id    = 132621870

data/polyvore_images/132621870.jpg
```

oppure da un path relativo scritto nell'item:

```json
{
  "set_id": "199244701",
  "items": [
    {
      "index": 1,
      "item_id": "132621870",
      "image": "199244701/132621870.jpg"
    }
  ]
}
```

Con `image_root="data/images"`, il loader apre:

```text
data/images/199244701/132621870.jpg
```

### Esempio reale: una domanda positiva di compatibility

Per lo stesso outfit, la
[preview pubblica delle domande di compatibility](https://huggingface.co/datasets/owj0421/polyvore-outfits)
mostra questa domanda positiva del training set disjoint:

```text
1 199244701_1 199244701_2 199244701_3 199244701_4 199244701_5 199244701_6
```

La riga significa:

```text
label = 1
numero di item = 6
set_id di tutti i token = 199244701
```

Il loader traduce i sei token nei sei `item_id` del JSON precedente, carica
sei immagini e sei descrizioni, quindi restituisce:

```text
CompatibilityExample(
    outfit_id="compatibility_train:<numero-riga>",
    images=Tensor[6, 3, 224, 224],
    descriptions=tuple[str, str, str, str, str, str],
    label=1.0,
)
```

### Esempio esplicativo: una domanda negativa

La riga seguente è illustrativa e serve a mostrare la relazione tra item
provenienti da outfit diversi:

```text
0 199244701_1 200742384_2 206955877_3
```

Il loader la interpreta così:

| Token | Outfit sorgente | Posizione |
|---|---|---:|
| `199244701_1` | `199244701` | 1 |
| `200742384_2` | `200742384` | 2 |
| `206955877_3` | `206955877` | 3 |

La label `0` dice che la combinazione è incompatibile. Il loader non decide
da solo che questi item stanno male insieme: durante il training usa
esclusivamente le righe negative già presenti in
`compatibility_train.txt`.

### Varianti del dataset

| Variante | Separazione | Quando usarla |
|---|---|---|
| `nondisjoint` | Gli outfit non si sovrappongono tra gli split, ma lo stesso item può comparire in split diversi | Training più semplice e confronto con il benchmark PO |
| `disjoint` | Nessun item del training compare in validation o test | Valutazione più difficile e più realistica della generalizzazione |

La scelta della variante non cambia la forma dei tensori: cambia quali outfit
e item sono presenti negli split.

### Uso rispettivo degli split

| Split del loader | File di label | Uso nel progetto |
|---|---|---|
| `train` | `compatibility_train.txt` | Batch mescolati, Focal Loss, backward e aggiornamento dei pesi |
| `validation` | `compatibility_valid.txt` | Valutazione senza aggiornare i pesi e scelta del checkpoint migliore |
| `test` | `compatibility_test.txt` | Valutazione finale su dati mai usati per ottimizzare il modello |

`train_cp.py` usa automaticamente train e validation. Il test è esposto dal
loader, ma deve essere eseguito separatamente dopo avere scelto il checkpoint:
non viene usato durante il training.

## Come prepariamo Polyvore per Compatibility Prediction

Il task CP risponde a una domanda semplice: questi item stanno bene insieme?
La risposta è nel primo numero della riga:

```text
1 199244701_1 199244701_2 199244701_3
0 199244701_1 200742384_2 206955877_3
```

- `1`: outfit compatibile;
- `0`: outfit incompatibile.

I valori dopo la label sono token `set_id_index`. Sono solo riferimenti, non
contengono né immagini né descrizioni.

Per ogni token il loader fa questi passaggi:

1. legge `199244701_1` da `compatibility_train.txt`;
2. separa `set_id = 199244701` e `index = 1`;
3. cerca in `nondisjoint/train.json` quale `item_id` sta in quella posizione;
4. ottiene per esempio `item_id = 132621870`;
5. prende l'immagine di `132621870` dal Parquet;
6. prende il testo di `132621870` da `polyvore_item_metadata.json`;
7. applica resize, tensor e normalizzazione all'immagine;
8. restituisce un `CompatibilityExample`.

```text
compatibility_train.txt  -> 199244701_1
train.json               -> 199244701 + 1 = item_id 132621870
train.parquet            -> item_id 132621870 = immagine
polyvore_item_metadata.json -> item_id 132621870 = descrizione
```

Per il testo, il loader prova questi campi: `description`, `text`, `title`,
`name`, `url_name`, `semantic_category`. Se non trova nulla, usa
`"fashion item"`.

### Forma di un singolo esempio

Se la riga di compatibility contiene `N` item:

| Campo | Tipo/forma | Contenuto |
|---|---|---|
| `outfit_id` | `str` | ID tecnico della domanda, per esempio `compatibility_train:42` |
| `images` | `[N,3,224,224]`, `float32` | Un'immagine normalizzata per item |
| `descriptions` | tupla di `N` stringhe | Una descrizione per ciascuna immagine |
| `label` | scalare `float` | `1.0` compatibile oppure `0.0` incompatibile |

Le dimensioni significano:

```text
N   numero di item nell'outfit
3   canali RGB
224 altezza dell'immagine
224 larghezza dell'immagine
```

Esempio già risolto dal loader:

```text
compatibility_train.txt:
1 199244701_1 199244701_2 199244701_3

risoluzione dei token:
199244701_1 -> item_id 132621870
199244701_2 -> item_id 153967122
199244701_3 -> item_id 171169800

output dell'esempio:
images.shape = [3, 3, 224, 224]
descriptions = (
    "<testo letto da polyvore_item_metadata.json per 132621870>",
    "<testo letto da polyvore_item_metadata.json per 153967122>",
    "<testo letto da polyvore_item_metadata.json per 171169800>",
)
label = 1.0
```

### Forma di un batch

Gli outfit hanno lunghezze differenti. Dato un batch di `B` outfit, `L` è il
numero di item dell'outfit più lungo nel batch:

| Campo | Forma | Contenuto |
|---|---|---|
| `images` | `[B,L,3,224,224]`, `float32` | Immagini reali più eventuali immagini di padding a zero |
| `descriptions` | `B` tuple di lunghezza variabile | Solo i testi degli item reali |
| `padding_mask` | `[B,L]`, `bool` | `False` per item reali, `True` per padding |
| `labels` | `[B]`, `float32` | Label binaria di ciascun outfit |
| `outfit_ids` | tupla di `B` stringhe | Identificativi delle domande |

Esempio con due outfit da tre e due item:

```text
images.shape       = [2, 3, 3, 224, 224]
labels.shape       = [2]
labels             = [1.0, 0.0]
padding_mask       = [
    [False, False, False],
    [False, False, True ],
]
```

### Dal batch agli input del modello

Il dataset non contiene embedding già pronti. Vengono calcolati durante il
forward:

```text
images                         [B,L,3,224,224]
  → ResNet-18                  [B,L,64]
descriptions
  → SentenceBERT + FC          [B,L,64]
concatenazione                 [B,L,128]
aggiunta token OUTFIT          [B,L+1,128]
Transformer                   [B,L+1,128]
output token OUTFIT            [B,128]
classifier logit               [B]
sigmoid compatibility score    [B]
```

Il training confronta i logits `[B]` con `labels [B]` tramite Binary Focal
Loss. Le descrizioni restano sequenze Python perché SentenceBERT riceve testo;
non vengono convertite in ID dal dataset loader.

## Caricamento da Hugging Face

Prima dell'uso:

```powershell
hf auth login
```

Il factory loader scarica la configurazione richiesta, il file di
compatibility corretto e i metadati:

```python
from torch.utils.data import DataLoader

from data import (
    collate_compatibility,
    load_polyvore_compatibility_dataset,
)

train_dataset = load_polyvore_compatibility_dataset(
    variant="nondisjoint",  # oppure "disjoint"
    split="train",
)
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_compatibility,
)
```

Sono disponibili gli split `train`, `validation` e `test`.

I file scaricati non vengono salvati dentro il repository. Vanno nella cache
di Hugging Face:

- se passi `--cache-dir D:\datasets\huggingface`, Parquet, JSON e metadata
  vengono messi lì;
- se non passi `--cache-dir`, Hugging Face usa la cache predefinita, di solito
  `~/.cache/huggingface/datasets` per i Parquet e
  `~/.cache/huggingface/hub` per file come `compatibility_*.txt`,
  `<split>.json` e `polyvore_item_metadata.json`.

`train_cp.py` stampa questi percorsi all'avvio.

Per test locali o layout personalizzati, `PolyvoreCompatibilityDataset`
accetta direttamente le righe outfit, il percorso del file
`compatibility_*.txt` e, se le immagini non sono incorporate nelle righe,
un `image_root`.

## Loader generico alternativo

`OutfitDataset` legge un manifest JSON:

```json
[
  {
    "outfit_id": "outfit-001",
    "items": [
      {
        "image": "outfit-001/top.jpg",
        "text": "white cotton shirt"
      },
      {
        "image": "outfit-001/trousers.jpg",
        "text": "navy tailored trousers"
      }
    ]
  }
]
```

I percorsi delle immagini sono relativi a `image_root`. Ogni immagine viene:

1. convertita in RGB;
2. ridimensionata a `224 × 224`;
3. convertita in tensor;
4. normalizzata con media e deviazione standard ImageNet.

Il file
[example_manifest.json](manifest_loader/example_manifest.json) mostra il formato
atteso, ma contiene percorsi dimostrativi: le immagini devono essere fornite
separatamente.

## DataLoader

```python
from torch.utils.data import DataLoader

from data import OutfitDataset, collate_outfits

dataset = OutfitDataset(
    manifest_path="data/manifest.json",
    image_root="data/images",
)
loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=collate_outfits,
)
```

Ogni iterazione restituisce un `OutfitBatch` senza label CP:

| Campo | Forma | Contenuto |
|---|---|---|
| `images` | `[B,L,3,224,224]` | Immagini, comprese le posizioni padded |
| `descriptions` | `B` sequenze | Testi dei capi reali |
| `padding_mask` | `[B,L]` | Posizioni reali e padded |
| `outfit_ids` | `B` stringhe | Identificativi degli outfit |

## Padding mask

Gli outfit possono contenere numeri diversi di capi, ma un batch PyTorch deve
avere una forma rettangolare. `collate_outfits()` usa la lunghezza dell'outfit
più grande (`L`) e aggiunge posizioni `PAD` agli outfit più corti:

```text
Outfit 1: maglia   pantaloni   scarpe
Outfit 2: camicia  jeans       PAD
```

In questo esempio:

```text
B = 2 outfit
L = 3 posizioni per outfit
capi reali = 5
```

Il batch risultante contiene:

```text
images:        [2, 3, 3, 224, 224]
descriptions:  2 sequenze di testi
padding_mask:  [2, 3]
```

La padding mask ha un valore booleano per ogni posizione:

```python
padding_mask = [
    [False, False, False],
    [False, False, True],
]
```

- `False` significa che la posizione contiene un capo reale;
- `True` significa che la posizione è padding e deve essere ignorata.

Un singolo valore della maschera controlla l'intero embedding:

```text
padding_mask[1, 2] = True
                    ↓
ignora item_embeddings[1, 2, :]
```

La maschera non modifica separatamente le singole feature.

### Uso prima del Transformer

`OutfitEncoder.encode_items()` inverte la maschera per trovare i capi validi:

```python
valid_mask = ~padding_mask
```

Soltanto le immagini e descrizioni reali vengono inviate a ResNet-18 e
SentenceBERT. Gli embedding validi vengono reinseriti nella forma
`[B,L,128]`, lasciando un vettore di zeri nelle posizioni padded.

### Uso nel Transformer

La stessa maschera viene passata come `src_key_padding_mask`. Le posizioni
`True` vengono escluse dall'attenzione e azzerate nuovamente nell'output.

Quando CP o CIR aggiungono un task token, antepongono `False` alla maschera:

```text
mask originale:       [False, False, True]
con OUTFIT o TARGET:  [False, False, False, True]
                       ↑
                    task token
```

## File

```text
data/
  batch.py              batch, padding e maschere
  transforms.py         preprocessing ImageNet
  manifest_loader/
    README.md            guida del loader JSON generico
    dataset.py           lettura del manifest e delle immagini
    example_manifest.json
  polyvore_loader/
    README.md            guida del loader Polyvore CP
    dataset.py           split ufficiali Polyvore per CP
```
