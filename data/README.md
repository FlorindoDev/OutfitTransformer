# Dati e batching

Il package `data` legge gli outfit, applica il preprocessing delle immagini e
costruisce batch con padding e maschere.

Torna al [README principale](../README.md) oppure consulta
l'[architettura comune](../model/common/README.md).

## Manifest

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

Il file [example_manifest.json](example_manifest.json) mostra il formato
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

Ogni iterazione restituisce un `OutfitBatch` con:

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
  dataset.py            lettura del manifest e delle immagini
  transforms.py         preprocessing ImageNet
  example_manifest.json formato di esempio
```
