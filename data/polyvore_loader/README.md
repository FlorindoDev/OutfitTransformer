# Polyvore compatibility loader

Loader specifico per il task Compatibility Prediction sul dataset
[`mvasil/polyvore-outfits`](https://huggingface.co/datasets/mvasil/polyvore-outfits).

La spiegazione completa di file, identificatori, esempi reali e forme dei
tensori si trova nella
[guida Polyvore](../README.md#dataset-fornito-polyvore-outfits).

## Indice

- [Input](#input)
- [Uso](#uso)
- [Output](#output)
- [Responsabilità della cartella](#responsabilità-della-cartella)

## Input

Il loader combina quattro sorgenti:

```text
Parquet dello split
  + compatibility_<split>.txt
  + <split>.json
  + polyvore_item_metadata.json
```

- il Parquet fornisce righe `item_id` + immagine;
- `compatibility_*.txt` fornisce label e token `set_id_index`;
- `<split>.json` traduce ogni token `set_id_index` nel relativo `item_id`;
- `polyvore_item_metadata.json` fornisce il testo dell'item, indicizzato per
  `item_id`.

Quindi `descriptions` non viene letto da `compatibility_*.txt`. Quella riga
contiene solo label e token. Il loader usa il token per trovare l'`item_id`;
con lo stesso `item_id` prende l'immagine dal Parquet e il testo da
`polyvore_item_metadata.json`, che sta nella radice del dataset.

Non genera esempi negativi e non usa i file FITB.

## Uso

```python
from torch.utils.data import DataLoader

from data import (
    collate_compatibility,
    load_polyvore_compatibility_dataset,
)

dataset = load_polyvore_compatibility_dataset(
    variant="nondisjoint",
    split="train",
)
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_compatibility,
)
```

## Output

Un esempio contiene:

```text
CompatibilityExample(
    outfit_id: str,
    images: Tensor[N, 3, 224, 224],
    descriptions: tuple di N stringhe,
    label: 0.0 oppure 1.0,
)
```

Il collate produce:

```text
CompatibilityBatch(
    images: Tensor[B, L, 3, 224, 224],
    descriptions: B tuple,
    padding_mask: Tensor[B, L],
    labels: Tensor[B],
)
```

## Responsabilità della cartella

```text
polyvore_loader/
  __init__.py
  dataset.py
  README.md
```
