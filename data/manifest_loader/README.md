# Manifest loader

Loader generico per outfit descritti da un manifest JSON locale. Non conosce
Polyvore e non contiene label di Compatibility Prediction.

Torna alla [guida completa dei dati](../README.md).

## Indice

- [Input](#input)
- [Uso](#uso)
- [Output](#output)
- [Responsabilità della cartella](#responsabilità-della-cartella)

## Input

Il file [example_manifest.json](example_manifest.json) mostra il formato:

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

`image` è relativo a `image_root`. Ogni item deve contenere esattamente
un'immagine e una descrizione.

## Uso

```python
from data import create_manifest_outfit_loader

loader = create_manifest_outfit_loader(
    manifest_path="data/manifest_loader/example_manifest.json",
    image_root="data/images",
    batch_size=8,
    shuffle=True,
)
```

Per accesso al solo Dataset resta disponibile `OutfitDataset`.

## Output

Un esempio contiene:

```text
OutfitExample(
    outfit_id: str,
    images: Tensor[N, 3, 224, 224],
    descriptions: tuple di N stringhe,
)
```

Un batch contiene immagini `[B,L,3,224,224]` e `padding_mask [B,L]`, ma non
contiene label CP. Questo loader è utile per inferenza, demo o dataset
personalizzati già organizzati dall'utente.

## Responsabilità della cartella

```text
manifest_loader/
  __init__.py
  dataset.py
  loader.py
  example_manifest.json
  README.md
```
