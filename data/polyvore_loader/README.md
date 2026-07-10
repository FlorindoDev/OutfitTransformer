# Polyvore compatibility loader

Loader specifico per il task Compatibility Prediction sul dataset
[`mvasil/polyvore-outfits`](https://huggingface.co/datasets/mvasil/polyvore-outfits).

La spiegazione completa di file, identificatori, esempi reali e forme dei
tensori si trova nella
[guida Polyvore](../README.md#dataset-fornito-polyvore-outfits).

## Indice

- [Input](#input)
- [`download.py`](#downloadpy-acquisizione-delle-risorse)
- [`dataset.py`](#datasetpy-interpretazione-di-polyvore)
- [`loader.py`](#loaderpy-creazione-dei-batch)
- [`__init__.py`](#__init__py-api-pubblica)
- [`README.md`](#readmemd-documentazione)
- [Flusso completo](#flusso-completo)

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

## `download.py`: acquisizione delle risorse

Rappresenta il confine con Hugging Face. Riceve variante, split e posizione
della cache; scarica oppure riusa immagini, domande di compatibility, mapping e
metadati. Raggruppa i riferimenti ottenuti in `PolyvoreResources`.

Non interpreta gli outfit, non trasforma immagini e non crea batch. Il suo
unico compito è rendere disponibili tutte le sorgenti richieste da Polyvore.

## `dataset.py`: interpretazione di Polyvore

Conosce il formato interno del dataset. Collega i token `set_id_index` agli
item, associa immagine e descrizione, applica il preprocessing condiviso e
costruisce un `CompatibilityExample` alla volta.

Implementa quindi il concetto PyTorch di `Dataset`: espone numero di esempi e
accesso a un singolo esempio. Non decide batch size, shuffle o parallelismo e
non scarica file.

## `loader.py`: creazione dei batch

Coordina i componenti precedenti. Ottiene le risorse, crea il Dataset e lo
inserisce nel `DataLoader` PyTorch. Usa `data/batch.py` per padding, maschera e
label; usa `data/transforms.py` per il formato delle immagini.

Il risultato è un flusso di `CompatibilityBatch` già compatibile con il
modello. Qui vengono configurati batch size, shuffle, worker, memoria pinned e
gestione dell'ultimo batch.


## Flusso completo

```text
Hugging Face
    ↓
download.py → PolyvoreResources
    ↓
dataset.py → CompatibilityExample
    ↓
loader.py + data/batch.py + data/transforms.py
    ↓
CompatibilityBatch
    ↓
modello
```

Ogni livello conosce solo la propria responsabilità. Un nuovo dataset può
replicare questa struttura mantenendo invariato il formato ricevuto dal
modello.
