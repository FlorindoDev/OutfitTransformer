# Componenti comuni del training

`training.common` raccoglie le utilità condivise dai runner di training CP e
CIR: profili delle feature, lettura degli embedding precomputati, persistenza
atomica degli artefatti, metriche binarie, riproducibilità e scelta del device.

## Indice

- [Scopo](#scopo)
- [File](#file)
- [Profili delle feature](#profili-delle-feature)
- [Cache degli embedding](#cache-degli-embedding)
- [Checkpoint e configurazioni](#checkpoint-e-configurazioni)
- [Metriche](#metriche)

## Scopo

Il package evita di duplicare nei training CP e CIR il codice operativo che
non dipende dal task. Non definisce dataset, architetture o cicli di training:
queste responsabilità rimangono rispettivamente nei moduli `data.py`,
`model.py` e `trainer.py` di ciascun task.

## File

| File | Cosa fa |
|---|---|
| [`features.py`](features.py) | Definisce i profili `classic`, `new_classic` e `precomputed`, seleziona dimensioni e configurazione del Transformer del task e serializza le informazioni sugli encoder. |
| [`embeddings.py`](embeddings.py) | Espone una cache read-only indicizzata per `item_id`, caricata da manifest e shard PyTorch memory-mapped. |
| [`checkpointing.py`](checkpointing.py) | Salva e copia checkpoint, legge state dict validati, carica i pesi del modello e scrive configurazioni JSON con operazioni atomiche. |
| [`metrics.py`](metrics.py) | Accumula loss, accuracy e ROC AUC sull'intera epoca per i task di classificazione binaria. |
| [`runtime.py`](runtime.py) | Imposta i seed riproducibili e risolve automaticamente o valida il device PyTorch. |
| [`__init__.py`](__init__.py) | Riunisce ed espone l'API pubblica di `training.common`. |
| [`README.md`](README.md) | Documenta responsabilità, file e comportamento dei componenti condivisi. |

## Profili delle feature

`FeatureMode` identifica la sorgente delle rappresentazioni degli item:

| Modalità | Input durante il training | Configurazione predefinita |
|---|---|---|
| `classic` | Immagini e descrizioni originali | ResNet-18 e SentenceTransformer, proiettati a `64 + 64` feature |
| `new_classic` | Immagini e descrizioni originali | ResNet-18 e SentenceTransformer, proiettati a `512 + 512` feature |
| `precomputed` | Embedding letti dalla cache | Rappresentazioni combinate da 1024 feature per il Transformer del task |

`default_transformer_config()` associa ogni modalità alle dimensioni corrette.
`feature_config()` produce invece i metadati serializzabili salvati nella
configurazione del run, senza includere credenziali. Il parser accetta anche i
nomi legacy `fashion_clip_approach`, `clip` e `openrouter` presenti nei vecchi
checkpoint.

## Cache degli embedding

`EmbeddingCache` implementa una mappa read-only da `item_id` a tensore. Legge
un `manifest.json` con schema 2 e carica gli shard tramite memory mapping,
evitando di duplicare in memoria l'intero contenuto dei file.

Prima di esporre i dati verifica:

- identità di dataset, subset e split, quando specificati dal chiamante;
- dimensione e numero degli embedding dichiarati nel manifest;
- esistenza e formato di ogni shard;
- forma, tipo floating-point e valori finiti dei tensori;
- presenza di identificatori vuoti o duplicati.

Una ricerca di un `item_id` assente termina con un errore esplicito, così una
cache incompleta viene rilevata prima o durante la preparazione del training.

## Checkpoint e configurazioni

Le funzioni di `checkpointing.py` scrivono prima un file temporaneo con
estensione `.tmp` e lo sostituiscono alla destinazione soltanto al termine
dell'operazione. Lo stesso comportamento viene usato per checkpoint, copia del
best model e file JSON.

`load_model_weights()` ripristina esclusivamente lo state dict del modello con
controllo `strict=True`. Accetta sia un checkpoint completo contenente
`model_state_dict`, sia uno state dict salvato direttamente; optimizer,
scheduler e history non vengono ripristinati.

`load_checkpoint_state_dict()` espone la stessa lettura e validazione senza
applicare i pesi a un modello. Permette ai task di implementare trasferimenti
selettivi, come l'inizializzazione CIR da CP.

## Metriche

`BinaryEpochAccumulator` riceve loss, probabilità e target di ogni batch,
stacca i tensori dal grafo e conserva su CPU i valori necessari. Al termine
dell'epoca produce `EpochMetrics` con:

- loss media per esempio;
- accuracy con soglia `0.5`;
- ROC AUC calcolata dall'implementazione pubblica in `metrics`;
- numero totale di esempi.

Questo accumulatore è destinato alla classificazione binaria CP. Il training
CIR calcola separatamente le proprie metriche di retrieval FITB.
