# Metriche

Questo package contiene metriche indipendenti dal training e dagli entry point
di valutazione. Possono quindi essere riutilizzate durante training,
validation e test senza creare dipendenze tra questi moduli.

- Torna al [README principale](../README.md).
- Consulta la [guida alla valutazione](../evaluation/README.md).
- Consulta il [training CP](../training/CP/README.md).
- Consulta il [training CIR](../training/CIR/README.md).

## Indice

- [Metriche di classificazione](#metriche-di-classificazione)
  - [`BinaryAccuracy`](#binaryaccuracy)
  - [`binary_roc_auc`](#binary_roc_auc)
  - [`binary_classification_metrics`](#binary_classification_metrics)
- [Metriche di retrieval CIR](#metriche-di-retrieval-cir)
  - [Rank e `retrieval_rank`](#rank-e-retrieval_rank)
  - [Metriche calcolate dal CIR](#metriche-calcolate-dal-cir)
- [Accuracy, AUC e loss](#accuracy-auc-e-loss)
- [Struttura](#struttura)

## Metriche di classificazione

Le metriche disponibili sono esportate da `metrics`:

```python
from metrics import (
    BinaryAccuracy,
    binary_classification_metrics,
    binary_roc_auc,
)
```

### `BinaryAccuracy`

`BinaryAccuracy` accumula il numero di classificazioni corrette su uno o più
batch. Riceve logits e target con la stessa forma e applica queste soglie:

```text
logit >= 0   → classe compatibile
target >= 0.5 → classe compatibile
```

La soglia `logit >= 0` equivale a una probabilità sigmoid maggiore o uguale a
`0.5`. La metrica viene aggiornata per ogni batch e `compute()` restituisce:

```text
accuracy = predizioni corrette / numero di esempi
```

Esempio:

```python
import torch

from metrics import BinaryAccuracy

accuracy = BinaryAccuracy()
accuracy.update(
    logits=torch.tensor([1.2, -0.4, 0.8]),
    targets=torch.tensor([1.0, 0.0, 0.0]),
)

print(accuracy.compute())  # 0.666...
```

Nel CP è usata per i log progressivi e per l'accuracy finale di training,
validation e test.

### `binary_roc_auc`

`binary_roc_auc(scores, targets)` misura quanto spesso un esempio positivo
riceve un punteggio maggiore di un esempio negativo:

```text
AUC = 1.0 → ordinamento perfetto
AUC = 0.5 → ordinamento casuale
AUC = 0.0 → ordinamento completamente invertito
```

La funzione:

- riceve tensori monodimensionali della stessa lunghezza;
- richiede score finiti e target binari `0/1`;
- richiede almeno un esempio positivo e uno negativo;
- assegna mezzo punto ai punteggi in parità;
- usa direttamente i logits, perché la sigmoid non ne cambia l'ordinamento.

Esempio:

```python
import torch

from metrics import binary_roc_auc

auc = binary_roc_auc(
    scores=torch.tensor([1.2, 0.8, -0.3, -1.0]),
    targets=torch.tensor([1, 1, 0, 0]),
)

print(auc)  # 1.0
```

Nel CP l'AUC viene calcolata sull'intero test set. È una metrica di
valutazione: non genera gradienti e non aggiorna i pesi.

### `binary_classification_metrics`

`binary_classification_metrics(probabilities, targets, threshold=0.5)` calcola
accuracy, precision, recall, F1 e ROC AUC. Riceve probabilita in `[0, 1]` e
target binari monodimensionali. Soglia riguarda solo metriche discrete; AUC
usa probabilita originali. Risultato include anche numero totale di esempi.

## Metriche di retrieval CIR

Durante la validation CIR, ogni outfit parziale è una **query** confrontata con
quattro candidati: l'item corretto e tre distrattori. Il modello calcola la
distanza euclidea tra i rispettivi embedding; una distanza minore indica una
corrispondenza migliore.

### Rank e `retrieval_rank`

Il **rank** è la posizione dell'item corretto dopo aver ordinato i candidati
dalla distanza più piccola alla più grande:

- rank 1: il positivo è il candidato più vicino;
- rank 2: un distrattore è più vicino del positivo;
- rank 4: il positivo è l'ultimo dei quattro candidati.

Per esempio, se le distanze ordinate sono `distrattore=0.4`, `positivo=0.7`,
`distrattore=1.1`, `distrattore=1.8`, il positivo ha rank 2. La funzione
`retrieval_rank(candidate_distances, positive_index=0)` calcola questa posizione
a partire dalle distanze. In caso di parità, il concorrente viene considerato
prima del positivo: così embedding tutti uguali non producono risultati
artificialmente ottimistici.

### Metriche calcolate dal CIR

| Valore | Concetto | Lettura |
|---|---|---|
| `train_loss` | Triplet Margin Loss tra il positivo e il negativo più difficile trovato tra gli altri positivi del microbatch. | Più bassa è meglio; è l'unico valore che guida la backpropagation. |
| `val_loss` | Stessa penalità di margine, ma il negativo più difficile è scelto tra i tre distrattori FITB ufficiali. | Più bassa è meglio; serve per diagnosi e non seleziona il checkpoint. |
| `val_fitb_accuracy` | Frazione di query nelle quali il positivo ha rank 1. | Misura quante domande hanno una risposta esatta; più alta è meglio. |
| `val_mrr` | Media di `1 / rank`: rank 1 vale `1`, rank 2 vale `0.5`, rank 4 vale `0.25`. | Premia anche i miglioramenti che non portano ancora il positivo al primo posto. |
| `val_recall@2` | Frazione di query nelle quali il positivo ha rank 1 o 2. | Misura quanto spesso la risposta corretta compare tra i primi due candidati. |

Con i rank `[1, 3, 2]`, FITB accuracy vale `1/3`, MRR vale
`(1 + 1/3 + 1/2) / 3` e Recall@2 vale `2/3`.

`retrieval_metrics(ranks)` restituisce insieme questi tre riepiloghi di ranking
e il numero di esempi nella struttura `RetrievalMetrics`. Le funzioni
`fitb_accuracy()`, `mean_reciprocal_rank()` e `recall_at_k()` permettono di
calcolarli separatamente. Nessuna di queste metriche genera gradienti.

Il checkpoint `best.pt` e l'early stopping dipendono soltanto dalla massima
`val_fitb_accuracy`; MRR, Recall@2 e le loss aiutano a interpretare il training.

## Accuracy, AUC e loss

| Valore | Cosa misura | Dipende da una soglia | Usato dal backpropagation |
|---|---|---:|---:|
| Accuracy | Percentuale di classificazioni corrette | Sì | No |
| ROC AUC | Qualità dell'ordinamento positivi/negativi | No | No |
| Binary Focal Loss | Errore da minimizzare | No | Sì, durante il training |

## Struttura

```text
metrics/
  __init__.py
  classification.py   Metriche binarie per training ed evaluation CP
  retrieval.py        Rank, FITB accuracy, MRR e Recall@K per CIR
  README.md           descrizione e utilizzo delle metriche
```
