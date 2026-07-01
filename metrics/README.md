# Metriche

Questo package contiene metriche indipendenti dal training e dagli entry point
di valutazione. Possono quindi essere riutilizzate durante training,
validation e test senza creare dipendenze tra questi moduli.

- Torna al [README principale](../README.md).
- Consulta la [guida alla valutazione](../evaluate/README.md).
- Consulta il [training CP](../training/cp/README.md).

## Indice

- [Metriche di classificazione](#metriche-di-classificazione)
  - [`BinaryAccuracy`](#binaryaccuracy)
  - [`binary_roc_auc`](#binary_roc_auc)
- [Accuracy, AUC e loss](#accuracy-auc-e-loss)
- [Struttura](#struttura)

## Metriche di classificazione

Le metriche disponibili sono esportate da `metrics`:

```python
from metrics import BinaryAccuracy, binary_roc_auc
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
  classification.py   BinaryAccuracy e binary_roc_auc
  README.md           descrizione e utilizzo delle metriche
```
