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
  - [Dalla distanza al rank](#dalla-distanza-al-rank)
  - [Loss di training e validation](#loss-di-training-e-validation)
  - [Metriche di ranking in validation](#metriche-di-ranking-in-validation)
  - [Esempio completo](#esempio-completo)
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

### Dalla distanza al rank

Indicando con $q$ l'embedding della query e con $c$ quello di un candidato, la
distanza euclidea è:

$$
d(q,c) = \sqrt{\sum_j (q_j-c_j)^2}
$$

La distanza vale $0$ per embedding identici e cresce quanto più i due embedding
sono lontani. Non è limitata all'intervallo $[0,1]$.

Il **rank** è la posizione dell'item corretto dopo aver ordinato i candidati
dalla distanza più piccola alla più grande:

- rank $1$: il positivo è il candidato più vicino;
- rank $2$: un distrattore è più vicino del positivo;
- rank $4$: il positivo è l'ultimo dei quattro candidati.

Operativamente, il rank del positivo si calcola così:

$$
\operatorname{rank}(c^+)
= 1 + \#\left\{c^- : d(q,c^-) \leq d(q,c^+)\right\}
$$

dove $c^+$ è il candidato positivo e $c^-$ indica un distrattore.

Per esempio, con le distanze $d^+=0.7$ per il positivo e $0.4$, $1.1$, $1.8$
per i distrattori, un solo distrattore precede il positivo e quindi il rank è
$2$. In caso di parità, il distrattore viene considerato prima del positivo. Se
le quattro
distanze fossero tutte uguali, il positivo avrebbe quindi rank $4$, evitando
un risultato artificialmente ottimistico.

### Loss di training e validation

Per ogni query viene usato il negativo più difficile, cioè quello scorretto con
la distanza minore. Indicando con $d^+$ la distanza dal positivo, con $d^-$ la
distanza dal negativo più difficile e con $m$ il margine configurato, la
penalità è:

$$
\mathcal{L} = \max\left(0, d^+ - d^- + m\right)
$$

La loss è zero quando il negativo è più lontano del positivo di almeno $m$.
Altrimenti misura quanto manca per rispettare questo margine. Per esempio, con
$d^+=1.2$, $d^-=2.5$ e $m=2.0$, la loss è
$\max(0, 1.2-2.5+2.0)=0.7$.

Il CIR riporta due loss medie per epoca:

- `train_loss`: durante il training, il negativo più difficile è scelto tra i
  positivi delle altre righe dello stesso microbatch. È l'obiettivo che guida
  la backpropagation; più è bassa, meglio il modello separa le coppie di
  training;
- `val_loss`: durante la validation, il negativo più difficile è scelto tra i
  tre distrattori ufficiali associati alla query. Serve a controllare la
  generalizzazione; non aggiorna i pesi e non decide quale checkpoint salvare
  come migliore.

Le due loss usano insiemi di negativi diversi, quindi è più utile osservare il
loro andamento nel tempo che confrontarne direttamente i valori assoluti.

### Metriche di ranking in validation

Siano $N$ il numero di query di validation e $r_i$ il rank del positivo per la
query $i$. Il CIR calcola le metriche seguenti.

#### FITB accuracy

$$
\text{FITB accuracy}
= \frac{\left|\left\{i : r_i=1\right\}\right|}{N}
$$

Misura la percentuale di domande **Fill In The Blank** per le quali il modello
sceglie subito l'item corretto. Varia tra $0$ e $1$; più è alta, meglio è. Con
quattro candidati, una scelta casuale ha in media accuracy $0.25$. È la metrica
usata per scegliere `best.pt` e per l'early stopping.

#### Mean Reciprocal Rank (MRR)

$$
\operatorname{MRR} = \frac{1}{N}\sum_{i=1}^{N}\frac{1}{r_i}
$$

Assegna a ogni query un punteggio che diminuisce con la posizione del positivo:
rank $1$ vale $1$, rank $2$ vale $0.5$, rank $3$ vale circa $0.333$ e rank $4$
vale $0.25$. È utile perché rileva miglioramenti anche quando il positivo non ha
ancora raggiunto il primo posto. Con quattro candidati varia tra $0.25$ e $1$;
più è alta, meglio è.

#### Recall@2

$$
\operatorname{Recall@2}
= \frac{\left|\left\{i : r_i \leq 2\right\}\right|}{N}
$$

Misura quanto spesso l'item corretto compare tra i primi due candidati. Ogni
query vale $1$ se il positivo è al rank $1$ o $2$ e $0$ negli altri casi. Varia
tra $0$ e $1$; più è alta, meglio è. È meno severa della FITB accuracy e mostra
se il modello riesce almeno a restringere la scelta ai due candidati migliori.

#### Numero di esempi

Insieme ai valori precedenti viene registrato anche il numero di esempi
elaborati. Nel training conta le coppie query-positivo usate; nella validation
conta le query FITB valutate. Non è una misura di qualità, ma serve a verificare
che le medie siano state calcolate sul numero atteso di dati, anche in training
distribuito.

### Esempio completo

Supponiamo che quattro query producano i rank $[1,3,2,4]$:

$$
\begin{aligned}
\text{FITB accuracy} &= \frac{1}{4} = 0.25 \\
\operatorname{MRR} &= \frac{1+\frac{1}{3}+\frac{1}{2}+\frac{1}{4}}{4}
\approx 0.521 \\
\operatorname{Recall@2} &= \frac{2}{4} = 0.50
\end{aligned}
$$

Il modello trova subito un positivo su quattro, colloca due positivi su quattro
nei primi due posti e ottiene un MRR superiore alla FITB accuracy perché questa
metrica riconosce anche i piazzamenti al secondo, terzo e quarto posto.

Il checkpoint `best.pt` e l'early stopping dipendono soltanto dalla massima
FITB accuracy di validation; MRR, Recall@2 e le loss aiutano a interpretare il
training. Le metriche di ranking non generano gradienti e non aggiornano i pesi.

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
