# Compatibility Prediction

Il modulo `model/cp` assegna a un outfit un punteggio di compatibilità compreso
tra 0 e 1.

Torna al [README principale](../../README.md) oppure consulta
l'[architettura condivisa](../common/README.md).

## Flusso

L'encoder antepone un token `OUTFIT` apprendibile agli item embedding:

```text
[OUTFIT, capo 1, capo 2, ..., capo L]
```

```mermaid
flowchart LR
    A["Item embeddings<br/>B × L × 128"]
    B["Outfit token<br/>B × 1 × 128"]
    A --> C["Prepend"]
    B --> C
    C --> D["Transformer"]
    D --> E["Output OUTFIT<br/>B × 128"]
    E --> F["TaskMLP<br/>128 → 128 → 1"]
    F --> G["Logit"]
    G --> H["Sigmoid"]
    H --> I["Compatibility score<br/>0–1"]
    G --> J["Binary Focal Loss"]
```

L'output del token in posizione zero è la rappresentazione globale
dell'outfit. `TaskMLP` la trasforma in un logit; la sigmoid produce il
compatibility score.

## Utilizzo

```python
from model import BinaryFocalLoss, CompatibilityPredictor

model = CompatibilityPredictor()
criterion = BinaryFocalLoss()

output = model(
    batch.images,
    batch.descriptions,
    batch.padding_mask,
)
loss = criterion(output.logits, compatibility_labels)
```

`compatibility_labels` deve avere la stessa forma di `output.logits` e
contenere valori nell'intervallo `[0,1]`:

- `1` indica un outfit compatibile;
- `0` indica un outfit incompatibile.

`CompatibilityOutput` contiene:

| Campo | Forma | Significato |
|---|---|---|
| `logits` | `[B]` | Valori non normalizzati usati dalla loss |
| `compatibility_score` | `[B]` | Probabilità ottenute con la sigmoid |
| `outfit_embedding` | `[B,128]` | Rappresentazione globale dell'outfit |

## Binary Focal Loss

La Binary Focal Loss è una funzione di errore per classificazione binaria che
riduce il contributo degli esempi già classificati facilmente e concentra
l'addestramento su quelli incerti o sbagliati.

### Dal logit alla probabilità

Il modello produce un logit $z$, che la sigmoid trasforma nella probabilità
di compatibilità:

$$
p=\sigma(z)=\frac{1}{1+e^{-z}}
$$

Per esempio, $p=0.9$ significa che il modello considera l'outfit compatibile
con probabilità 90%.

### Probabilità della classe corretta

Durante il training è disponibile l'etichetta reale $y$. Si definisce:

$$
p_t=
\begin{cases}
p & \text{se } y=1\\
1-p & \text{se } y=0
\end{cases}
$$

$p_t$ è quindi la probabilità assegnata alla classe corretta:

| Etichetta $y$ | Predizione $p$ | $p_t$ | Interpretazione |
|---:|---:|---:|---|
| 1 | 0.95 | 0.95 | Corretta e facile |
| 1 | 0.20 | 0.20 | Sbagliata e difficile |
| 0 | 0.05 | 0.95 | Corretta e facile |
| 0 | 0.80 | 0.20 | Sbagliata e difficile |

Non serve assegnare manualmente una difficoltà:

- $p_t$ alto indica un esempio facile;
- $p_t$ vicino a 0.5 indica un esempio incerto;
- $p_t$ basso indica un esempio difficile o classificato erroneamente.

La difficoltà non è permanente. Uno stesso outfit può essere difficile
all'inizio del training e diventare facile quando il modello impara a
riconoscerlo.

### Binary cross-entropy e $-\log(p_t)$

La Binary Cross-Entropy per un singolo esempio può essere scritta come:

$$
\mathrm{BCE}=-\log(p_t)
$$

Il logaritmo trasforma la probabilità assegnata alla classe corretta in una
penalità:

| $p_t$ | $-\log(p_t)$ | Interpretazione |
|---:|---:|---|
| 0.99 | 0.010 | Penalità quasi nulla |
| 0.90 | 0.105 | Penalità piccola |
| 0.50 | 0.693 | Modello incerto |
| 0.10 | 2.303 | Penalità grande |
| 0.01 | 4.605 | Penalità molto grande |

Quando $p_t$ tende a 1, $-\log(p_t)$ tende a 0. Quando $p_t$ tende a 0,
$-\log(p_t)$ cresce rapidamente: una risposta sbagliata data con grande
sicurezza riceve una penalità molto alta.

Il segno meno è necessario perché $\log(p_t)$ è negativo per $0<p_t<1$,
mentre la loss deve essere positiva. Rispetto alla semplice quantità $1-p_t$,
il logaritmo penalizza più severamente gli errori commessi con grande
sicurezza.

### Il peso focale e $\gamma$

Molti esempi facili possono dominare l'addestramento quando le loro loss
vengono sommate. La Focal Loss riduce il loro contributo moltiplicando la BCE
per:

$$
(1-p_t)^\gamma
$$

Senza bilanciamento delle classi:

$$
\mathrm{FL}(p_t)=-(1-p_t)^\gamma\log(p_t)
$$

$\gamma$ controlla quanto aggressivamente vengono ridimensionati gli esempi
facili. Con $\gamma=2$:

| $p_t$ | Peso $(1-p_t)^2$ |
|---:|---:|
| 0.95 | 0.0025 |
| 0.50 | 0.25 |
| 0.10 | 0.81 |

Un esempio facile riceve un peso molto piccolo, mentre un esempio difficile
conserva gran parte della propria penalità. Con $\gamma=0$, il peso è sempre 1
e la Focal Loss coincide con la normale BCE.

### Come collaborano BCE e peso focale

Con $\gamma=2$ e senza considerare per il momento $\alpha$:

$$
\mathrm{FL}=(1-p_t)^2[-\log(p_t)]
$$

| $p_t$ | BCE $-\log(p_t)$ | Peso focale | Focal Loss |
|---:|---:|---:|---:|
| 0.95 | 0.051 | 0.0025 | 0.00013 |
| 0.50 | 0.693 | 0.25 | 0.173 |
| 0.10 | 2.303 | 0.81 | 1.865 |

I fattori hanno ruoli differenti:

$$
\underbrace{-\log(p_t)}_{\text{penalità della predizione}}
\qquad
\underbrace{(1-p_t)^\gamma}_{\text{attenzione data all'esempio}}
$$

La Focal Loss non penalizza maggiormente gli esempi facili: riduce quasi a
zero il loro contributo e concentra gli aggiornamenti sugli errori.

### Il bilanciamento delle classi con $\alpha$

La forma completa è:

$$
\mathrm{FL}(p_t)=-\alpha_t(1-p_t)^\gamma\log(p_t)
$$

In forma espansa:

$$
\mathrm{FL}(p,y)=
-\alpha y(1-p)^\gamma\log(p)
-(1-\alpha)(1-y)p^\gamma\log(1-p)
$$

I due iperparametri hanno scopi distinti:

- $\gamma$ bilancia esempi facili e difficili;
- $\alpha$ bilancia classe positiva e negativa.

Se gli outfit compatibili sono rari, il valore di $\alpha$ può essere scelto
per dare più importanza alla classe positiva.

`BinaryFocalLoss` usa:

```text
alpha:     0.25
gamma:     2.0
reduction: mean
```

Impostando `alpha=None` si disabilita il bilanciamento delle classi.
Impostando `gamma=0` e `alpha=None` si ottiene la normale BCE.

In sintesi:

$$
\boxed{
\text{Focal Loss}
=
\underbrace{-\log(p_t)}_{\text{errore di classificazione}}
\cdot
\underbrace{(1-p_t)^\gamma}_{\text{difficoltà dell'esempio}}
\cdot
\underbrace{\alpha_t}_{\text{peso della classe}}
}
$$

L'implementazione riceve direttamente i logits e usa
`binary_cross_entropy_with_logits`, evitando instabilità numeriche dovute al
calcolo separato di sigmoid e logaritmo.

## Quali componenti vengono aggiornati

La loss viene calcolata sul logit prodotto dal classificatore, ma il gradiente
può attraversare tutta la rete:

```text
immagini e testi
    → encoder
    → Transformer
    → outfit token
    → classifier
    → logit
    → Binary Focal Loss
```

Un passo di training tipico è:

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

optimizer.zero_grad()
output = model(
    batch.images,
    batch.descriptions,
    batch.padding_mask,
)
loss = criterion(output.logits, compatibility_labels)
loss.backward()
optimizer.step()
```

`loss.backward()` calcola i gradienti per i parametri con
`requires_grad=True`; `optimizer.step()` aggiorna soltanto i parametri
registrati nell'optimizer.

Con `model.parameters()` vengono aggiornati:

- classificatore `TaskMLP`;
- outfit token;
- Transformer;
- ResNet-18;
- proiezione FC del text encoder.

Il backbone SentenceBERT rimane congelato.

Per allenare soltanto il classificatore:

```python
optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-4)
```

La Focal Loss stabilisce il segnale d'errore; `requires_grad` e l'optimizer
stabiliscono quali parametri cambiano.

## Training CP completo

Lo script usa gli split ufficiali Polyvore, ADAM, Binary Focal Loss,
validazione a ogni epoca, scheduler e checkpoint del modello con validation
loss migliore:

```powershell
python train_cp.py --variant nondisjoint --epochs 20 --batch-size 32
```

Per lo split senza item condivisi:

```powershell
python train_cp.py --variant disjoint
```

Il checkpoint predefinito viene scritto in `checkpoints/cp_best.pt`.
Iperparametri principali:

```text
--learning-rate 1e-4
--focal-alpha 0.25
--focal-gamma 2.0
--lr-step-size 10
--lr-gamma 0.5
--max-grad-norm <valore opzionale>
```

Il loop riutilizzabile si trova in `training.cp`: `run_cp_epoch()` esegue una
singola epoca, mentre `train_cp()` gestisce training, validation, scheduler e
checkpoint. Tutti gli argomenti, gli iperparametri architetturali e i comandi
pronti all'uso sono raccolti nella
[guida completa al training](../../training/README.md).

## Test

Dalla radice del progetto:

```powershell
python -m unittest tests.test_losses.BinaryFocalLossTests -v
python -m unittest tests.test_task_models.CompatibilityPredictorTests -v
```

## File

```text
model/cp/
  compatibility.py  outfit embedding e compatibility score
  focal_loss.py      Binary Focal Loss
```

```text
training/
  README.md          comandi e iperparametri
  cp.py              loop di training e validazione CP
train_cp.py          CLI per Polyvore Outfits
```
