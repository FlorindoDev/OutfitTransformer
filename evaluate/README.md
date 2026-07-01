# Valutazione

Questa cartella contiene gli entry point per valutare checkpoint già
addestrati. La valutazione usa soltanto il test set e non aggiorna i pesi del
modello.

- Torna al [README principale](../README.md).
- Consulta il [training CP](../training/cp/README.md).
- Consulta il [modello CP](../model/cp/README.md).
- Consulta la [guida alle metriche](../metrics/README.md).

## Indice

- [Compatibility Prediction](#compatibility-prediction)
  - [Perché specificare `focal-gamma` durante la valutazione](#perché-specificare-focal-gamma-durante-la-valutazione)
- [Opzioni](#opzioni)
- [Struttura](#struttura)

## Compatibility Prediction

Il comando CP carica un checkpoint, esegue il modello sullo split `test` di
Polyvore Outfits e stampa:

- Binary Focal Loss media;
- accuracy con soglia `0.5`;
- ROC AUC calcolata su tutti i logits del test set;
- numero di esempi elaborati.

```powershell
python -m evaluate.cp `
  --variant disjoint `
  --checkpoint checkpoints\cp_best.pt `
  --focal-alpha 0.5 `
  --focal-gamma 1.0
```

Variante, SentenceBERT e parametri della Focal Loss devono corrispondere a
quelli usati durante il training. `focal-alpha` e `focal-gamma` modificano la
loss riportata, ma non accuracy e AUC.

### Perché specificare `focal-gamma` durante la valutazione

Durante la valutazione non vengono eseguiti backpropagation o aggiornamenti dei
pesi. `focal-gamma` serve esclusivamente a calcolare `test_loss` con la stessa
Focal Loss utilizzata durante il training:

```text
gamma = 0  → equivale alla Binary Cross-Entropy
gamma = 1  → riduce moderatamente il peso degli esempi facili
gamma = 2  → riduce più fortemente il peso degli esempi facili
```

Cambiare gamma durante la valutazione non cambia i logits prodotti dal modello
e quindi non modifica accuracy o ROC AUC. Cambia soltanto `test_loss`. Per
confrontare correttamente validation loss e test loss bisogna usare lo stesso
valore del training; con il training predefinito attuale va quindi passato
`--focal-gamma 1.0`.

L'output finale ha questa forma:

```text
test_loss=0.123456 test_accuracy=0.8500 test_auc=0.9100 test_examples=30290
```

L'AUC viene calcolata soltanto dopo avere raccolto i logits dell'intero test
set. I log intermedi mostrano quindi loss e accuracy, mentre `test_auc` compare
alla fine. Definizioni, formule ed esempi sono disponibili nella
[guida alle metriche](../metrics/README.md).

## Opzioni

```powershell
python -m evaluate.cp --help
```

| Flag | Default | Funzione |
|---|---:|---|
| `--variant` | `disjoint` | Variante Polyvore da valutare |
| `--batch-size` | `16` | Outfit per batch |
| `--workers` | `0` | Processi del DataLoader |
| `--device` | automatico | `cuda` se disponibile, altrimenti `cpu` |
| `--cache-dir` | cache Hugging Face | Posizione della cache |
| `--checkpoint` | `checkpoints/cp_best.pt` | Checkpoint CP da caricare |
| `--text-model` | `sentence-transformers/all-MiniLM-L6-v2` | Encoder testuale |
| `--focal-alpha` | `0.5` | Peso della classe positiva nella Focal Loss |
| `--focal-gamma` | `2.0` | Riduzione del peso degli esempi facili |
| `--log-interval` | `50` | Frequenza dei log batch; `0` li disabilita |

Per visualizzare soltanto il risultato finale:

```powershell
python -m evaluate.cp `
  --checkpoint checkpoints\cp_best.pt `
  --log-interval 0
```

## Struttura

```text
evaluate/
  __init__.py
  cp.py        valutazione CP sul test set
  README.md    comandi, opzioni e metriche
```
