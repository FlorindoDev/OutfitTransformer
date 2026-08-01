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
  - [Loss coerente col checkpoint](#loss-coerente-col-checkpoint)
- [Opzioni](#opzioni)
- [Struttura](#struttura)

## Compatibility Prediction

Il comando CP usa configurazione e tutti i tensori del modello salvati nel
checkpoint, poi esegue il modello sullo split `test` di Polyvore Outfits e
stampa:

- loss media usata nel training (`focal` o `bce`);
- accuracy con soglia `0.5`;
- ROC AUC calcolata su tutti i logits del test set;
- numero di esempi elaborati.

```powershell
python -m evaluate.cp `
  --checkpoint checkpoints\cp_best.pt
```

Variante, architettura, SentenceBERT e loss (`focal` o `bce`) vengono letti da
`run_config`. Se un flag CLI esplicito entra in conflitto col checkpoint, il
comando termina invece di produrre una valutazione non confrontabile. I flag
restano fallback per checkpoint legacy privi di `run_config`.

Optimizer, scheduler, history e RNG descrivono il training e non vengono
ripristinati: non partecipano all'inferenza. Il test set resta sempre `test`,
non viene mescolato e i pesi vengono caricati con controllo stretto di ogni
parametro. Device, worker, cache e batch size restano opzioni runtime: non
cambiano modello o protocollo di valutazione.

### Loss coerente col checkpoint

Durante la valutazione non vengono eseguiti backpropagation o aggiornamenti dei
pesi. Tipo di loss e relativi parametri vengono ripresi automaticamente dal
checkpoint per rendere `test_loss` confrontabile con la validation loss.
`focal-gamma` serve esclusivamente nel calcolo della Focal Loss:

```text
gamma = 0  → equivale alla Binary Cross-Entropy
gamma = 1  → riduce moderatamente il peso degli esempi facili
gamma = 2  → riduce più fortemente il peso degli esempi facili
```

Gamma non cambia i logits prodotti dal modello, quindi non modifica accuracy o
ROC AUC; cambia soltanto `test_loss`. `--loss`, `--focal-alpha` e
`--focal-gamma` servono solo per checkpoint legacy che non hanno questi valori
in `run_config`.

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
| `--variant` | checkpoint; fallback `disjoint` | Variante Polyvore per checkpoint legacy |
| `--batch-size` | `16` | Outfit per batch |
| `--workers` | `0` | Processi del DataLoader |
| `--device` | automatico | `cuda` se disponibile, altrimenti `cpu` |
| `--cache-dir` | cache Hugging Face | Posizione della cache |
| `--checkpoint` | `checkpoints/cp_best.pt` | Checkpoint CP da caricare |
| `--text-model` | checkpoint; fallback MiniLM-L6-v2 | Encoder testuale per checkpoint legacy |
| `--loss` | checkpoint; fallback `focal` | Loss per checkpoint legacy: `focal` o `bce` |
| `--focal-alpha` | checkpoint; fallback `0.5` | Peso positivo per checkpoint legacy |
| `--focal-gamma` | checkpoint; fallback `2.0` | Gamma per checkpoint legacy |
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
