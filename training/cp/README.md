# Training CP

Guida operativa per allenare OutfitTransformer sul task di **Compatibility
Prediction (CP)**: dato un outfit, il modello predice se è compatibile (`1`) o
incompatibile (`0`).

- Torna alla [panoramica Training CP/CIR](../README.md).
- Consulta il [modello CP](../../model/cp/README.md).
- Consulta il [formato dei dati Polyvore](../../data/README.md).
- Consulta la [guida alla valutazione](../../evaluate/README.md).
- Consulta la [guida alle metriche](../../metrics/README.md).

## Indice

- [Flusso](#flusso)
  - [Cosa aggiorna la backpropagation](#cosa-aggiorna-la-backpropagation)
- [Avvio rapido](#avvio-rapido)
- [Flag della CLI](#flag-della-cli)
- [Iperparametri](#iperparametri)
  - [Ottimizzazione](#ottimizzazione)
  - [Preprocessing](#preprocessing)
- [Scheduler](#scheduler)
- [Training e validation](#training-e-validation)
- [Checkpoint e resume](#checkpoint-e-resume)
- [Valutazione sul test set](#valutazione-sul-test-set)
- [Comandi utili](#comandi-utili)
- [File](#file)

## Flusso

```text
immagini + descrizioni
        ↓
ResNet-18 + SentenceBERT/FC
        ↓
item embedding da 128 feature
        ↓
token OUTFIT + Transformer encoder-only
        ↓
TaskMLP → logit
        ↓
Binary Focal Loss
        ↓
backpropagation + Adam
```

### Cosa aggiorna la backpropagation

```mermaid
flowchart TD
    A["Binary Focal Loss<br/>nessun parametro"] -->|gradiente| B["TaskMLP CP<br/>aggiornato"]
    B --> C["Transformer encoder-only<br/>aggiornato"]

    C --> D["Token OUTFIT<br/>aggiornato"]
    C --> E["ResNet-18 + FC visuale<br/>aggiornati"]
    C --> F["Proiezione testuale FC<br/>aggiornata"]
    F -.->|"gradiente interrotto"| G["SentenceBERT<br/>congelato"]

    classDef trained fill:#d5f5e3,stroke:#239b56,color:#17202a
    classDef frozen fill:#f2f3f4,stroke:#7b7d7d,color:#17202a
    classDef loss fill:#fdebd0,stroke:#ca6f1e,color:#17202a

    class B,C,D,E,F trained
    class G frozen
    class A loss
```

SentenceBERT produce le feature testuali dentro `torch.no_grad()`: il suo
backbone non cambia, mentre la proiezione FC successiva viene allenata.

Il token `OUTFIT` è un unico `nn.Parameter` di forma `[1, 1, 128]`, condiviso
da tutti gli outfit. I suoi 128 valori sono essi stessi pesi allenabili: non
sono prodotti dall'attenzione o da un'altra rete. Vengono inizializzati una
sola volta con valori casuali e diventano una rappresentazione base appresa.

Durante il forward questa stessa base viene espansa alla dimensione del batch
e inserita davanti agli item. Il Transformer ne produce poi una versione
contestualizzata diversa per ciascun outfit:

```text
token OUTFIT base condiviso + item dell'outfit
                       ↓ Transformer
outfit embedding contestualizzato e specifico dell'outfit
```

La loss dipende dall'outfit embedding; di conseguenza `loss.backward()` fa
passare il gradiente attraverso classificatore e Transformer fino anche ai 128
valori del token base. `optimizer.step()` aggiorna direttamente sia il token
base sia i pesi del Transformer. Il token non viene ricreato a ogni forward e,
durante validation e inferenza, rimane fisso.

| Elemento | Che cos'è | Come cambia |
|---|---|---|
| `outfit_token` | Parametro base globale, salvato nel checkpoint | Aggiornato direttamente da Adam |
| `outfit_embedding` | Output contestualizzato per un singolo outfit | Ricalcolato a ogni forward, non è un parametro |

## Avvio rapido

Eseguire dalla root del progetto:

```powershell
python -m pip install -r requirements.txt
hf auth login
python -m training.cp.train_cp
```

La configurazione predefinita usa la variante `disjoint`, 30 epoche, batch da
32 e il device CUDA quando disponibile.

## Flag della CLI

```powershell
python -m training.cp.train_cp --help
```

| Flag | Default | Funzione |
|---|---:|---|
| `-h`, `--help` | — | Mostra l'help |
| `--variant` | `disjoint` | Variante Polyvore: `disjoint` o `nondisjoint` |
| `--epochs` | `30` | Ultima epoca da eseguire |
| `--batch-size` | `32` | Numero di outfit per batch |
| `--learning-rate` | `5e-5` | Learning rate iniziale di Adam |
| `--weight-decay` | `1e-4` | Regolarizzazione L2 di Adam |
| `--lr-step-size` | `10` | Epoche tra due riduzioni del learning rate |
| `--lr-gamma` | `0.5` | Fattore moltiplicativo dello scheduler |
| `--focal-alpha` | `0.5` | Bilanciamento della classe positiva |
| `--focal-gamma` | `1.0` | Riduzione del peso degli esempi facili |
| `--max-grad-norm` | `1.0` | Limite della norma globale dei gradienti |
| `--workers` | `0` | Processi del DataLoader |
| `--seed` | `42` | Seed Python, PyTorch e CUDA |
| `--log-interval` | `50` | Stampa ogni N batch; `0` disabilita i log batch |
| `--device` | automatico | `cuda` se disponibile, altrimenti `cpu` |
| `--cache-dir` | cache Hugging Face | Posizione della cache dataset e Hub |
| `--checkpoint` | `checkpoints/cp_best.pt` | Checkpoint con validation loss minima |
| `--checkpoint-dir` | `checkpoints/cp_epochs` | Directory dei checkpoint per epoca |
| `--resume` | disabilitato | Riprende training, optimizer e scheduler |
| `--text-model` | `sentence-transformers/all-MiniLM-L6-v2` | SentenceBERT Hub o locale |
| `--no-pretrained-image` | falso | Non usa i pesi ImageNet di ResNet-18 |

## Iperparametri

### Ottimizzazione

| Iperparametro | Default |
|---|---:|
| optimizer | Adam |
| learning rate | `5e-5` |
| weight decay | `1e-4` |
| batch size | `32` |
| epoche | `30` |
| gradient clipping | `1.0` |
| Focal Loss alpha | `0.5` |
| Focal Loss gamma | `1.0` |

La Focal Loss concentra il training sugli outfit incerti o classificati male.
Il clipping viene applicato dopo `loss.backward()` e prima
di `optimizer.step()`.

Gli iperparametri dell'architettura sono documentati una sola volta nella
sezione [Transformer encoder-only](../../model/cp/README.md#transformer-encoder-only)
del modello CP.

### Preprocessing

| Impostazione | Valore |
|---|---|
| dimensione immagine | `224 × 224` |
| normalizzazione | media e deviazione standard ImageNet |
| ResNet-18 | pesi ImageNet, salvo `--no-pretrained-image` |
| SentenceBERT | congelato |
| padding | a destra |
| train shuffle | attivo |
| validation shuffle | attivo |

## Scheduler

Lo scheduler modifica il learning rate durante il training. Il progetto usa
`StepLR`:

```python
scheduler = StepLR(
    optimizer,
    step_size=10,
    gamma=0.5,
)
```

Con il learning rate predefinito:

```text
epoche 1–10:   0.000050
epoche 11–20:  0.000025
epoche 21–30:  0.0000125
```

Alla fine di ogni epoca viene chiamato `scheduler.step()`. Lo scheduler:

- non calcola gradienti;
- non modifica direttamente i pesi;
- riduce il learning rate di Adam secondo una scadenza fissa;
- non sceglie il checkpoint migliore: quello dipende dalla validation loss.

Il suo stato viene salvato nei checkpoint e ripristinato con `--resume`.

## Training e validation

Il training usa gli split ufficiali di `mvasil/polyvore-outfits`. Ogni
esempio viene ricostruito attraverso questi mapping:

```text
compatibility_*.txt      -> label + token set_id_index
<split>.json             -> token set_id_index -> item_id
Parquet                  -> item_id -> immagine
polyvore_item_metadata   -> item_id -> descrizione
```

Il dataset lo restituisce come:

```text
CompatibilityExample(
    images=[N, 3, 224, 224],
    descriptions=tuple di N stringhe,
    label=1.0 oppure 0.0,
)
```

Il dettaglio di file, mapping e caricamento è raccolto nella
[guida ai dati Polyvore](../../data/README.md).

Ogni epoca esegue:

1. training con forward, Focal Loss, backward e aggiornamento dei pesi;
2. validation con `model.eval()` e gradienti disabilitati;
3. step dello scheduler;
4. salvataggio dei checkpoint.

I log batch mostrano:

```text
loss=... running_loss=... running_accuracy=... examples=...
```

- `loss`: loss del batch corrente;
- `running_loss`: media cumulativa dell'epoca;
- `running_accuracy`: accuracy cumulativa;
- `examples`: outfit elaborati fino a quel momento.

A fine epoca vengono stampate `train_loss`, `train_accuracy`, `val_loss`,
`val_accuracy` e learning rate.

Il loop riutilizzabile espone:

| Funzione | Ruolo |
|---|---|
| `run_cp_epoch()` | Esegue un'epoca di training o validation |
| `train_cp()` | Gestisce epoche, validation, scheduler e checkpoint |

## Checkpoint e resume

Vengono salvati:

```text
checkpoints/cp_epochs/cp_epoch_001.pt
checkpoints/cp_epochs/cp_epoch_002.pt
...
checkpoints/cp_best.pt
```

- `cp_epoch_NNN.pt` conserva ogni epoca;
- `cp_best.pt` viene aggiornato solo quando la validation loss raggiunge un
  nuovo minimo.

Ogni checkpoint contiene:

| Campo | Contenuto |
|---|---|
| `epoch` | Ultima epoca completata |
| `model_state_dict` | Pesi di encoder, Transformer e classificatore |
| `optimizer_state_dict` | Stato di Adam |
| `scheduler_state_dict` | Stato dello scheduler, se presente |
| `monitored_loss` | Validation loss usata per selezionare il modello |
| `train_metrics` | Loss, accuracy ed esempi di training |
| `validation_metrics` | Loss, accuracy ed esempi di validation |

Per riprendere:

```powershell
python -m training.cp.train_cp `
  --epochs 40 `
  --resume checkpoints\cp_best.pt
```

`--epochs` indica l'ultima epoca totale, non quante epoche aggiungere.
Il modello SentenceBERT e la configurazione architetturale devono coincidere
con quelli del checkpoint; altrimenti il caricamento strict dei pesi segnala
l'incompatibilità.

## Valutazione sul test set

Il test set non viene usato durante il training. Dopo avere scelto
`cp_best.pt`:

```powershell
python -m evaluate.cp `
  --variant disjoint `
  --checkpoint checkpoints\cp_best.pt `
  --focal-gamma 1.0
```

La valutazione non aggiorna i pesi e stampa test loss, accuracy, ROC AUC ed
esempi. L'AUC usa i logits per misurare quanto spesso un outfit compatibile
riceve un punteggio superiore a uno incompatibile.
Variante, SentenceBERT e parametri della Focal Loss devono coincidere con il
training.
Per tutte le opzioni e la descrizione delle metriche consulta la
[guida alla valutazione](../../evaluate/README.md).

## Comandi utili

```powershell
# Variante nondisjoint
python -m training.cp.train_cp --variant nondisjoint

# GPU specifica
python -m training.cp.train_cp --device cuda:0

# VRAM limitata
python -m training.cp.train_cp --batch-size 8

# Cache personalizzata
python -m training.cp.train_cp --cache-dir D:\datasets\huggingface

# SentenceBERT locale
python -m training.cp.train_cp `
  --text-model D:\models\all-MiniLM-L6-v2

# Disabilita i log batch
python -m training.cp.train_cp --log-interval 0
```

## File

```text
training/cp/
  train_cp.py   CLI e configurazione della run
  trainer.py    epoche, validation e checkpoint
  README.md     questa guida
```
