# Training Compatibility Prediction

Questa guida descrive come allenare OutfitTransformer esclusivamente sul task
di Compatibility Prediction (CP): dato un insieme di capi, il modello deve
predire se l'outfit è compatibile (`1`) oppure incompatibile (`0`).

Il comando principale è [train_cp.py](../train_cp.py). Il ciclo riutilizzabile
si trova in [cp.py](cp.py), mentre formato e preparazione di Polyvore sono
descritti nel [README dei dati](../data/README.md).

## Cosa viene allenato

Il forward usato durante il training è:

```text
immagini + descrizioni
        ↓
ResNet-18 + SentenceBERT/FC
        ↓
item embedding multimodale
        ↓
token OUTFIT + Transformer
        ↓
TaskMLP
        ↓
logit di compatibilità
        ↓
Binary Focal Loss
```

Vengono aggiornati:

- ResNet-18;
- proiezione FC delle feature testuali;
- token apprendibile `OUTFIT`;
- Transformer;
- classificatore `TaskMLP`.

Il backbone SentenceBERT resta congelato. Il suo output viene comunque
calcolato a ogni batch, ma non riceve aggiornamenti dei pesi.

## Prerequisiti

Eseguire i comandi dalla radice del progetto.

### Ambiente Python

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Le dipendenze principali sono PyTorch, torchvision, Sentence Transformers,
Pillow e Hugging Face Datasets.

### Accesso a Polyvore Outfits

Il dataset `mvasil/polyvore-outfits` è gated. Richiedere l'accesso dalla
[pagina Hugging Face](https://huggingface.co/datasets/mvasil/polyvore-outfits)
e autenticare la macchina:

```powershell
hf auth login
```

Per verificare l'account:

```powershell
hf auth whoami
```

### Verifica CUDA

```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'nessuna')"
```

`train_cp.py` sceglie automaticamente `cuda` quando disponibile, altrimenti
usa `cpu`.

## Comando minimo

```powershell
python train_cp.py
```

Equivale a:

```powershell
python train_cp.py `
  --variant disjoint `
  --epochs 20 `
  --batch-size 32 `
  --learning-rate 1e-4 `
  --weight-decay 0 `
  --lr-step-size 10 `
  --lr-gamma 0.5 `
  --focal-alpha 0.25 `
  --focal-gamma 2 `
  --workers 0 `
  --seed 42 `
  --log-interval 50 `
  --checkpoint checkpoints/cp_best.pt `
  --checkpoint-dir checkpoints/cp_epochs `
  --text-model sentence-transformers/all-MiniLM-L6-v2
```

Il device non compare nel comando equivalente perché il valore predefinito è
dinamico: `cuda` se PyTorch rileva una GPU CUDA, altrimenti `cpu`.

Per visualizzare l'help sempre aggiornato:

```powershell
python train_cp.py --help
```

## Tutti gli argomenti della CLI

| Argomento | Default | Valori/vincoli | Effetto |
|---|---:|---|---|
| `--variant` | `nondisjoint` | `nondisjoint`, `disjoint` | Seleziona la variante Polyvore |
| `--epochs` | `20` | intero `> 0` | Numero completo di passaggi sul training set |
| `--batch-size` | `32` | intero `> 0` | Numero di outfit per aggiornamento |
| `--learning-rate` | `1e-4` | float `> 0` | Learning rate iniziale di ADAM |
| `--weight-decay` | `0.0` | float | Regolarizzazione L2 applicata da ADAM |
| `--lr-step-size` | `10` | intero `> 0` | Epoche tra due riduzioni del learning rate |
| `--lr-gamma` | `0.5` | float in `(0,1]` | Fattore moltiplicativo di ogni riduzione |
| `--focal-alpha` | `0.25` | float in `[0,1]` | Peso relativo della classe positiva |
| `--focal-gamma` | `2.0` | float `>= 0` | Intensità con cui la loss riduce gli esempi facili |
| `--max-grad-norm` | disabilitato | float `> 0` | Abilita il gradient clipping sulla norma globale |
| `--workers` | `0` | intero `>= 0` | Processi DataLoader per caricare i dati |
| `--seed` | `42` | intero | Seed per `random`, PyTorch e CUDA |
| `--log-interval` | `50` | intero `>= 0` | Stampa avanzamento ogni N batch; `0` disabilita i log batch |
| `--device` | automatico | `cpu`, `cuda`, `cuda:0`, ... | Device usato da batch e modello |
| `--cache-dir` | cache Hugging Face | percorso | Directory in cui scaricare dataset e metadati |
| `--checkpoint` | `checkpoints/cp_best.pt` | percorso `.pt` | Destinazione del checkpoint migliore |
| `--checkpoint-dir` | `checkpoints/cp_epochs` | directory | Salva un checkpoint per ogni epoca |
| `--text-model` | `sentence-transformers/all-MiniLM-L6-v2` | nome Hub o directory locale | Backbone SentenceBERT congelato |
| `--no-pretrained-image` | falso | flag senza valore | Inizializza ResNet-18 casualmente invece che con ImageNet |

### Epoche e batch size

`--epochs` stabilisce quante volte vengono attraversate tutte le domande di
`compatibility_train.txt`.

`--batch-size` controlla memoria e frequenza degli aggiornamenti:

- un batch più grande usa più VRAM e produce gradienti mediamente più stabili;
- un batch più piccolo usa meno memoria, ma esegue più aggiornamenti per epoca.

Il training loader usa `shuffle=True`; la validation usa `shuffle=False`.

### Ottimizzatore

Il modello usa:

```python
torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay,
)
```

Tutti i parametri allenabili sono registrati nell'optimizer. SentenceBERT è
presente nel modello ma ha `requires_grad=False`.

I valori ADAM non esposti dalla CLI restano quelli predefiniti di PyTorch:

| Parametro ADAM | Valore |
|---|---:|
| `beta1` | `0.9` |
| `beta2` | `0.999` |
| `eps` | `1e-8` |
| `amsgrad` | `False` |

### Scheduler

Il learning rate è gestito da `StepLR`:

```text
lr = learning_rate × lr_gamma^(numero di riduzioni)
```

Con i default:

| Epoche | Learning rate |
|---|---:|
| 1–10 | `1e-4` |
| 11–20 | `5e-5` |
| 21–30 | `2.5e-5` |

`--lr-gamma 1` mantiene costante il learning rate, pur lasciando attivo lo
scheduler.

### Binary Focal Loss

La loss media del batch è:

```text
FL = -alpha_t × (1 - p_t)^gamma × log(p_t)
```

`--focal-gamma` controlla l'attenzione agli esempi difficili:

- `0`: nessuna modulazione focale;
- `2`: default;
- valori maggiori riducono più aggressivamente il contributo degli esempi
  già classificati correttamente.

`--focal-alpha` bilancia le classi:

```text
classe positiva: alpha
classe negativa: 1 - alpha
```

Con `alpha=0.25`, i positivi ricevono peso `0.25` e i negativi `0.75`.
`alpha=0.5` assegna lo stesso peso relativo alle due classi.

La CLI accetta soltanto valori numerici. Per disabilitare completamente
`alpha`, usare programmaticamente:

```python
from model import BinaryFocalLoss

criterion = BinaryFocalLoss(alpha=None, gamma=2.0)
```

La riduzione della loss è fissata a `mean`: il valore usato per il backward è
la media delle Focal Loss degli outfit nel batch.

### Gradient clipping

Il clipping è disabilitato per default. Per limitare la norma globale dei
gradienti:

```powershell
python train_cp.py --max-grad-norm 1.0
```

Il clipping viene applicato dopo `loss.backward()` e prima
di `optimizer.step()`.

### Worker del DataLoader

Su Windows il valore più sicuro è:

```powershell
--workers 0
```

Valori come `2`, `4` o `8` possono velocizzare il caricamento, ma aumentano
RAM e processi. La configurazione CUDA abilita automaticamente
`pin_memory=True`.

### Seed e riproducibilità

Il seed inizializza:

- `random`;
- PyTorch CPU;
- tutti i device CUDA disponibili.

```powershell
python train_cp.py --seed 123
```

Questo migliora la ripetibilità, ma non garantisce risultati bit-per-bit
identici su tutte le GPU: il codice non forza gli algoritmi deterministici di
PyTorch.

## Iperparametri dell'architettura

Questi valori sono definiti in `OutfitEncoderConfig` e non sono esposti come
argomenti di `train_cp.py`:

| Iperparametro | Default | Significato |
|---|---:|---|
| `image_embedding_dim` | `64` | Dimensione dell'output ResNet-18 |
| `text_embedding_dim` | `64` | Dimensione della proiezione testuale |
| `item_embedding_dim` | `128` | Concatenazione `64 + 64` |
| `transformer_layers` | `6` | Numero di Transformer Encoder layer |
| `attention_heads` | `16` | Teste di self-attention per layer |
| `feedforward_dim` | `512` | Dimensione interna del feed-forward |
| `dropout` | `0.1` | Dropout del Transformer |
| `pretrained_image_encoder` | `True` | Inizializzazione ImageNet di ResNet-18 |
| hidden dimension `TaskMLP` | `128` | Classificatore `128 → 128 → 1` |
| attivazione Transformer | ReLU | Non linearità dei feed-forward layer |
| attivazione `TaskMLP` | ReLU | Non linearità tra i due layer lineari |
| positional encoding | assente | Gli outfit sono trattati come insiemi non ordinati |

Il vincolo principale è:

```text
item_embedding_dim % attention_heads == 0
```

Con i default: `128 % 16 == 0`.

Per cambiare questi valori bisogna costruire programmaticamente la
configurazione:

```python
from model import CompatibilityPredictor, OutfitEncoderConfig

config = OutfitEncoderConfig(
    image_embedding_dim=128,
    text_embedding_dim=128,
    transformer_layers=8,
    attention_heads=16,
    feedforward_dim=1024,
    dropout=0.1,
)
model = CompatibilityPredictor(config=config, hidden_dim=256)
```

Questa configurazione produce item embedding da `256` feature e un
classificatore `256 → 256 → 1`. Per allenarla con `train_cp()` bisogna
replicare il setup DataLoader/optimizer di `train_cp.py`.

## Iperparametri fissi del preprocessing

| Valore | Configurazione |
|---|---|
| formato immagine | RGB |
| dimensione | `224 × 224` |
| dtype | `float32` |
| range prima della normalizzazione | `[0,1]` |
| media ImageNet | `(0.485, 0.456, 0.406)` |
| deviazione standard ImageNet | `(0.229, 0.224, 0.225)` |
| padding immagini | tensor di zeri |

Le posizioni padded sono indicate da `padding_mask=True` e non contribuiscono
agli encoder degli item né all'attenzione.

## Altre scelte fisse del training

| Componente | Valore | Conseguenza |
|---|---|---|
| optimizer | ADAM | Aggiorna tutti i parametri con `requires_grad=True` |
| riduzione loss | media | Una loss scalare per batch |
| soglia classe positiva | `logit >= 0` | Equivale a score sigmoid `>= 0.5` |
| training shuffle | attivo | Ordine diverso degli esempi a ogni epoca |
| validation shuffle | disattivo | Ordine stabile durante la valutazione |
| checkpoint monitorato | validation loss | Salva il minimo osservato |
| gradient accumulation | assente | Un optimizer step per batch |
| mixed precision | assente | Forward e backward standard |

## Comandi di training

### Nondisjoint con impostazioni predefinite

```powershell
python train_cp.py --variant nondisjoint
```

Gli outfit non si sovrappongono tra gli split, ma alcuni item possono
comparire sia nel training sia nella validation o nel test.

### Disjoint

```powershell
python train_cp.py --variant disjoint
```

Gli item del training non compaiono nella validation o nel test. È la
configurazione più severa per misurare la generalizzazione.

### GPU specifica

```powershell
python train_cp.py --device cuda:0
```

Per usare un'altra GPU:

```powershell
python train_cp.py --device cuda:1
```

### CPU

```powershell
python train_cp.py --device cpu --batch-size 8 --workers 0
```

Il training completo su CPU può essere molto lento.

### Configurazione consigliata per VRAM limitata

```powershell
python train_cp.py `
  --variant nondisjoint `
  --device cuda `
  --batch-size 4 `
  --max-grad-norm 1.0 `
  --workers 0
```

Il codice non implementa gradient accumulation: ridurre il batch size cambia
anche la dimensione effettiva del batch.

### Run più lunga con learning rate decrescente

```powershell
python train_cp.py `
  --variant nondisjoint `
  --epochs 50 `
  --batch-size 32 `
  --learning-rate 1e-4 `
  --lr-step-size 10 `
  --lr-gamma 0.5 `
  --checkpoint checkpoints/nondisjoint_50e.pt
```

### Training disjoint personalizzato

```powershell
python train_cp.py `
  --variant disjoint `
  --epochs 30 `
  --batch-size 16 `
  --learning-rate 5e-5 `
  --weight-decay 1e-4 `
  --focal-alpha 0.5 `
  --focal-gamma 2 `
  --max-grad-norm 1.0 `
  --checkpoint checkpoints/disjoint_best.pt
```

### Cache del dataset in una directory scelta

```powershell
python train_cp.py `
  --cache-dir D:\datasets\huggingface `
  --checkpoint checkpoints/cp_best.pt
```

La cache deve avere spazio sufficiente per i Parquet, i metadati e i modelli
scaricati.

Se `--cache-dir` non viene passato, Hugging Face usa la propria cache
predefinita, per esempio `~/.cache/huggingface/datasets` e
`~/.cache/huggingface/hub`. All'avvio `train_cp.py` stampa i percorsi usati:

```text
dataset_cache=...
hub_cache=...
```

### SentenceBERT locale

```powershell
python train_cp.py `
  --text-model D:\models\all-MiniLM-L6-v2 `
  --cache-dir D:\datasets\huggingface
```

### ResNet-18 senza pesi ImageNet

```powershell
python train_cp.py --no-pretrained-image
```

Questa opzione inizializza casualmente ResNet-18. Non disabilita né modifica
SentenceBERT.

### Esempio completo

```powershell
python train_cp.py `
  --variant nondisjoint `
  --epochs 40 `
  --batch-size 16 `
  --learning-rate 1e-4 `
  --weight-decay 1e-4 `
  --lr-step-size 8 `
  --lr-gamma 0.5 `
  --focal-alpha 0.25 `
  --focal-gamma 2 `
  --max-grad-norm 1.0 `
  --workers 4 `
  --seed 42 `
  --device cuda:0 `
  --cache-dir D:\datasets\huggingface `
  --checkpoint checkpoints\nondisjoint_best.pt `
  --text-model sentence-transformers/all-MiniLM-L6-v2
```

## Output durante il training

All'avvio vengono stampati dataset, cache, device, numero di esempi, numero di
batch e percorsi dei checkpoint:

```text
training=compatibility_prediction
dataset=mvasil/polyvore-outfits variant=disjoint
dataset_cache=...
hub_cache=...
train_examples=...
validation_examples=...
checkpoint_best=...\checkpoints\cp_best.pt
checkpoint_epochs=...\checkpoints\cp_epochs
```

Durante ogni epoca viene stampato l'avanzamento ogni `--log-interval` batch:

```text
epoch=1 phase=train batch=50/320 loss=0.052100 running_loss=0.061233 running_accuracy=0.8125 examples=1600
```

Al termine di ogni epoca viene stampata una riga simile:

```text
epoch=3 train_loss=0.041382 train_accuracy=0.8641 train_examples=18000 lr=0.00010000 val_loss=0.047926 val_accuracy=0.8415 val_examples=2000
```

| Valore | Significato |
|---|---|
| `train_loss` | Binary Focal Loss media sul training set |
| `train_accuracy` | Accuratezza training con soglia logit `>= 0` |
| `val_loss` | Binary Focal Loss media sulla validation |
| `val_accuracy` | Accuratezza validation con soglia logit `>= 0` |
| `lr` | Learning rate dopo lo step dello scheduler dell'epoca |

L'accuracy usa `sigmoid(logit) >= 0.5`, equivalente a `logit >= 0`.

## Checkpoint

Lo script salva due tipi di checkpoint:

- `checkpoints/cp_epochs/cp_epoch_001.pt`, `cp_epoch_002.pt`, ...: uno per
  ogni epoca;
- `checkpoints/cp_best.pt`: sovrascritto solo quando la validation loss
  raggiunge un nuovo minimo.

Ogni salvataggio viene stampato:

```text
checkpoint=epoch epoch=3 path=...\checkpoints\cp_epochs\cp_epoch_003.pt monitored_loss=0.047926
checkpoint=best epoch=3 path=...\checkpoints\cp_best.pt monitored_loss=0.047926
```

Ogni checkpoint contiene:

```text
epoch
model_state_dict
optimizer_state_dict
scheduler_state_dict
monitored_loss
train_metrics
validation_metrics
```

Per leggere il checkpoint:

```powershell
python -c "import torch; c=torch.load('checkpoints/cp_best.pt', map_location='cpu', weights_only=True); print(c['epoch'], c['monitored_loss'])"
```

Il checkpoint selezionato è quello con validation loss migliore, non
necessariamente quello dell'ultima epoca.

Il resume automatico non è ancora implementato: `train_cp.py` non accetta un
argomento `--resume`. Il checkpoint contiene comunque gli stati necessari
per aggiungerlo in futuro.

## Cosa non è incluso

La pipeline attuale non implementa:

- mixed precision/AMP;
- gradient accumulation;
- early stopping;
- resume automatico;
- TensorBoard o servizi di experiment tracking;
- training multi-GPU;
- valutazione AUC sul test set;
- training CIR.

Queste limitazioni non impediscono il training CP, ma vanno considerate quando
si confrontano esperimenti o si pianificano run lunghe.

## Errori comuni

### Dataset non autorizzato

```text
Access to dataset mvasil/polyvore-outfits is restricted
```

Accettare le condizioni sulla pagina del dataset e rieseguire:

```powershell
hf auth login
```

### CUDA out of memory

Ridurre il batch:

```powershell
python train_cp.py --batch-size 4 --workers 0
```

### Modello SentenceBERT non scaricabile

Usare un modello già presente sul disco:

```powershell
python train_cp.py --text-model D:\models\all-MiniLM-L6-v2
```

### Training troppo lento su Windows

Provare gradualmente:

```powershell
python train_cp.py --workers 2
python train_cp.py --workers 4
```

Se il processo diventa instabile o consuma troppa RAM, tornare a
`--workers 0`.

## Controllo rapido prima di una run lunga

```powershell
python -m unittest discover -s tests -v
python train_cp.py --help
hf auth whoami
python -c "import torch; print(torch.cuda.is_available())"
```

Verificare inoltre:

- spazio libero nella cache;
- percorso del checkpoint;
- variante corretta;
- batch size compatibile con la VRAM;
- accesso già approvato al dataset;
- download o disponibilità locale di SentenceBERT.
