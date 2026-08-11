# Training Compatibility Prediction

Training CP supporta due sorgenti di feature selezionabili con flag mutuamente
esclusivi. `--classic` usa encoder del paper durante il training; `--clip` usa
embedding FashionCLIP precomputati. Transformer common, Transformer CP, token e
testa di classificazione vengono allenati in entrambe le modalità.

## Modalità

| Flag | Feature degli item | Dimensione | Parti allenabili | Precomputazione |
|---|---|---:|---|---|
| `--classic` | ResNet-18 ImageNet + SentenceBERT | `64 + 64 = 128` | ResNet-18, proiezioni, Transformer e CP; backbone SentenceBERT congelato | Non richiesta |
| `--clip` | FashionCLIP visuale + testo | `512 + 512 = 1024` | Transformer e CP; tower FashionCLIP congelate | Richiesta per train e validation |

Profilo `classic` segue encoder, dimensioni e Transformer 6 layer/16 teste del
paper. Usa inoltre baseline storica del progetto: feed-forward 512, dropout
`0.1` e post-norm. Profilo `clip` mantiene configurazione modello corrente: 6
layer, 16 teste, feed-forward 2024, dropout `0.3` e pre-norm.

## Configurazione predefinita

| Aspetto | Valore |
|---|---|
| Dataset | `nondisjoint` |
| Modalità feature | `clip` |
| Epoche | 200 |
| Microbatch | 512 |
| Accumulo gradienti | 4 microbatch, batch effettivo 2048 |
| Ottimizzatore | AdamW, LR massima `2e-5`, weight decay `0.01` |
| Scheduler | OneCycleLR, aggiornato dopo ogni optimizer step |
| Gradient clipping | `1.0`, sempre attivo |
| Seed | 42, sempre attivo |
| Best model | `val_auc`; disponibili `val_accuracy` e `val_loss` |
| Early stopping | disabilitato; attivabile con patience |
| Resume | carica soltanto i pesi; optimizer, scheduler e history nuovi |

## Flag CLI

| Flag | Default | Cosa fa |
|---|---|---|
| `-h`, `--help` | — | Mostra guida dei comandi e termina. |
| `--classic` | disabilitato | Usa immagini e testi originali con ResNet-18 ImageNet e SentenceBERT secondo profilo paper. È mutuamente esclusivo con `--clip`. |
| `--clip` | abilitato | Usa embedding FashionCLIP prodotti da `precompute_embeddings`. È mutuamente esclusivo con `--classic`. |
| `--variant` | `nondisjoint` | Seleziona variante Polyvore. Valori ammessi: `disjoint`, `nondisjoint`. |
| `--embedding-root` | `precomputed_embeddings/patrickjohncyh-fashion-clip` | In modalità `clip`, indica root delle cache embedding; training aggiunge automaticamente `<variant>/<split>`. Ignorato da `classic`. |
| `--checkpoint-dir` | `checkpoints/<variant>/cp_<mode>` | Indica directory di configurazione, checkpoint e grafici. Deve non contenere già un run. |
| `--cache-dir` | `None` | Imposta directory cache usata da Hugging Face per dataset e annotazioni. |
| `--epochs` | `200` | Imposta numero massimo di epoche. |
| `--batch-size` | `512` | Imposta numero di outfit per microbatch. |
| `--gradient-accumulation-steps` | `4` | Accumula gradienti per questo numero di microbatch prima di ogni optimizer step. |
| `--learning-rate` | `2e-5` | Imposta LR massimo di OneCycleLR e LR configurato per AdamW. |
| `--weight-decay` | `0.01` | Imposta weight decay di AdamW. |
| `--focal-alpha` | `0.5` | Imposta bilanciamento tra classi della Focal Loss. |
| `--focal-gamma` | `2.0` | Imposta attenuazione degli esempi facili nella Focal Loss. |
| `--seed` | `42` | Imposta seed di Python, NumPy, PyTorch e DataLoader. |
| `--best-metric` | `val_auc` | Seleziona metrica del best model: `val_auc`, `val_accuracy` o `val_loss`. |
| `--early-stopping-patience` | `None` | Attiva early stopping e indica quante epoche senza miglioramento attendere. Se omesso, resta disabilitato. |
| `--early-stopping-min-delta` | `0.0` | Imposta miglioramento minimo richiesto per aggiornare best model e azzerare patience. |
| `--num-workers` | `0` | Imposta processi DataLoader. Con `0`, caricamento avviene nel processo principale. |
| `--pin-memory` | disabilitato | Abilita pinned memory nel DataLoader; flag booleano, utile con CUDA. |
| `--device` | `auto` | Seleziona device PyTorch. `auto` preferisce CUDA, poi MPS, poi CPU. Accetta anche valori espliciti come `cuda:0` o `cpu`. |
| `--log-every` | `10` | Scrive avanzamento in console ogni N microbatch e sempre all'ultimo microbatch. |
| `--resume` | `None` | Carica soltanto pesi da checkpoint. Optimizer, scheduler, epoche e history ripartono da zero. |
| `--token` | token locale Hugging Face | Usa token esplicito per scaricare risorse. È mutuamente esclusivo con `--no-token`. |
| `--no-token` | disabilitato | Forza accesso Hugging Face senza autenticazione. È mutuamente esclusivo con `--token`. |

## Preparazione CLIP

Solo `--clip` richiede cache separate per train e validation:

```powershell
python -m scripts.precompute_embeddings --variant nondisjoint --split train
python -m scripts.precompute_embeddings --variant nondisjoint --split validation
```

## Avvio

Versione classic del paper, senza precomputazione:

```powershell
python -m training.CP.train_cp --classic
```

Versione FashionCLIP con embedding precomputati:

```powershell
python -m training.CP.train_cp --clip
```

Esempio con early stopping e directory dedicata:

```powershell
python -m training.CP.train_cp `
  --clip `
  --early-stopping-patience 10 `
  --best-metric val_auc `
  --checkpoint-dir checkpoints/nondisjoint/esperimento_01
```

Resume dei soli pesi in un nuovo run:

```powershell
python -m training.CP.train_cp `
  --clip `
  --resume checkpoints/nondisjoint/esperimento_01/best.pt `
  --checkpoint-dir checkpoints/nondisjoint/esperimento_02
```

Il token Hugging Face non viene mai scritto nella configurazione. Si può usare
la sessione creata da `hf auth login`, oppure passare `--token`.

## Artefatti

Ogni run mantiene la struttura già usata sotto `checkpoints`:

```text
checkpoints/nondisjoint/esperimento_01/
  config.json
  best.pt
  epochs/
    cp_epoch_001.pt
    cp_epoch_002.pt
  plots/
    cp_loss_epoch_001.png
    cp_accuracy_epoch_001.png
    cp_auc_epoch_001.png
    cp_validation_accuracy_auc_epoch_001.png
```

Ogni checkpoint contiene pesi, configurazione, metriche, selezione best e
history cumulativa, inclusa modalità feature. Non contiene stato di optimizer
o scheduler, coerentemente con resume dei soli pesi. Resume richiede stessa
modalità e stessa architettura del checkpoint. I grafici vengono rigenerati
cumulativamente dopo ogni epoca. Una directory contenente già checkpoint non
viene sovrascritta.
