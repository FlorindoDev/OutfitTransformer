# Training Compatibility Prediction

Training CP supporta tre profili selezionabili con flag mutuamente esclusivi.
`--classic` mantiene le dimensioni storiche dei checkpoint, `--new-classic`
usa gli stessi encoder con rappresentazioni più ampie e `--clip` usa embedding
FashionCLIP precomputati. Transformer common, Transformer CP, token e testa di
classificazione vengono allenati in tutte le modalità.

## Modalità

| Flag | Feature degli item | Dimensione | Parti allenabili | Precomputazione |
|---|---|---:|---|---|
| `--classic` | ResNet-18 ImageNet + SentenceBERT | `64 + 64 = 128` | ResNet-18, proiezioni, Transformer e CP; backbone SentenceBERT congelato | Non richiesta |
| `--new-classic` | ResNet-18 ImageNet + SentenceBERT | `512 + 512 = 1024` | ResNet-18, proiezioni, Transformer e CP; backbone SentenceBERT congelato | Non richiesta |
| `--clip` | FashionCLIP visuale + testo | `512 + 512 = 1024` | Transformer e CP; tower FashionCLIP congelate | Richiesta per train e validation |

`classic` e `new_classic` condividono encoder, 6 layer, 16 teste, feed-forward
512, dropout `0.1` e post-norm. Cambia soltanto la dimensione di ogni modalità:
`classic` usa 64 feature per mantenere la forma dei checkpoint storici;
`new_classic` ne usa 512. `clip` mantiene 6 layer, 16 teste, feed-forward 2024,
dropout `0.3` e pre-norm.

## Configurazione predefinita

| Aspetto | Valore |
|---|---|
| Dataset | `polyvore`, subset `nondisjoint` |
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
| `--classic` | disabilitato | Usa immagini e testi originali con ResNet-18 ImageNet e SentenceBERT, proiettati a `64 + 64`. È mutuamente esclusivo con gli altri profili. |
| `--new-classic` | disabilitato | Usa la pipeline classic con proiezioni ampliate a `512 + 512`. È mutuamente esclusivo con gli altri profili. |
| `--clip` | abilitato | Usa embedding FashionCLIP prodotti da `precompute_embeddings`. È mutuamente esclusivo con gli altri profili. |
| `--dataset` | `polyvore` | Seleziona source registrata nell'API pubblica `data`. |
| `--subset` | `nondisjoint` | Seleziona subset della source. `--variant` resta alias compatibile. |
| `--embedding-root` | `precomputed_embeddings/patrickjohncyh-fashion-clip` | In modalità `clip`, indica root delle cache embedding; training aggiunge automaticamente `<subset>/<split>`. Ignorato da `classic` e `new_classic`. |
| `--dataset-root` | `datasets/polyvore-outfits` | Cerca qui dataset e annotazioni prima di usare cache Hugging Face o download. |
| `--checkpoint-dir` | `checkpoints/<subset>/cp_<mode>` | Indica directory di configurazione, checkpoint e grafici. Deve non contenere già un run. |
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
python -m scripts.precompute_embeddings --subset nondisjoint --split train
python -m scripts.precompute_embeddings --subset nondisjoint --split validation
```

Per tutte le modalità, risorse Polyvore seguono ordine locale, cache Hugging
Face, download. In `clip` vengono cercate soltanto annotazioni outfit/CP:
immagini e metadata non vengono caricati durante training.

Training usa soltanto `DatasetSource`, `DatasetRequest` e tipi pubblici di
`data`. Parser, download e struttura Polyvore restano confinati in
`data/polyvore`.

## Avvio

Versione classic del paper, senza precomputazione:

```powershell
python -m training.CP.train_cp --classic
```

Versione classic ampliata a 512 feature per modalità:

```powershell
python -m training.CP.train_cp --new-classic
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

Ogni profilo usa per default una directory distinta:
`cp_classic`, `cp_new_classic` oppure `cp_clip`. Struttura e contenuto degli
artefatti sono descritti nel [README generale del training](../README.md#checkpoint-e-monitoraggio).
