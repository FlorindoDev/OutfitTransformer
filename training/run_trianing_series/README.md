# Serie di training CP

Runner dei quattro esperimenti CP eseguiti in successione.

- [Training generale](../README.md)
- [Training CP](../cp/README.md)

## Indice

- [Panoramica](#panoramica)
- [Iperparametri completi dei quattro training](#iperparametri-completi-dei-quattro-training)
- [Avvio della serie](#avvio-della-serie)
- [Riprendere la serie da un checkpoint](#riprendere-la-serie-da-un-checkpoint)
  - [Caso base: stage 2 completato](#caso-base-stage-2-completato)
  - [Esempio con flag opzionali](#esempio-con-flag-opzionali)
  - [Caso stage 2 interrotto](#caso-stage-2-interrotto)
  - [Caso stage 3 o 4 interrotto](#caso-stage-3-o-4-interrotto)

## Panoramica

`run_training_series.py` richiama in successione le CLI esistenti; non duplica
training, loader o checkpointing:

1. `01_paper_end_to_end`: CP end-to-end con gli iperparametri dichiarati nel paper;
2. `02_fc_only_base`: base con sola FC ResNet;
3. `03_layer4_plateau`: `fc_and_layer4`, LR backbone `1e-6`, early stopping AUC;
4. `04_full_low_lr`: full per 4 epoche massime, LR backbone `3e-7`.

| Stage | ResNet | Epoche max | LR task | LR backbone | Weight decay | Scheduler | Early stopping |
|---|---|---:|---:|---:|---:|---|---|
| `01_paper_end_to_end` | `full` | 30 | `1e-5` | `1e-5` | `0.0` | StepLR, ogni 10 epoche × `0.5` | Disabilitato |
| `02_fc_only_base` | `fc_only` | 12 | `1e-5` | Backbone congelato | `1e-4` | StepLR, ogni 10 epoche × `0.5` | patience 3, delta `1e-4` |
| `03_layer4_plateau` | `fc_and_layer4` | 30 aggiuntive | `1e-5` | `1e-6` | `1e-4` | Cosine, `T_max=30`, minimo `0` | patience 4, delta `1e-4` |
| `04_full_low_lr` | `full` | 4 aggiuntive | `3e-6` | `3e-7` | `1e-4` | Cosine, `T_max=4`, minimo `0` | patience 2, delta `1e-4` |

## Iperparametri completi dei quattro training

I valori riportati sono i default effettivi di `run_training_series.py`.
Quelli indicati come ereditati provengono dal checkpoint dello stage
precedente.

| Iperparametro | `01_paper_end_to_end` | `02_fc_only_base` | `03_layer4_plateau` | `04_full_low_lr` |
|---|---|---|---|---|
| CLI | `training.cp.train_cp` | `training.cp.train_cp` | `training.cp.fine_tune_cp` | `training.cp.fine_tune_cp` |
| Sorgente pesi | ResNet-18 ImageNet e SentenceBERT preaddestrati; componenti CP inizializzati dal modello | ResNet-18 ImageNet e SentenceBERT preaddestrati; componenti CP inizializzati dal modello | `02_fc_only_base/best.pt` | `03_layer4_plateau/best.pt` |
| Dataset | `mvasil/polyvore-outfits` | `mvasil/polyvore-outfits` | `mvasil/polyvore-outfits` | `mvasil/polyvore-outfits` |
| Variante dataset | `disjoint` | `disjoint` | `disjoint` | `disjoint` |
| Epoche massime | 30 | 12 | 30 aggiuntive | 4 aggiuntive |
| Batch size | 50 | 50 | 50 | 50 |
| Modalità ResNet | `full` | `fc_only` | `fc_and_layer4` | `full` |
| Blocchi ResNet allenabili | intera ResNet, FC e BatchNorm | solo FC `512 → 64`; backbone e BatchNorm congelati | `layer4`, relative BatchNorm e FC | intera ResNet, FC e BatchNorm |
| Componenti CP allenabili | Transformer, token `OUTFIT`, proiezione testo e classificatore | Transformer, token `OUTFIT`, proiezione testo e classificatore | Transformer, token `OUTFIT`, proiezione testo e classificatore | Transformer, token `OUTFIT`, proiezione testo e classificatore |
| SentenceBERT | congelato | congelato | congelato, ereditato | congelato, ereditato |
| Modello testuale | `sentence-transformers/all-MiniLM-L6-v2` | `sentence-transformers/all-MiniLM-L6-v2` | ereditato dallo stage 2 | ereditato dallo stage 3 |
| Image embedding | 64 | 64 | 64, ereditato | 64, ereditato |
| Text embedding | 64 | 64 | 64, ereditato | 64, ereditato |
| Item embedding / `d_model` | 128 | 128 | 128, ereditato | 128, ereditato |
| Layer Transformer | 6 | 6 | 6, ereditati | 6, ereditati |
| Teste di attenzione | 16 | 16 | 16, ereditate | 16, ereditate |
| Dimensione feed-forward | 512 | 512 | 512, ereditata | 512, ereditata |
| Dropout | `0.1` | `0.1` | `0.1`, ereditato | `0.1`, ereditato |
| Loss | Binary Focal Loss | Binary Focal Loss | Binary Focal Loss | Binary Focal Loss |
| Focal alpha | `0.5` | `0.5` | `0.5` | `0.5` |
| Focal gamma | `1.0` | `1.0` | `1.0` | `1.0` |
| Optimizer | Adam | Adam | Adam | Adam |
| Adam beta1 | `0.9` | `0.9` | `0.9` | `0.9` |
| Adam beta2 | `0.999` | `0.999` | `0.999` | `0.999` |
| Adam epsilon | `1e-8` | `1e-8` | `1e-8` | `1e-8` |
| LR task | `1e-5` | `1e-5` | `1e-5` | `3e-6` |
| LR backbone ResNet | `1e-5` | non applicabile: congelato | `1e-6` | `3e-7` |
| Weight decay | `0.0` | `1e-4` | `1e-4` | `1e-4` |
| Scheduler | StepLR | StepLR | CosineAnnealingLR | CosineAnnealingLR |
| Step size | 10 | 10 | non applicabile | non applicabile |
| Gamma scheduler | `0.5` | `0.5` | non applicabile | non applicabile |
| `T_max` cosine | non applicabile | non applicabile | 30 | 4 |
| LR minimo cosine | non applicabile | non applicabile | `0.0` | `0.0` |
| Metrica best checkpoint | validation ROC AUC | validation ROC AUC | validation ROC AUC | validation ROC AUC |
| Early stopping | disabilitato | abilitato | abilitato | abilitato |
| Patience | non applicabile | 3 | 4 | 2 |
| `min_delta` | non applicabile | `1e-4` | `1e-4` | `1e-4` |
| Gradient clipping | norma massima `1.0` | norma massima `1.0` | norma massima `1.0` | norma massima `1.0` |
| Seed | 42 | 42 | 42 | 42 |
| DataLoader workers | 0 | 0 | 0 | 0 |
| Device | CUDA se disponibile, altrimenti CPU | CUDA se disponibile, altrimenti CPU | CUDA se disponibile, altrimenti CPU | CUDA se disponibile, altrimenti CPU |
| Log batch | ogni 50 batch | ogni 50 batch | ogni 50 batch | ogni 50 batch |
| Grafici | abilitati | abilitati | abilitati | abilitati |
| Checkpoint | best + uno per epoca | best + uno per epoca | best + uno per epoca | best + uno per epoca |

Lo stage 1 è un baseline indipendente. La catena progressiva usa invece
`02_fc_only_base → 03_layer4_plateau → 04_full_low_lr`.

## Avvio della serie

```powershell
python -m training.run_trianing_series.run_training_series
```

Anteprima senza allenare:

```powershell
python -m training.run_trianing_series.run_training_series --dry-run
```

Gli artefatti finiscono in `checkpoints/cp_training_series/<nome-stage>/`.
Directory contenenti checkpoint non vengono sovrascritte. Poiché il paper non
dichiara il numero di epoche, lo stage 1 usa 30 epoche per default,
modificabili con `--paper-epochs`.

## Riprendere la serie da un checkpoint

Esistono due operazioni diverse:

1. uno stage è completato e la serie deve partire dallo stage successivo:
   usare `run_training_series --start-stage`;
2. uno stage è stato interrotto durante il training: riprendere prima quello
   stage tramite la sua CLI, poi continuare la serie.

### Caso base: stage 2 completato

Se `02_fc_only_base/best.pt` esiste, questo è l'unico comando necessario per
eseguire gli stage 3 e 4:

```powershell
python -m training.run_trianing_series.run_training_series --start-stage 3
```

Lo script cerca automaticamente:

```text
checkpoints/cp_training_series/02_fc_only_base/best.pt
```

Non serve chiamare manualmente `fine_tune_cp`: lo stage 3 usa quel checkpoint,
poi lo stage 4 usa il `best.pt` prodotto dallo stage 3.

Se anche lo stage 3 è già completato, si può eseguire soltanto lo stage 4:

```powershell
python -m training.run_trianing_series.run_training_series --start-stage 4
```

### Esempio con flag opzionali

I flag seguenti non sono necessari con i default. Permettono di usare una serie
salvata altrove, ridurre il batch, scegliere GPU e worker, disabilitare i
grafici:

```powershell
python -m training.run_trianing_series.run_training_series `
  --output-root checkpoints\mia_serie `
  --start-stage 3 `
  --batch-size 8 `
  --device cuda:0 `
  --workers 4 `
  --no-plots
```

Se gli stage precedenti usano un `--output-root` personalizzato, ripassare lo
stesso valore non è opzionale: serve a trovare il relativo `best.pt`.

Directory degli stage da eseguire non devono contenere file `.pt`, perché lo
script evita sovrascritture.

Lo stage 1 non alimenta lo stage 2: è un baseline indipendente. Non esiste
quindi una ripartenza dello stage 2 dal checkpoint dello stage 1.

### Caso stage 2 interrotto

`--start-stage 3` funziona soltanto dopo avere completato lo stage 2. Se lo
stage 2 si è fermato all'epoca 7, prima bisogna riprenderlo. Questo esempio
contiene solo i flag necessari per mantenere modalità, best checkpoint,
checkpoint per epoca ed early stopping dello stage 2:

```powershell
python -m training.cp.train_cp `
  --resume checkpoints\cp_training_series\02_fc_only_base\epochs\cp_epoch_007.pt `
  --epochs 12 `
  --image-fine-tune-mode fc_only `
  --checkpoint checkpoints\cp_training_series\02_fc_only_base\best.pt `
  --checkpoint-dir checkpoints\cp_training_series\02_fc_only_base\epochs `
  --early-stopping-patience 3 `
  --early-stopping-min-delta 0.0001
```

Il resume ripristina modello, optimizer, scheduler, history e RNG. Quando lo
stage 2 termina, basta continuare con:

```powershell
python -m training.run_trianing_series.run_training_series --start-stage 3
```

### Caso stage 3 o 4 interrotto

Gli stage 3 e 4 usano `fine_tune_cp`. Questa CLI carica i pesi, ma crea
optimizer, scheduler, history e patience nuovi. Serve una directory diversa.
Esempio minimo per continuare lo stage 3:

```powershell
python -m training.cp.fine_tune_cp `
  --source-checkpoint checkpoints\cp_training_series\03_layer4_plateau\epochs\cp_epoch_020.pt `
  --output-dir checkpoints\cp_training_series\03_layer4_plateau_continued `
  --image-backbone-learning-rate 1e-6 `
  --scheduler cosine `
  --early-stopping-patience 4 `
  --early-stopping-min-delta 0.0001
```

Le 10 epoche aggiuntive, modalità `fc_and_layer4`, Adam, LR task `1e-5`,
weight decay `1e-4`, Focal Loss e best su validation AUC sono già i default,
quindi non serve ripeterli.

Questo caso è una nuova fase di fine-tuning, non un resume esatto.
Il relativo `best.pt` non viene agganciato automaticamente da
`run_training_series`: per usarlo nello stage successivo bisogna passarlo
esplicitamente a un nuovo comando `fine_tune_cp --source-checkpoint`.
