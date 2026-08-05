# Serie di training CP

Guida alle serie CP: training progressivo dipendente ed esperimenti end-to-end
indipendenti sullo split `nondisjoint`.

- [Training generale](../README.md)
- [Training CP](../cp/README.md)

## Indice

- [Serie progressiva archiviata](#serie-progressiva-archiviata)
- [Serie progressiva training_series](#serie-progressiva-training_series)
  - [Fasi progressive](#fasi-progressive)
  - [Iperparametri completi](#iperparametri-completi)
  - [Uso della serie progressiva](#uso-della-serie-progressiva)
  - [Fase 05: rifinitura dal best della fase 4](#fase-05-rifinitura-dal-best-della-fase-4)
- [Serie end-to-end nondisjoint](#serie-end-to-end-nondisjoint)
  - [Stage della nuova serie](#stage-della-nuova-serie)
  - [Iperparametri completi stage 1-5](#iperparametri-completi-stage-1-5)
  - [Iperparametri completi stage 6-10](#iperparametri-completi-stage-6-10)
  - [Uso della serie](#uso-della-serie)
  - [Artefatti prodotti](#artefatti-prodotti)
  - [Ripresa dopo un blocco](#ripresa-dopo-un-blocco)

## Serie progressiva archiviata

Runner rimosso. Configurazione reale e risultati restano in
[`checkpoints/run_training_series/README.md`](../../checkpoints/run_training_series/README.md).

## Serie progressiva `training_series`

`run_training_series.py` esegue warm-up e quattro fasi dipendenti. Ogni fase carica
il `best.pt` della precedente e sblocca gradualmente la ResNet-18:

```text
01_warmup -> 02_fc_only -> 03_fc_and_layer4 -> 04_full_backbone -> 05_full_refine
```

Warm-up usa solo strumenti esistenti: 2 epoche `fc_only` con LR dieci volte più
basso, poi una nuova fase `fc_only` con LR nominale. Non è una rampa
step-by-step: è un warm-up a fase, con optimizer nuovo al passaggio. I blocchi
convoluzionali restano congelati; Transformer contestuale, FC visuale e testa CP
restano allenabili.

Le fasi successive usano cosine decay più lunghi e minimi proporzionali al LR
iniziale. In questo modo i gruppi appena sbloccati non raggiungono un LR quasi
nullo dopo poche epoche. Le fasi full usano un unico LR medio `5e-7` per tutti
i blocchi del backbone; `05_full_refine` riavvia optimizer e cosine dal best di
fase 4.

### Fasi progressive

| Fase | Sorgente | Parte ResNet allenabile | Obiettivo |
|---|---|---|---|
| `01_warmup` | pesi preaddestrati | FC finale | stabilizzare Transformer, testa CP e proiezione visuale con LR basso |
| `02_fc_only` | `01_warmup/best.pt` | FC finale | adattare la proiezione visuale senza esaurire prematuramente il LR della FC |
| `03_fc_and_layer4` | `02_fc_only/best.pt` | FC + `layer4` | specializzare le feature visuali di alto livello con un cosine più lungo |
| `04_full_backbone` | `03_fc_and_layer4/best.pt` | intera ResNet | rifinitura globale con LR medio uniforme e tempo per stabilizzare le BatchNorm |
| `05_full_refine` | `04_full_backbone/best.pt` | intera ResNet | sfruttare capacità residua con un nuovo optimizer e un nuovo cosine |

### Iperparametri completi

| Iperparametro | `01_warmup` | `02_fc_only` | `03_fc_and_layer4` | `04_full_backbone` | `05_full_refine` |
|---|---|---|---|---|---|
| CLI | `training.cp.train_cp` | `training.cp.fine_tune_cp` | `training.cp.fine_tune_cp` | `training.cp.fine_tune_cp` | `training.cp.fine_tune_cp` |
| Sorgente pesi | ResNet-18 ImageNet + SentenceBERT | `01_warmup/best.pt` | `02_fc_only/best.pt` | `03_fc_and_layer4/best.pt` | `04_full_backbone/best.pt` |
| Dataset | `mvasil/polyvore-outfits` | `mvasil/polyvore-outfits` | `mvasil/polyvore-outfits` | `mvasil/polyvore-outfits` | `mvasil/polyvore-outfits` |
| Variante predefinita | `nondisjoint` | `nondisjoint` | `nondisjoint` | `nondisjoint` | `nondisjoint` |
| Epoche massime | 2 | 8 aggiuntive | 8 aggiuntive | 8 aggiuntive | 6 aggiuntive |
| Batch size | 50 | 50 | 50 | 50 | 50 |
| Modalità ResNet | `fc_only` | `fc_only` | `fc_and_layer4` | `full` | `full` |
| Blocchi ResNet allenabili | FC | FC | FC + `layer4` | intera ResNet | intera ResNet |
| Transformer contestuale | allenabile | allenabile | allenabile | allenabile | allenabile |
| SentenceBERT | congelato | congelato | congelato | congelato | congelato |
| Modello testuale | `all-MiniLM-L6-v2` | `all-MiniLM-L6-v2` | `all-MiniLM-L6-v2` | `all-MiniLM-L6-v2` | `all-MiniLM-L6-v2` |
| Dropout | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` |
| Normalizzazione Transformer | post-norm | post-norm ereditata | post-norm ereditata | post-norm ereditata | post-norm ereditata |
| Loss | Binary Focal Loss | Binary Focal Loss | Binary Focal Loss | Binary Focal Loss | Binary Focal Loss |
| Focal alpha / gamma | `0.5` / `2.0` | `0.5` / `2.0` | `0.5` / `2.0` | `0.5` / `2.0` | `0.5` / `2.0` |
| Optimizer | Adam | AdamW | AdamW | AdamW | AdamW |
| LR task/base | `1e-6` | `1e-5` | `5e-6` | `2e-6` | `2e-6` |
| LR Transformer | `1e-6`, gruppo base | `1e-5` | `5e-6` | `2e-6` | `2e-6` |
| LR FC ResNet | `3e-6` | `3e-5` | `1e-5` | `5e-6` | `5e-6` |
| LR feature ResNet | non applicabile | non applicabile | `1e-6` | `5e-7`, uniforme | `5e-7`, uniforme |
| Weight decay | `1e-4` | `1e-4` | `1e-4` | `1e-4` | `1e-4` |
| Scheduler task/base | nessuno | nessuno | cosine | cosine | nuovo cosine |
| Scheduler Transformer | nessuno, gruppo base | nessuno | cosine | cosine | nuovo cosine |
| Scheduler FC ResNet | nessuno | cosine | cosine | cosine | nuovo cosine |
| Scheduler feature ResNet | non applicabile | non applicabile | cosine | cosine | nuovo cosine |
| LR minimo task/base | non applicabile | non applicabile | `1e-6` | `5e-7` | `5e-7` |
| LR minimo Transformer | non applicabile | non applicabile | `5e-7` | `2e-7` | `2e-7` |
| LR minimo FC ResNet | non applicabile | `3e-6` | `1e-6` | `5e-7` | `5e-7` |
| LR minimo feature ResNet | non applicabile | non applicabile | `1e-7` | `5e-8` | `5e-8` |
| Gradient clipping | disabilitato | disabilitato | disabilitato | disabilitato | disabilitato |
| Metrica best checkpoint | validation ROC AUC | validation ROC AUC | validation ROC AUC | validation ROC AUC | validation ROC AUC |
| Early stopping | patience 2, delta `1e-4` | patience 3, delta `1e-4` | patience 3, delta `1e-4` | patience 3, delta `1e-4` | patience 3, delta `1e-4` |
| Seed | 42 | 42 | 42 | 42 | 42 |
| Checkpoint | best + uno per epoca | best + uno per epoca | best + uno per epoca | best + uno per epoca | best + uno per epoca |

Le epoche sono budget massimi. Fase seguente usa sempre miglior validation AUC,
non ultima epoca. Se fase 5 peggiora, mantenere `04_full_backbone/best.pt`.
`--min-learning-rate` è condiviso dai gruppi task/base e FC ResNet; nella fase 2
agisce solo sulla FC perché task e Transformer non usano scheduler.

### Uso della serie progressiva

```powershell
# Mostra comandi e dipendenze
python -m training.run_trianing_series.run_training_series --dry-run

# Esegue tutte le fasi
python -m training.run_trianing_series.run_training_series

# Esegue fasi 2, 3, 4 e 5; warm-up deve già esistere
python -m training.run_trianing_series.run_training_series --phases 2 3 4 5

# Esegue solo il refine; fase 4 deve già esistere
python -m training.run_trianing_series.run_training_series --phases 5
```

Output predefinito: `checkpoints/training_series`. Runner rifiuta directory di
fase contenenti checkpoint, evitando sovrascritture accidentali.

### Fase 05: rifinitura dal best della fase 4

`05_full_refine` è parte della serie. Carica automaticamente
`04_full_backbone/best.pt`, crea optimizer e cosine nuovi ed esegue fino a 6
epoche full-backbone con LR uniforme `5e-7`. Una serie completa la esegue
automaticamente; con fasi 1–4 già presenti, avviarla da sola:

```text
python -m training.run_trianing_series.run_training_series --phases 5
```

La directory `05_full_refine` deve essere nuova o non contenere checkpoint. Con
il checkpoint attuale di fase 4 all'epoca 16, produrrà le epoche 17–22.

## Serie end-to-end nondisjoint

`run_end_to_end_series.py` esegue dieci training indipendenti sullo split
`nondisjoint`. Tutti ripartono dai pesi preaddestrati; nessuno eredita pesi da
uno stage precedente. Questa sezione descrive anche stage non ancora eseguiti.
README dentro `checkpoints` documentano invece solo artefatti già prodotti.

### Stage della nuova serie

| Stage | Variazione rispetto al baseline | Dropout | Weight decay | Focal alpha | StepLR step size |
|---|---|---:|---:|---:|---:|
| `01_paper_standard_defaults` | baseline paper + default standard | `0.1` | `0.0` | `0.25` | 10 |
| `02_dropout_0` | dropout disabilitato | `0.0` | `0.0` | `0.25` | 10 |
| `03_dropout_0_weight_decay_1e4` | dropout disabilitato + weight decay | `0.0` | `1e-4` | `0.25` | 10 |
| `04_focal_alpha_05` | dropout disabilitato + alpha bilanciato | `0.0` | `0.0` | `0.5` | 10 |
| `05_weight_decay_1e4_focal_alpha_05` | dropout disabilitato + weight decay + alpha bilanciato | `0.0` | `1e-4` | `0.5` | 10 |
| `06_step_lr_3_standard_defaults` | baseline con riduzione LR rapida | `0.1` | `0.0` | `0.25` | 3 |
| `07_step_lr_3_dropout_0` | dropout disabilitato + riduzione LR rapida | `0.0` | `0.0` | `0.25` | 3 |
| `08_step_lr_3_dropout_0_weight_decay_1e3` | dropout disabilitato + weight decay medio + riduzione LR rapida | `0.0` | `1e-3` | `0.25` | 3 |
| `09_step_lr_3_dropout_0_focal_alpha_05` | dropout disabilitato + alpha bilanciato + riduzione LR rapida | `0.0` | `0.0` | `0.5` | 3 |
| `10_step_lr_3_dropout_0_weight_decay_1e3_focal_alpha_05` | dropout disabilitato + weight decay medio + alpha bilanciato + riduzione LR rapida | `0.0` | `1e-3` | `0.5` | 3 |

Negli stage 8 e 10, livello medio più aggressivo significa weight decay
`1e-3`: dieci volte `1e-4`. Weight decay resta costante dall'inizio alla fine;
StepLR modifica solo learning rate.

### Iperparametri completi stage 1-5

| Iperparametro | `01_paper_standard_defaults` | `02_dropout_0` | `03_dropout_0_weight_decay_1e4` | `04_focal_alpha_05` | `05_weight_decay_1e4_focal_alpha_05` |
|---|---|---|---|---|---|
| CLI | `training.cp.train_cp` | `training.cp.train_cp` | `training.cp.train_cp` | `training.cp.train_cp` | `training.cp.train_cp` |
| Sorgente pesi | ResNet-18 ImageNet + SentenceBERT | ResNet-18 ImageNet + SentenceBERT | ResNet-18 ImageNet + SentenceBERT | ResNet-18 ImageNet + SentenceBERT | ResNet-18 ImageNet + SentenceBERT |
| Dataset | `mvasil/polyvore-outfits` | `mvasil/polyvore-outfits` | `mvasil/polyvore-outfits` | `mvasil/polyvore-outfits` | `mvasil/polyvore-outfits` |
| Variante dataset | `nondisjoint` | `nondisjoint` | `nondisjoint` | `nondisjoint` | `nondisjoint` |
| Epoche massime | 30 | 30 | 30 | 30 | 30 |
| Batch size | 50 | 50 | 50 | 50 | 50 |
| Modalità ResNet | `full` | `full` | `full` | `full` | `full` |
| Blocchi ResNet allenabili | intera ResNet, FC e BatchNorm | intera ResNet, FC e BatchNorm | intera ResNet, FC e BatchNorm | intera ResNet, FC e BatchNorm | intera ResNet, FC e BatchNorm |
| SentenceBERT | congelato | congelato | congelato | congelato | congelato |
| Modello testuale | `all-MiniLM-L6-v2` | `all-MiniLM-L6-v2` | `all-MiniLM-L6-v2` | `all-MiniLM-L6-v2` | `all-MiniLM-L6-v2` |
| Image embedding | 64 | 64 | 64 | 64 | 64 |
| Text embedding | 64 | 64 | 64 | 64 | 64 |
| Item embedding / `d_model` | 128 | 128 | 128 | 128 | 128 |
| Layer Transformer | 6 | 6 | 6 | 6 | 6 |
| Teste di attenzione | 16 | 16 | 16 | 16 | 16 |
| Dimensione feed-forward | 512 | 512 | 512 | 512 | 512 |
| Dropout | `0.1` | `0.0` | `0.0` | `0.0` | `0.0` |
| Normalizzazione Transformer | post-norm | post-norm | post-norm | post-norm | post-norm |
| Loss | Binary Focal Loss | Binary Focal Loss | Binary Focal Loss | Binary Focal Loss | Binary Focal Loss |
| Focal alpha | `0.25` | `0.25` | `0.25` | `0.5` | `0.5` |
| Focal gamma | `2.0` | `2.0` | `2.0` | `2.0` | `2.0` |
| Optimizer | Adam | Adam | Adam | Adam | Adam |
| Adam beta1 / beta2 / epsilon | `0.9` / `0.999` / `1e-8` | `0.9` / `0.999` / `1e-8` | `0.9` / `0.999` / `1e-8` | `0.9` / `0.999` / `1e-8` | `0.9` / `0.999` / `1e-8` |
| LR task e ResNet | `1e-5` | `1e-5` | `1e-5` | `1e-5` | `1e-5` |
| Weight decay | `0.0` | `0.0` | `1e-4` | `0.0` | `1e-4` |
| Scheduler | StepLR | StepLR | StepLR | StepLR | StepLR |
| Step size / gamma | `10` / `0.5` | `10` / `0.5` | `10` / `0.5` | `10` / `0.5` | `10` / `0.5` |
| Metrica best checkpoint | validation ROC AUC | validation ROC AUC | validation ROC AUC | validation ROC AUC | validation ROC AUC |
| Early stopping | patience 3, delta `1e-4` | patience 3, delta `1e-4` | patience 3, delta `1e-4` | patience 3, delta `1e-4` | patience 3, delta `1e-4` |
| Gradient clipping | disabilitato | disabilitato | disabilitato | disabilitato | disabilitato |
| Seed | 42 | 42 | 42 | 42 | 42 |
| DataLoader workers | 0 | 0 | 0 | 0 | 0 |
| Device | CUDA se disponibile, altrimenti CPU | CUDA se disponibile, altrimenti CPU | CUDA se disponibile, altrimenti CPU | CUDA se disponibile, altrimenti CPU | CUDA se disponibile, altrimenti CPU |
| Log batch | ogni 50 batch | ogni 50 batch | ogni 50 batch | ogni 50 batch | ogni 50 batch |
| Grafici | abilitati | abilitati | abilitati | abilitati | abilitati |
| Checkpoint | best + uno per epoca | best + uno per epoca | best + uno per epoca | best + uno per epoca | best + uno per epoca |

### Iperparametri completi stage 6-10

| Iperparametro | `06_step_lr_3_standard_defaults` | `07_step_lr_3_dropout_0` | `08_step_lr_3_dropout_0_weight_decay_1e3` | `09_step_lr_3_dropout_0_focal_alpha_05` | `10_step_lr_3_dropout_0_weight_decay_1e3_focal_alpha_05` |
|---|---|---|---|---|---|
| CLI | `training.cp.train_cp` | `training.cp.train_cp` | `training.cp.train_cp` | `training.cp.train_cp` | `training.cp.train_cp` |
| Sorgente pesi | ResNet-18 ImageNet + SentenceBERT | ResNet-18 ImageNet + SentenceBERT | ResNet-18 ImageNet + SentenceBERT | ResNet-18 ImageNet + SentenceBERT | ResNet-18 ImageNet + SentenceBERT |
| Dataset | `mvasil/polyvore-outfits` | `mvasil/polyvore-outfits` | `mvasil/polyvore-outfits` | `mvasil/polyvore-outfits` | `mvasil/polyvore-outfits` |
| Variante dataset | `nondisjoint` | `nondisjoint` | `nondisjoint` | `nondisjoint` | `nondisjoint` |
| Epoche massime | 30 | 30 | 30 | 30 | 30 |
| Batch size | 50 | 50 | 50 | 50 | 50 |
| Modalità ResNet | `full` | `full` | `full` | `full` | `full` |
| Blocchi ResNet allenabili | intera ResNet, FC e BatchNorm | intera ResNet, FC e BatchNorm | intera ResNet, FC e BatchNorm | intera ResNet, FC e BatchNorm | intera ResNet, FC e BatchNorm |
| SentenceBERT | congelato | congelato | congelato | congelato | congelato |
| Modello testuale | `all-MiniLM-L6-v2` | `all-MiniLM-L6-v2` | `all-MiniLM-L6-v2` | `all-MiniLM-L6-v2` | `all-MiniLM-L6-v2` |
| Image embedding | 64 | 64 | 64 | 64 | 64 |
| Text embedding | 64 | 64 | 64 | 64 | 64 |
| Item embedding / `d_model` | 128 | 128 | 128 | 128 | 128 |
| Layer Transformer | 6 | 6 | 6 | 6 | 6 |
| Teste di attenzione | 16 | 16 | 16 | 16 | 16 |
| Dimensione feed-forward | 512 | 512 | 512 | 512 | 512 |
| Dropout | `0.1` | `0.0` | `0.0` | `0.0` | `0.0` |
| Normalizzazione Transformer | post-norm | post-norm | post-norm | post-norm | post-norm |
| Loss | Binary Focal Loss | Binary Focal Loss | Binary Focal Loss | Binary Focal Loss | Binary Focal Loss |
| Focal alpha | `0.25` | `0.25` | `0.25` | `0.5` | `0.5` |
| Focal gamma | `2.0` | `2.0` | `2.0` | `2.0` | `2.0` |
| Optimizer | Adam | Adam | Adam | Adam | Adam |
| Adam beta1 / beta2 / epsilon | `0.9` / `0.999` / `1e-8` | `0.9` / `0.999` / `1e-8` | `0.9` / `0.999` / `1e-8` | `0.9` / `0.999` / `1e-8` | `0.9` / `0.999` / `1e-8` |
| LR iniziale task e ResNet | `1e-5` | `1e-5` | `1e-5` | `1e-5` | `1e-5` |
| Weight decay | `0.0` | `0.0` | `1e-3` | `0.0` | `1e-3` |
| Scheduler | StepLR | StepLR | StepLR | StepLR | StepLR |
| Step size / gamma | `3` / `0.5` | `3` / `0.5` | `3` / `0.5` | `3` / `0.5` | `3` / `0.5` |
| Metrica best checkpoint | validation ROC AUC | validation ROC AUC | validation ROC AUC | validation ROC AUC | validation ROC AUC |
| Early stopping | patience 3, delta `1e-4` | patience 3, delta `1e-4` | patience 3, delta `1e-4` | patience 3, delta `1e-4` | patience 3, delta `1e-4` |
| Gradient clipping | disabilitato | disabilitato | disabilitato | disabilitato | disabilitato |
| Seed | 42 | 42 | 42 | 42 | 42 |
| DataLoader workers | 0 | 0 | 0 | 0 | 0 |
| Device | CUDA se disponibile, altrimenti CPU | CUDA se disponibile, altrimenti CPU | CUDA se disponibile, altrimenti CPU | CUDA se disponibile, altrimenti CPU | CUDA se disponibile, altrimenti CPU |
| Log batch | ogni 50 batch | ogni 50 batch | ogni 50 batch | ogni 50 batch | ogni 50 batch |
| Grafici | abilitati | abilitati | abilitati | abilitati | abilitati |
| Checkpoint | best + uno per epoca | best + uno per epoca | best + uno per epoca | best + uno per epoca | best + uno per epoca |

Con `step_size=3` e gamma `0.5`, LR vale `1e-5` nelle epoche 1-3 e
`5e-6` dalla quarta alla sesta; viene poi dimezzato ancora ogni tre epoche.

### Uso della serie

Eseguire i comandi dalla root del progetto, con ambiente virtuale attivo.

Mostrare opzioni disponibili:

```powershell
python -m training.run_trianing_series.run_end_to_end_series --help
```

Mostrare tutti i comandi senza avviare training:

```powershell
python -m training.run_trianing_series.run_end_to_end_series --dry-run
```

Avviare tutti i dieci stage in una root nuova e vuota:

```powershell
python -m training.run_trianing_series.run_end_to_end_series `
  --output-root checkpoints\nondisjoint_v2
```

Nella root predefinita `checkpoints\nondisjoint`, stage 1 risulta già eseguito.
Avviare quindi solo stage mancanti:

```powershell
python -m training.run_trianing_series.run_end_to_end_series `
  --stages 2 3 4 5 6 7 8 9 10
```

Avviare un singolo stage o un sottoinsieme:

```powershell
# Solo stage 8
python -m training.run_trianing_series.run_end_to_end_series --stages 8

# Stage 6 e 10, eseguiti in questo ordine
python -m training.run_trianing_series.run_end_to_end_series --stages 6 10
```

Opzioni comuni vengono applicate a tutti gli stage selezionati:

```powershell
python -m training.run_trianing_series.run_end_to_end_series `
  --stages 2 3 4 5 6 7 8 9 10 `
  --device cuda `
  --workers 4 `
  --cache-dir data\huggingface
```

Cambiare `--epochs`, `--batch-size`, `--text-model`, early stopping o altre
opzioni rende configurazione diversa da quella riportata nelle tabelle.

Stage sono indipendenti. Runner li esegue in ordine e ogni stage riparte dai
pesi preaddestrati, non dal best dello stage precedente.

### Artefatti prodotti

Ogni stage salva file nella propria directory:

```text
checkpoints/nondisjoint/03_dropout_0_weight_decay_1e4/
├── best.pt
├── epochs/
│   ├── cp_epoch_001.pt
│   └── ...
└── plots/
```

Runner rifiuta uno stage selezionato se sua directory contiene già file `.pt`.
Protezione evita sovrascrittura accidentale. Per continuare serie, selezionare
solo stage non ancora iniziati. Per continuare uno stage parziale, usare resume
descritto sotto.

### Ripresa dopo un blocco

#### Blocco tra due stage

Se stage 2 è completo e blocco avviene prima di stage 3, rilanciare solo stage
rimanenti:

```powershell
python -m training.run_trianing_series.run_end_to_end_series `
  --stages 3 4 5 6 7 8 9 10
```

Non includere stage già completati: runner trova loro checkpoint e interrompe
per protezione.

#### Blocco durante uno stage

Usare checkpoint epoca più recente, non `best.pt`: conserva massimo avanzamento.
Esempio per stage 3:

generare comando esatto dello stage:

```powershell
python -m training.run_trianing_series.run_end_to_end_series `
  --dry-run `
  --stages 3
```

Copiare riga mostrata dopo `command=` e aggiungere in fondo checkpoint trovato:

```powershell
--resume checkpoints\nondisjoint\03_dropout_0_weight_decay_1e4\epochs\cp_epoch_006.pt
```

Comando copiato contiene già configurazione completa e directory originali di
`best.pt`, `epochs` e `plots`. `training.cp.train_cp` ripristina modello,
optimizer, scheduler, history, migliore metrica e RNG.

Non usare direttamente runner sullo stage parziale: protezione contro
sovrascrittura lo rifiuta. Terminato resume dello stage 3, continuare serie:

```powershell
python -m training.run_trianing_series.run_end_to_end_series `
  --stages 4 5 6 7 8 9 10
```

`--epochs` indica epoca finale totale, non epoche aggiuntive. Con checkpoint
epoca 6 e `--epochs 30`, training riparte da 7 e termina al massimo a 30.

Se blocco avviene prima del primo checkpoint, rilanciare normalmente quello
stage. Se manca cartella `epochs`, usare `best.pt`, accettando ripartenza dalla
sua epoca invece che dall'ultima epoca eseguita.
