# Serie di training CP

Guida alla serie corrente di cinque esperimenti end-to-end sullo split
`nondisjoint`. Serie progressiva precedente archiviata accanto ai checkpoint.

- [Training generale](../README.md)
- [Training CP](../cp/README.md)

## Indice

- [Serie progressiva archiviata](#serie-progressiva-archiviata)
- [Serie end-to-end nondisjoint](#serie-end-to-end-nondisjoint)
  - [Stage della nuova serie](#stage-della-nuova-serie)
  - [Iperparametri completi della nuova serie](#iperparametri-completi-della-nuova-serie)
  - [Uso della serie](#uso-della-serie)
  - [Artefatti prodotti](#artefatti-prodotti)
  - [Ripresa dopo un blocco](#ripresa-dopo-un-blocco)

## Serie progressiva archiviata

Runner rimosso. Configurazione reale e risultati restano in
[`checkpoints/run_training_series/README.md`](../../checkpoints/run_training_series/README.md).

## Serie end-to-end nondisjoint

`run_end_to_end_series.py` esegue cinque training indipendenti sullo split
`nondisjoint`. Tutti ripartono dai pesi preaddestrati; nessuno eredita pesi da
uno stage precedente. Questa sezione descrive anche stage non ancora eseguiti.
README dentro `checkpoints` documentano invece solo artefatti già prodotti.

### Stage della nuova serie

| Stage | Variazione rispetto al baseline | Dropout | Weight decay | Focal alpha |
|---|---|---:|---:|---:|
| `01_paper_standard_defaults` | baseline paper + default standard | `0.1` | `0.0` | `0.25` |
| `02_dropout_0` | dropout disabilitato | `0.0` | `0.0` | `0.25` |
| `03_dropout_0_weight_decay_1e4` | dropout disabilitato + weight decay | `0.0` | `1e-4` | `0.25` |
| `04_focal_alpha_05` | alpha bilanciato | `0.1` | `0.0` | `0.5` |
| `05_weight_decay_1e4_focal_alpha_05` | weight decay + alpha bilanciato | `0.1` | `1e-4` | `0.5` |

### Iperparametri completi della nuova serie

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
| Dropout | `0.1` | `0.0` | `0.0` | `0.1` | `0.1` |
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

Avviare tutti i cinque stage in una root nuova e vuota:

```powershell
python -m training.run_trianing_series.run_end_to_end_series `
  --output-root checkpoints\nondisjoint_v2
```

Nella root predefinita `checkpoints\nondisjoint`, stage 1 risulta già eseguito.
Avviare quindi solo stage mancanti:

```powershell
python -m training.run_trianing_series.run_end_to_end_series --stages 2 3 4 5
```

Avviare un singolo stage o un sottoinsieme:

```powershell
# Solo stage 3
python -m training.run_trianing_series.run_end_to_end_series --stages 3

# Stage 2 e 5, eseguiti in questo ordine
python -m training.run_trianing_series.run_end_to_end_series --stages 2 5
```

Opzioni comuni vengono applicate a tutti gli stage selezionati:

```powershell
python -m training.run_trianing_series.run_end_to_end_series `
  --stages 2 3 4 5 `
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
python -m training.run_trianing_series.run_end_to_end_series --stages 3 4 5
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
python -m training.run_trianing_series.run_end_to_end_series --stages 4 5
```

`--epochs` indica epoca finale totale, non epoche aggiuntive. Con checkpoint
epoca 6 e `--epochs 30`, training riparte da 7 e termina al massimo a 30.

Se blocco avviene prima del primo checkpoint, rilanciare normalmente quello
stage. Se manca cartella `epochs`, usare `best.pt`, accettando ripartenza dalla
sua epoca invece che dall'ultima epoca eseguita.
