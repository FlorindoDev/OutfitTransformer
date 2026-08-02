# Serie end-to-end disjoint

## Test eseguiti

| Stage presente | Epoche salvate | Best epoca | Seed | Dropout | Weight decay | Focal alpha | Best val AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| `01_paper_standard_defaults` | 9 | 4 | 42 | `0.1` | `0.0` | `0.25` | `0.848733` |
| `02_seed_7` | 6 | 4 | 7 | `0.1` | `0.0` | `0.25` | `0.850146` |
| `04_dropout_0` | 6 | 4 | 42 | `0.0` | `0.0` | `0.25` | `0.861856` |
| `05_dropout_02` | 6 | 5 | 42 | `0.2` | `0.0` | `0.25` | `0.837997` |

## Configurazione completa

Configurazione ricavata dai `best.pt` realmente presenti; limite epoche dal
runner storico.

| Iperparametro | `01_paper_standard_defaults` | `02_seed_7` | `04_dropout_0` | `05_dropout_02` |
|---|---|---|---|---|
| CLI | `training.cp.train_cp` | `training.cp.train_cp` | `training.cp.train_cp` | `training.cp.train_cp` |
| Sorgente pesi | ResNet-18 ImageNet + SentenceBERT | ResNet-18 ImageNet + SentenceBERT | ResNet-18 ImageNet + SentenceBERT | ResNet-18 ImageNet + SentenceBERT |
| Dataset | `mvasil/polyvore-outfits` | `mvasil/polyvore-outfits` | `mvasil/polyvore-outfits` | `mvasil/polyvore-outfits` |
| Variante dataset | `disjoint` | `disjoint` | `disjoint` | `disjoint` |
| Epoche massime richieste | 30 | 30 | 30 | 30 |
| Epoche salvate | 1–9 | 1–6 | 1–6 | 1–6 |
| Batch size | 50 | 50 | 50 | 50 |
| Modalità ResNet | `full` | `full` | `full` | `full` |
| Blocchi ResNet allenabili | intera ResNet, FC e BatchNorm | intera ResNet, FC e BatchNorm | intera ResNet, FC e BatchNorm | intera ResNet, FC e BatchNorm |
| SentenceBERT | congelato | congelato | congelato | congelato |
| Modello testuale | `sentence-transformers/all-MiniLM-L6-v2` | `sentence-transformers/all-MiniLM-L6-v2` | `sentence-transformers/all-MiniLM-L6-v2` | `sentence-transformers/all-MiniLM-L6-v2` |
| Image embedding | 64 | 64 | 64 | 64 |
| Text embedding | 64 | 64 | 64 | 64 |
| Item embedding / `d_model` | 128 | 128 | 128 | 128 |
| Layer Transformer | 6 | 6 | 6 | 6 |
| Teste di attenzione | 16 | 16 | 16 | 16 |
| Dimensione feed-forward | 512 | 512 | 512 | 512 |
| Dropout | `0.1` | `0.1` | `0.0` | `0.2` |
| Normalizzazione Transformer | post-norm | post-norm | post-norm | post-norm |
| Loss | Binary Focal Loss | Binary Focal Loss | Binary Focal Loss | Binary Focal Loss |
| Focal alpha | `0.25` | `0.25` | `0.25` | `0.25` |
| Focal gamma | `2.0` | `2.0` | `2.0` | `2.0` |
| Optimizer | Adam | Adam | Adam | Adam |
| Learning rate | `1e-5` | `1e-5` | `1e-5` | `1e-5` |
| Weight decay | `0.0` | `0.0` | `0.0` | `0.0` |
| Scheduler | StepLR, step 10, gamma `0.5` | StepLR, step 10, gamma `0.5` | StepLR, step 10, gamma `0.5` | StepLR, step 10, gamma `0.5` |
| Metrica best checkpoint | validation ROC AUC | validation ROC AUC | validation ROC AUC | validation ROC AUC |
| Early stopping | patience 5, delta `1e-4` | patience 5, delta `1e-4` | patience 5, delta `1e-4` | patience 5, delta `1e-4` |
| Gradient clipping | disabilitato | disabilitato | disabilitato | disabilitato |
| Seed | 42 | 7 | 42 | 42 |
| Checkpoint | best + uno per epoca | best + uno per epoca | best + uno per epoca | best + uno per epoca |

Stage 3 e 6–9 non risultano presenti in questa root, quindi non inclusi.
