# Checkpoint nondisjoint eseguiti

Solo stage realmente presenti in questa root. Configurazioni future restano
nel README della serie di training.

## Stage eseguiti

| Stage | Epoche salvate | Best epoca | Dropout | Weight decay | Focal alpha | Best val AUC | Test AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| `01_paper_standard_defaults` | 5 | 4 | `0.1` | `0.0` | `0.25` | `0.887564` | `0.8923` |

## Configurazione completa

| Iperparametro | `01_paper_standard_defaults_Dropout_0.1` |
|---|---|
| CLI | `training.cp.train_cp` |
| Sorgente pesi | ResNet-18 ImageNet + SentenceBERT |
| Dataset | `mvasil/polyvore-outfits` |
| Variante dataset | `nondisjoint` |
| Epoche massime richieste | 30 |
| Epoche salvate | 5 |
| Batch size | 50 |
| Modalità ResNet | `full` |
| Blocchi ResNet allenabili | intera ResNet, FC e BatchNorm |
| SentenceBERT | congelato |
| Modello testuale | `sentence-transformers/all-MiniLM-L6-v2` |
| Image embedding | 64 |
| Text embedding | 64 |
| Item embedding / `d_model` | 128 |
| Layer Transformer | 6 |
| Teste di attenzione | 16 |
| Dimensione feed-forward | 512 |
| Dropout | `0.1` |
| Normalizzazione Transformer | post-norm |
| Loss | Binary Focal Loss |
| Focal alpha | `0.25` |
| Focal gamma | `2.0` |
| Optimizer | Adam |
| Learning rate | `1e-5` |
| Weight decay | `0.0` |
| Scheduler | StepLR, step 10, gamma `0.5` |
| Metrica best checkpoint | validation ROC AUC |
| Early stopping | patience 5, delta `1e-4` |
| Gradient clipping | disabilitato |
| Seed | 42 |
| Checkpoint | best + uno per epoca |

## Risultati best

| Split | Loss | Accuracy | ROC AUC | Esempi |
|---|---:|---:|---:|---:|
| Validation | `0.047193` | `0.7669` | `0.887564` | 10.000 |
| Test | `0.045939` | `0.7742` | `0.8923` | 20.000 |
