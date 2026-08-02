# Training progressivo eseguito

Solo esecuzioni realmente presenti in questa root. Configurazione ricavata dai
`best.pt`; runner originale eliminato dopo completamento serie.

## Stage e risultati validation

| Esecuzione | Epoche salvate | Best epoca | ResNet | Best val accuracy | Best val AUC |
|---|---:|---:|---|---:|---:|
| `01_paper_end_to_end` | 1–12 | 5 | `full` | `0.7597` | `0.841390` |
| `02_fc_only_base` | 1–12 | 12 | `fc_only` | `0.6757` | `0.739605` |
| `03_layer4_plateau_complete/phase_1` | 13–34 | 34 | `fc_and_layer4` | `0.7533` | `0.831550` |
| `03_layer4_plateau_complete/phase_2` | 35–39 | 39 | `fc_and_layer4` | `0.7543` | `0.835738` |
| `03_layer4_plateau_complete/phase_3` | 40–42 | 42 | `fc_and_layer4` | `0.7577` | `0.836451` |
| `04_full_low_lr` | 43–46 | 46 | `full` | `0.7587` | `0.834923` |

## Iperparametri completi

| Iperparametro | `01_paper_end_to_end` | `02_fc_only_base` | `03 phase_1` | `03 phase_2` | `03 phase_3` | `04_full_low_lr` |
|---|---|---|---|---|---|---|
| CLI | `training.cp.train_cp` | `training.cp.train_cp` | `training.cp.fine_tune_cp` | `training.cp.fine_tune_cp` | `training.cp.fine_tune_cp` | `training.cp.fine_tune_cp` |
| Sorgente | pesi preaddestrati | pesi preaddestrati | stage 2, epoca 12 | stage 3, epoca 34 | stage 3, epoca 39 | stage 3 phase 3, epoca 42 |
| Dataset | `mvasil/polyvore-outfits` | `mvasil/polyvore-outfits` | `mvasil/polyvore-outfits` | `mvasil/polyvore-outfits` | `mvasil/polyvore-outfits` | `mvasil/polyvore-outfits` |
| Variante | `disjoint` | `disjoint` | `disjoint` | `disjoint` | `disjoint` | `disjoint` |
| Batch size | 50 | 50 | 50 | 50 | 50 | 50 |
| Modalità ResNet | `full` | `fc_only` | `fc_and_layer4` | `fc_and_layer4` | `fc_and_layer4` | `full` |
| SentenceBERT | congelato | congelato | congelato | congelato | congelato | congelato |
| Modello testuale | `all-MiniLM-L6-v2` | `all-MiniLM-L6-v2` | ereditato | ereditato | ereditato | ereditato |
| Image / text embedding | 64 / 64 | 64 / 64 | 64 / 64 | 64 / 64 | 64 / 64 | 64 / 64 |
| Transformer layer / teste / FFN | 6 / 16 / 512 | 6 / 16 / 512 | 6 / 16 / 512 | 6 / 16 / 512 | 6 / 16 / 512 | 6 / 16 / 512 |
| Dropout / norm | `0.1` / post-norm | `0.1` / post-norm | `0.1` / post-norm | `0.1` / post-norm | `0.1` / post-norm | `0.1` / post-norm |
| Loss | Binary Focal Loss | Binary Focal Loss | Binary Focal Loss | Binary Focal Loss | Binary Focal Loss | Binary Focal Loss |
| Focal alpha / gamma | `0.5` / `1.0` | `0.5` / `1.0` | `0.5` / `1.0` | `0.5` / `1.0` | `0.5` / `1.0` | `0.5` / `1.0` |
| LR task | `1e-5` | `1e-5` | `1e-5` | `1e-5` | `1e-5` | `3e-6` |
| LR backbone | `1e-5` | congelato | `1e-6` | `1e-6` | `1e-6` | `3e-7` |
| Weight decay | `0.0` | `1e-4` | `1e-4` | `1e-4` | `1e-4` | `1e-4` |
| Scheduler | StepLR `10 × 0.5` | StepLR `10 × 0.5` | Cosine, minimo `0` | Cosine, minimo `0` | Cosine, minimo `0` | Cosine, minimo `0` |
| Best checkpoint | validation ROC AUC | validation ROC AUC | validation ROC AUC | validation ROC AUC | validation ROC AUC | validation ROC AUC |
| Early stopping | disabilitato | patience 3, delta `1e-4` | patience 4, delta `1e-4` | patience 4, delta `1e-4` | patience 4, delta `1e-4` | patience 2, delta `1e-4` |
| Gradient clipping | `1.0` | `1.0` | `1.0` | `1.0` | `1.0` | `1.0` |
| Seed | 42 | 42 | 42 | 42 | 42 | 42 |
