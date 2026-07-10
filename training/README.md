# Training

Il package separa il training per task:

- [Compatibility Prediction](cp/README.md): implementato;
- [Complementary Item Retrieval](cir/README.md): non ancora implementato.

## Indice

- [Compatibility Prediction](#compatibility-prediction)
  - [Avviare una nuova run](#avviare-una-nuova-run)
  - [Riprendere da checkpoint](#riprendere-da-checkpoint)
  - [ResNet-18](#resnet-18)
- [Complementary Item Retrieval](#complementary-item-retrieval)

## Compatibility Prediction

La pipeline CP è composta da runner di epoca, orchestratore, history,
checkpoint manager e plotter indipendenti. L'API breve `train_cp()` compone i
componenti standard; `CPTrainer` permette di sostituire il runner o collegare
callback senza duplicare persistenza e metriche.

Ogni epoca produce:

- train loss e accuracy;
- validation loss, accuracy e ROC AUC;
- checkpoint dell'epoca ed eventuale nuovo best;
- tre grafici cumulativi loss, accuracy e validation accuracy/AUC.

### Avviare una nuova run

Training con configurazione predefinita:

```powershell
python -m training.cp.train_cp
```

Training con `layer4` e FC visuale allenabili:

```powershell
python -m training.cp.train_cp `
  --variant disjoint `
  --epochs 20 `
  --batch-size 32 `
  --image-fine-tune-mode fc_and_layer4
```

Run con checkpoint e grafici isolati in una cartella dedicata:

```powershell
python -m training.cp.train_cp `
  --epochs 30 `
  --checkpoint checkpoints\experiment_01\best.pt `
  --checkpoint-dir checkpoints\experiment_01\epochs `
  --plot-dir checkpoints\experiment_01\plots
```

GPU specifica e log batch meno frequenti:

```powershell
python -m training.cp.train_cp `
  --device cuda:0 `
  --log-interval 100
```

### Riprendere da checkpoint

Ripresa dal checkpoint migliore:

```powershell
python -m training.cp.train_cp `
  --epochs 40 `
  --resume checkpoints\cp_best.pt `
  --image-fine-tune-mode fc_only
```

Ripresa da una specifica epoca:

```powershell
python -m training.cp.train_cp `
  --epochs 40 `
  --resume checkpoints\cp_epochs\cp_epoch_020.pt `
  --image-fine-tune-mode fc_and_layer4
```

Ripresa con checkpoint e grafici nuovi salvati in una cartella separata:

```powershell
python -m training.cp.train_cp `
  --epochs 40 `
  --resume checkpoints\cp_best.pt `
  --image-fine-tune-mode fc_only `
  --checkpoint checkpoints\resume_01\best.pt `
  --checkpoint-dir checkpoints\resume_01\epochs `
  --plot-dir checkpoints\resume_01\plots
```

Le nuove epoche vengono sempre salvate in `resume_01\epochs` e i nuovi grafici
in `resume_01\plots`. `resume_01\best.pt` viene invece creato soltanto se una
nuova epoca migliora la migliore validation loss contenuta nel checkpoint di
partenza. Se non avviene alcun miglioramento, il best originale resta
`checkpoints\cp_best.pt`.

`--epochs` indica l'ultima epoca totale. Nell'esempio, un checkpoint terminato
all'epoca 20 continua dall'epoca 21 fino alla 40; non esegue altre 40 epoche.

Optimizer, scheduler, history, migliore loss e RNG vengono ripristinati dal
checkpoint nuovo. Learning rate, weight decay e configurazione StepLR salvati
nel checkpoint prevalgono sugli stessi flag CLI. Cambiare
`--image-fine-tune-mode` è invece consentito e permette un fine-tuning a fasi.

I checkpoint legacy restano caricabili, ma non possono fornire history, RNG e
migliore loss precedenti completi.

### ResNet-18

La FC visuale è sempre allenabile. Il flag
`--image-fine-tune-mode` controlla il resto:

- `fc_only`: congela tutti i blocchi ResNet e le relative BatchNorm;
- `fc_and_layer4`: allena `layer4`, le sue BatchNorm e la FC.

`--no-pretrained-image` controlla solo l'inizializzazione ImageNet. Non rende
allenabili i blocchi congelati, quindi non rappresenta un training completo da
zero.

SentenceBERT resta congelato; la sua proiezione FC, il token `OUTFIT`, il
Transformer e il classificatore CP restano allenabili.

Dettagli, flag, resume, formato checkpoint, grafici ed esempi:
[guida completa CP](cp/README.md).

## Complementary Item Retrieval

Il training CIR non è ancora implementato. Riutilizzerà l'encoder comune, il
token `TARGET`, la proiezione CIR e la Set-wise Ranking Loss. Consulta la
[pagina CIR](cir/README.md) e il [modello CIR](../model/cir/README.md).
