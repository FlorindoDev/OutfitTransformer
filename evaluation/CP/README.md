# Evaluation Compatibility Prediction

## Indice

- [Metriche](#metriche)
  - [Accuracy e soglia inclusiva](#accuracy-e-soglia-inclusiva)
  - [ROC AUC tie-aware sull'intero split](#roc-auc-tie-aware-sullintero-split)
- [Preparazione](#preparazione)
- [Avvio](#avvio)
- [Flag principali](#flag-principali)

## Metriche

| Nome | Cosa fa | Usa la soglia |
|---|---|---:|
| Accuracy | Misura la percentuale di outfit classificati correttamente. | Sì |
| Precision | Tra gli outfit predetti compatibili, misura quanti lo sono realmente. | Sì |
| Recall | Tra gli outfit realmente compatibili, misura quanti vengono riconosciuti. | Sì |
| F1 | Combina precision e recall tramite media armonica. | Sì |
| ROC AUC | Misura quanto bene il modello assegna probabilità maggiori agli outfit compatibili rispetto a quelli non compatibili. | No |

### Accuracy e soglia inclusiva

Il modello restituisce per ogni outfit una probabilità di compatibilità tra
`0` e `1`. Per calcolare accuracy, precision, recall e F1, tale valore viene
convertito in una classe binaria:

```text
probabilità >= soglia  -> compatibile (1)
probabilità < soglia   -> non compatibile (0)
```

Con soglia predefinita `0.5`, anche una probabilità esattamente uguale a `0.5`
viene quindi classificata come compatibile. L'accuracy è la frazione di outfit
classificati correttamente:

```text
accuracy = predizioni corrette / numero totale di outfit
```

Esempio con soglia `0.5`:

| Probabilità | Label reale | Classe predetta | Esito |
|---:|---:|---:|---|
| `0.82` | `1` | `1` | corretta |
| `0.50` | `1` | `1` | corretta: soglia inclusiva |
| `0.49` | `0` | `0` | corretta |
| `0.15` | `1` | `0` | errata |

Tre predizioni su quattro sono corrette, quindi accuracy vale `3 / 4 = 0.75`.
Cambiare `--threshold` modifica accuracy, precision, recall e F1. Una soglia
più bassa tende a classificare più outfit come compatibili; una più alta tende
a essere più selettiva.

### ROC AUC tie-aware sull'intero split

ROC AUC non converte le probabilità in classi e non usa `--threshold`. Misura
quanto bene il modello ordina gli outfit: un outfit realmente compatibile
dovrebbe ricevere probabilità maggiore di uno non compatibile.

La metrica può essere interpretata confrontando tutte le coppie formate da un
esempio positivo e uno negativo:

```text
score positivo > score negativo  -> contributo 1.0
score positivo = score negativo  -> contributo 0.5
score positivo < score negativo  -> contributo 0.0
```

`Tie-aware` significa quindi che una parità riceve mezzo punto. Questo evita di
considerare due probabilità uguali come un ordinamento corretto o completamente
errato.

Le probabilità e le label di tutti i batch vengono prima aggregate. ROC AUC
viene poi calcolata una sola volta sull'intero split, non come media delle AUC
dei singoli batch. Il risultato non dipende dalla suddivisione in batch e
include anche confronti tra esempi appartenenti a batch diversi.

Interpretazione tipica:

```text
AUC = 1.0  -> tutti i positivi precedono tutti i negativi
AUC = 0.5  -> ordinamento equivalente al caso casuale
AUC = 0.0  -> ordinamento completamente invertito
```

ROC AUC richiede almeno un esempio positivo e uno negativo nell'intero split.
Un modello può avere buona AUC ma accuracy inferiore con una soglia poco adatta:
AUC valuta l'ordinamento, mentre accuracy valuta le classi dopo il taglio.

## Preparazione

`classic` e `new_classic` leggono immagini e descrizioni Polyvore. `clip`
richiede cache embedding dello split scelto. Per test predefinito:

```powershell
python -m scripts.precompute_embeddings `
  --variant nondisjoint `
  --split test
```

Root cache salvata nel checkpoint viene riusata. Se cache spostata, passare
`--embedding-root`.

Annotazioni vengono cercate prima in `datasets/polyvore-outfits/`, poi nella
cache Hugging Face; soltanto file mancanti vengono scaricati. Percorso locale
diverso si imposta con `--dataset-root`. In modalità `clip`, evaluation non
carica parquet immagini né metadata.

## Avvio

```powershell
python -m evaluation.CP.evaluate_cp `
  --checkpoint checkpoints/nondisjoint/cp_clip/best.pt
```

Esempio validation e output esplicito:

```powershell
python -m evaluation.CP.evaluate_cp `
  --checkpoint checkpoints/nondisjoint/cp_clip/best.pt `
  --split validation `
  --output results/cp_validation.json
```

## Flag principali

| Flag | Default | Funzione |
|---|---|---|
| `--checkpoint` | richiesto | Checkpoint schema v1 prodotto dal training CP. |
| `--split` | `test` | Split `test` o `validation`. |
| `--embedding-root` | valore checkpoint | Sovrascrive root cache CLIP. |
| `--dataset-root` | valore checkpoint o `datasets/polyvore-outfits` | Cerca qui annotazioni e dati prima del fallback Hugging Face. |
| `--output` | derivato dal checkpoint | Percorso JSON del report. |
| `--batch-size` | `512` | Outfit per batch. |
| `--threshold` | `0.5` | Soglia inclusiva per metriche discrete. |
| `--device` | `auto` | CUDA, MPS o CPU; accetta device esplicito. |
| `--num-workers` | `0` | Processi DataLoader. |
| `--pin-memory` | disabilitato | Abilita pinned memory. |
| `--token` / `--no-token` | token locale | Autenticazione Hugging Face. |

Senza `--output`, report scritto in:

```text
results/cp/<variant>/<directory-checkpoint>/<checkpoint>_<split>.json
```

JSON contiene identita checkpoint, epoca, dataset, feature mode, soglia e
tutte metriche. Scrittura atomica: report precedente non resta parziale.
