# Evaluation Complementary Item Retrieval

## Indice

- [Metriche](#metriche)
  - [FITB accuracy e ranking dei candidati](#fitb-accuracy-e-ranking-dei-candidati)
  - [MRR e Recall@2 sull'intero split](#mrr-e-recall2-sullintero-split)
- [Preparazione](#preparazione)
- [Compatibilita checkpoint](#compatibilita-checkpoint)
- [Avvio](#avvio)
- [Flag principali](#flag-principali)

## Metriche

| Nome | Cosa fa | Usa la soglia |
|---|---|---:|
| FITB accuracy | Misura la frazione di outfit parziali per cui l'item corretto è il candidato più vicino. | No |
| MRR | Misura la media dell'inverso della posizione dell'item corretto. | No |
| Recall@2 | Misura la frazione di query con l'item corretto tra i primi due candidati. | No |

Il protocollo scelto è **Fill-in-the-Blank (FITB)** della repository
[bigohofone/outfit-transformer](https://github.com/bigohofone/outfit-transformer).
Il suo [test CIR](https://github.com/bigohofone/outfit-transformer/blob/main/src/run/3_test_complementary.py)
usa query FITB, quattro candidati e distanza euclidea;
[`compute_cir_scores`](https://github.com/bigohofone/outfit-transformer/blob/main/src/evaluation/metrics.py)
restituisce l'accuracy con nome `acc`. Qui la stessa metrica si chiama
`fitb_accuracy`; MRR e Recall@2 sono diagnostiche aggiuntive già usate nel
[training CIR locale](../../training/CIR/README.md).

Il [paper OutfitTransformer, sezione 4.3](https://openaccess.thecvf.com/content/WACV2023/papers/Sarkar_OutfitTransformer_Learning_Outfit_Representations_for_Fashion_Recommendation_WACV_2023_paper.pdf)
riporta anche Recall@10/30/50 per il retrieval su un catalogo di categoria,
con il protocollo CSA-Net. Servono quel catalogo e quelle query per un confronto
equivalente. Questo comando valuta FITB: calcolare Recall@10 su quattro
candidati darebbe sempre 1 e non riprodurrebbe l'esperimento del paper.

### FITB accuracy e ranking dei candidati

Ogni esempio contiene un outfit parziale, il capo corretto e i distrattori
ufficiali di Polyvore. Il modello produce un embedding della query e un
embedding indipendente per ogni candidato. I candidati sono ordinati per
distanza euclidea crescente:

```text
distanza(query, candidato) = sqrt(sum((query_j - candidato_j)^2))
distanza minore -> candidato migliore
```

Le distanze non sono probabilità; il comando non usa `--threshold`. L'eventuale
normalizzazione degli embedding finali dipende dalla configurazione salvata
nel checkpoint, come nel training.

La risposta corretta viene ricavata da `blank_position` e dall'outfit originale,
anche quando non è il primo elemento nel file `answers`. Il loader la colloca
prima dei distrattori soltanto per organizzare i tensori. Il rank è:

```text
rank = 1 + numero di distrattori con distanza <= distanza del positivo
FITB accuracy = numero di query con rank 1 / numero totale di query
```

Le parità seguono la convenzione conservativa del training locale: il
distrattore precede il positivo. Quattro distanze identiche producono rank 4.
La repo di riferimento usa `argmin`, che in parità seleziona il primo candidato;
questa differenza è esplicita nel report come `tie_policy: "pessimistic"`.

Esempio su tre query, ognuna con quattro candidati:

| Distanza positivo | Distanze distrattori | Rank positivo | FITB corretta |
|---:|---|---:|---:|
| `0.2` | `0.4, 0.8, 1.0` | 1 | Sì |
| `0.7` | `0.4, 1.1, 1.8` | 2 | No |
| `1.5` | `0.2, 0.4, 1.0` | 4 | No |

FITB accuracy vale `1 / 3 = 0.3333`. Nel JSON tutte le metriche sono frazioni
tra `0` e `1`; `0.6854` corrisponde al `68.54%`.

### MRR e Recall@2 sull'intero split

MRR assegna più valore alle risposte corrette vicine alla prima posizione:

```text
MRR = mean(1 / rank)
rank 1 -> 1.0
rank 2 -> 0.5
rank 4 -> 0.25
```

Recall@2 considera corretta una query quando il positivo compare nei primi
due posti. Con un solo positivo per query coincide con Hit Rate@2:

```text
Recall@2 = numero di query con rank <= 2 / numero totale di query
```

Nell'esempio precedente, MRR vale `(1 + 0.5 + 0.25) / 3 = 0.5833` e Recall@2
vale `2 / 3 = 0.6667`.

I rank vengono raccolti per tutte le query prima di calcolare le metriche.
Ogni query pesa allo stesso modo, incluso l'ultimo batch incompleto. Cambiare
dimensione dei batch non cambia la definizione del ranking; piccole differenze
numeriche possono dipendere dal device e dai kernel PyTorch.

## Preparazione

`classic` e `new_classic` leggono immagini e descrizioni Polyvore.
`precomputed` richiede la cache embedding dello split scelto, completa di
item delle query, positivi e distrattori. Comandi per produrre cache FashionCLIP
o OpenRouter nella [guida degli script](../../scripts/README.md#esempi).

La root cache salvata nel checkpoint viene riusata. Se la cache è stata
spostata, passare `--embedding-root`. Per gli artefatti nella cartella locale
`huggingface/`, usare `--embedding-root huggingface/precomputed_embeddings`
quando si vuole leggere la cache distribuita insieme ai pesi.

Le annotazioni vengono cercate prima in `datasets/polyvore-outfits/`, poi nella
cache Hugging Face; soltanto i file mancanti vengono scaricati. Un percorso
locale diverso si imposta con `--dataset-root`. Si leggono soltanto lo split
scelto e il relativo `fill_in_blank_test.json` o `fill_in_blank_valid.json`;
non servono lo split train né campionamenti casuali dei target.

Con embedding precomputati non vengono caricati parquet immagini. I metadata
sono necessari soltanto se il checkpoint abilita `use_category_embedding`;
la categoria del target viene allora letta dalle annotazioni dei metadata.

## Compatibilita checkpoint

Evaluation legge checkpoint **CIR schema v1** prodotti dal training locale.
Dataset, subset, modalità feature, architettura Transformer, dimensione dello
spazio retrieval, normalizzazione finale e uso delle categorie vengono
ricavati dal checkpoint. I pesi della repo esterna hanno un formato diverso:
la repo è il riferimento per il protocollo di valutazione.

Sono supportati soltanto checkpoint allenati con l'architettura CIR attuale:
token CIR normalizzato L2 e nessuna LayerNorm finale aggiuntiva. I checkpoint
precedenti con `cir.encoder.norm.weight` e `cir.encoder.norm.bias` non sono
supportati e non vengono convertiti.

Il caricamento rimane stretto (`strict=True`): pesi mancanti, inattesi o di
forma incompatibile producono un errore. Non servono nuovi flag; le regole di
`--resume` del training sono descritte nella
[guida CIR](../../training/CIR/README.md#inizializzazione-da-cp-e-resume).

## Avvio

PowerShell:

```powershell
python -m evaluation.CIR.evaluate_cir `
  --checkpoint checkpoints/nondisjoint/cir_precomputed/best.pt
```

Linux (Bash):

```bash
python -m evaluation.CIR.evaluate_cir \
  --checkpoint checkpoints/nondisjoint/cir_precomputed/best.pt
```

Esempio validation e output esplicito:

PowerShell:

```powershell
python -m evaluation.CIR.evaluate_cir `
  --checkpoint checkpoints/nondisjoint/cir_precomputed/best.pt `
  --split validation `
  --output results/cir_validation.json
```

Linux (Bash):

```bash
python -m evaluation.CIR.evaluate_cir \
  --checkpoint checkpoints/nondisjoint/cir_precomputed/best.pt \
  --split validation \
  --output results/cir_validation.json
```

## Flag principali

| Flag | Default | Funzione |
|---|---|---|
| `--checkpoint` | richiesto | Checkpoint schema v1 prodotto dal training CIR. |
| `--split` | `test` | Split `test` o `validation`. |
| `--embedding-root` | valore checkpoint | Sovrascrive root cache embedding. |
| `--dataset-root` | valore checkpoint o `datasets/polyvore-outfits` | Cerca qui annotazioni e dati prima del fallback Hugging Face. |
| `--output` | derivato dal checkpoint | Percorso JSON del report. |
| `--cache-dir` | valore checkpoint | Directory cache Hugging Face. |
| `--batch-size` | `512` | Query per batch; minimo 1. |
| `--candidate-batch-size` | `512` | Massimo numero di candidati codificati insieme; minimo 1. |
| `--device` | `auto` | CUDA, MPS o CPU; accetta device esplicito. |
| `--num-workers` | `0` | Processi DataLoader. |
| `--pin-memory` | disabilitato | Abilita pinned memory. |
| `--seed` | `42` | Seed per la riproducibilità. |
| `--log-every` | `10` | Intervallo dei log, in batch di query. |
| `--token` / `--no-token` | token locale | Autenticazione Hugging Face. |

Le due dimensioni di batch permettono di limitare separatamente la memoria
richiesta dalle query e dai candidati senza cambiare i distrattori valutati.

Senza `--output`, il report viene scritto in:

```text
results/cir/<dataset>/<subset>/<directory-checkpoint>/<checkpoint>_<split>.json
```

Il JSON contiene identità checkpoint, epoca, dataset, feature mode, uso delle
categorie, protocollo `polyvore_fitb`, distanza, gestione delle parità e
`metrics` con `fitb_accuracy`, `mrr`, `recall_at_2`, `examples`.
Scrittura atomica: il report precedente non resta parziale in caso di errore.
