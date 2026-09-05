# Model

## Moduli

| Modulo | Responsabilità | Documentazione |
|---|---|---|
| `common` | Trasforma immagini e testi in embedding multimodali normalizzati e contiene i componenti condivisi. | [README common](common/README.md) |
| `cp` | Usa le rappresentazioni common per stimare la compatibilità complessiva di un outfit. | [README CP](cp/README.md) |
| `CIR` | Usa le rappresentazioni common per creare embedding confrontabili di outfit parziali e item complementari. | [README CIR](CIR/README.md) |

Tutta la configurazione del modello è centralizzata in `common/config.py`:
Transformer dei task, encoder, focal loss CP, spazio di retrieval e triplet loss CIR
condividono così un unico punto di accesso e validazione.

## Embedding dei task

`task_emb` è un parametro allenabile condiviso tra CP e CIR. `predict_emb` ed
`embed_emb` sono invece parametri allenabili interni rispettivamente a CP e
CIR. Tutti nascono da valori casuali con deviazione standard `0.02`: non
provengono da immagini, testo, FashionCLIP o dataset. Ogni task replica il
proprio token iniziale per gli outfit del batch; la self-attention lo rende poi
specifico dell'outfit con cui interagisce.

`TaskEmbedding` centralizza `task_emb` in `model.common`. Per condividerlo
realmente, il chiamante crea una sola istanza e passa la stessa istanza a CP e
CIR. Se un task non ne riceve una, crea un'istanza privata.

| Embedding | Dimensione | Ruolo |
|---|---:|---|
| `task_emb` | 512 | Parte condivisa tra CP e CIR. Apprende conoscenza generale sulle relazioni di compatibilità. |
| `predict_emb` | 512 | Parte specifica CP. Indica al Transformer di produrre una rappresentazione utile alla classificazione. |
| `embed_emb` | 512 | Parte specifica CIR. Indica al Transformer di produrre una rappresentazione utile al retrieval. |

I token completi sono concatenazioni da 1024 valori, normalizzate L2 prima
dell'inserimento nella sequenza:

```text
CP  = L2([task_emb | predict_emb])
CIR = L2([task_emb | embed_emb])
```

L2 agisce sull'intero token e ne porta la norma a 1, come per gli item common.
Con categoria CIR attiva si normalizza dopo la somma:
`L2([task_emb | embed_emb + category_emb])`. I parametri restano allenabili;
la normalizzazione fa parte del forward e non li sovrascrive.

Gli encoder CP e CIR mantengono le LayerNorm interne dei layer, senza una
LayerNorm finale aggiuntiva. Nel training CIR, `--pretrained-cp` trasferisce
pipeline common, task embedding e tutti i layer del Transformer CP. I dettagli
di compatibilità dei checkpoint sono nel
[README del training CIR](../training/CIR/README.md#inizializzazione-da-cp-e-resume).

Le parti specifiche cambiano la query usata dalla self-attention. CP e CIR
possono quindi estrarre informazioni diverse dallo stesso outfit prima delle
rispettive teste finali. Usare soltanto `task_emb` sarebbe possibile, ma
forzerebbe entrambi i task a condividere la stessa rappresentazione e potrebbe
aumentare il conflitto tra i loro gradienti.

Durante CP, la focal loss aggiorna sia `predict_emb` sia `task_emb`. Durante
CIR, la loss di retrieval aggiorna sia `embed_emb` sia lo stesso `task_emb`, se
il training include entrambi tra i parametri ottimizzati. `embed_emb` identifica
il task di retrieval. Il flag `--category-emb` aggiunge la categoria target;
la descrizione del target non viene usata nel token.
