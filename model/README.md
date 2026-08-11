# Model

## Moduli

| Modulo | Responsabilità | Documentazione |
|---|---|---|
| `common` | Trasforma immagini e testi degli item in rappresentazioni multimodali contestualizzate e contiene i componenti condivisi. | [README common](common/README.md) |
| `cp` | Usa le rappresentazioni common per stimare la compatibilità complessiva di un outfit. | [README CP](cp/README.md) |
| `cir` | Userà le rappresentazioni common per creare un embedding destinato al retrieval di item complementari. Non ancora implementato. | README non ancora disponibile |

## Embedding dei task

`task_emb` è un parametro allenabile condiviso tra CP e CIR. `predict_emb` ed
`embed_emb` sono invece parametri allenabili interni rispettivamente a CP e
CIR. Tutti nascono da valori casuali con deviazione standard `0.02`: non
provengono da immagini, testo, FashionCLIP o dataset. Ogni task replica il
proprio token iniziale per gli outfit del batch; la self-attention lo rende poi
specifico dell'outfit con cui interagisce.

`TaskEmbedding` centralizza `task_emb` in `model.common`. Per condividerlo
realmente, il futuro modello padre ne creerà una sola istanza e passerà la
stessa istanza a CP e CIR. Se CP non ne riceve una, crea un'istanza privata.

| Embedding | Dimensione | Ruolo |
|---|---:|---|
| `task_emb` | 512 | Parte condivisa tra CP e CIR. Apprende conoscenza generale sulle relazioni di compatibilità. |
| `predict_emb` | 512 | Parte specifica CP. Indica al Transformer di produrre una rappresentazione utile alla classificazione. |
| `embed_emb` | 512 | Parte specifica CIR. Indica al Transformer di produrre una rappresentazione utile al retrieval. Non ancora implementata. |

I token completi sono concatenazioni da 1024 valori:

```text
CP  = [task_emb | predict_emb]
CIR = [task_emb | embed_emb]
```

Le parti specifiche cambiano la query usata dalla self-attention. CP e CIR
possono quindi estrarre informazioni diverse dallo stesso outfit prima delle
rispettive teste finali. Usare soltanto `task_emb` sarebbe possibile, ma
forzerebbe entrambi i task a condividere la stessa rappresentazione e potrebbe
aumentare il conflitto tra i loro gradienti.

Durante CP, la focal loss aggiorna sia `predict_emb` sia `task_emb`. Durante
CIR, la loss di retrieval aggiornerà sia `embed_emb` sia lo stesso `task_emb`.
`embed_emb` identifica il task di retrieval, non la categoria o la descrizione
dell'item cercato: tale informazione dovrà essere fornita separatamente dal
modulo CIR.
