# Evaluation

Moduli per misurare checkpoint addestrati su split Polyvore completi.

Per checkpoint `precomputed`, preparare la cache dello split da valutare con lo
stesso encoder usato nel training (FashionCLIP, Marqo FashionSigLIP o
OpenRouter). CP e CIR leggono dimensione del modello dal checkpoint e cache da
`<embedding-root>/<subset>/<split>`. La root salvata nel checkpoint è usata per
default; `--embedding-root` permette di indicare una cache spostata. La cache
viene controllata per dataset, subset, split e dimensione.

| Task | Modulo | Guida |
|---|---|---|
| Compatibility Prediction | `evaluation.CP` | [README CP](CP/README.md) |
| Complementary Item Retrieval (FITB) | `evaluation.CIR` | [README CIR](CIR/README.md) |
