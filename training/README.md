# Training

Il package separa le responsabilità riusabili dal codice specifico dei task:

- `common`: cache embedding, metriche, checkpoint, seed e device;
- `CP`: training Compatibility Prediction in modalità `classic` del paper o
  `clip` con embedding FashionCLIP precomputati.

Per avviare CP, vedere [CP/README.md](CP/README.md).
