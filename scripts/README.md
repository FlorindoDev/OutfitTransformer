# Script di supporto

Questa cartella contiene gli strumenti eseguibili che preparano i dati usati dal progetto senza partecipare direttamente all'addestramento.

## Indice

- [File](#file)
- [Flag della CLI](#flag-della-cli)
- [Precomputazione FashionCLIP](#precomputazione-fashionclip)
- [Controlli e sicurezza](#controlli-e-sicurezza)
- [Artefatti prodotti](#artefatti-prodotti)
- [Contenuto degli shard](#contenuto-degli-shard)
- [Uso nel training CP](#uso-nel-training-cp)
- [Esempi](#esempi)

## File

| File | Funzione concettuale |
| --- | --- |
| `precompute_embeddings.py` | Trasforma immagini e descrizioni Polyvore in embedding FashionCLIP già pronti per il training CP in modalità CLIP. |

## Flag della CLI

| Flag | Default | Cosa fa |
| --- | --- | --- |
| `-h`, `--help` | — | Mostra la guida dei comandi e termina. |
| `--variant` | `disjoint` | Seleziona la variante Polyvore: `disjoint` oppure `nondisjoint`. |
| `--split` | `train` | Seleziona lo split da elaborare: `train`, `validation` oppure `test`. |
| `--model-name` | `patrickjohncyh/fashion-clip` | Sceglie il modello FashionCLIP usato per codificare immagini e testi. |
| `--output-dir` | `precomputed_embeddings` | Imposta la cartella radice in cui salvare cache e manifest. |
| `--cache-dir` | non impostato | Indica una cache personalizzata per i file scaricati da Hugging Face. |
| `--batch-size` | `128` | Numero di articoli codificati insieme; influenza soprattutto memoria e velocità di inferenza. |
| `--num-workers` | `0` | Numero di processi usati per caricare i dati. `0` mantiene il caricamento nel processo principale. |
| `--shard-size` | `10000` | Numero massimo di articoli in ogni file di output; non modifica il batch di inferenza. |
| `--output-dtype` | `float32` | Precisione degli embedding salvati: `float32` o `float16`. |
| `--device` | `auto` | Sceglie il dispositivo. In automatico prova CUDA, poi MPS e infine CPU; accetta anche un dispositivo esplicito. |
| `--limit` | non impostato | Limita il numero totale di articoli, utile per prove rapide; senza valore elabora tutto lo split. |
| `--overwrite` | disattivato | Sostituisce una cache già esistente, eliminando esclusivamente gli artefatti gestiti dallo script. |
| `--log-every` | `25` | Mostra l'avanzamento ogni N batch. |
| `--token` | credenziali locali | Usa un token Hugging Face esplicito; in assenza del flag vengono usate le credenziali locali disponibili. |
| `--no-token` | disattivato | Forza l'accesso anonimo a Hugging Face; è incompatibile con `--token`. |

## Precomputazione FashionCLIP

`precompute_embeddings.py` prepara una rappresentazione multimodale per ogni articolo Polyvore:

1. carica lo split e la variante richiesti;
2. codifica l'immagine con la torre visiva di FashionCLIP;
3. codifica la descrizione con la torre testuale, rispettando la lunghezza massima prevista dal modello;
4. normalizza separatamente le due rappresentazioni;
5. concatena `512` valori visivi e `512` testuali in un vettore finale da `1024` valori;
6. salva progressivamente i risultati in shard, senza calcolare gradienti o aggiornare FashionCLIP.

La precomputazione separa il costo di FashionCLIP dal training: il modulo CP legge vettori già pronti, riducendo tempo di calcolo e memoria richiesta durante le epoche.

## Controlli e sicurezza

Prima del salvataggio vengono verificati dimensioni, tipo numerico, valori finiti, vettori non nulli e unicità degli identificativi. Le dimensioni devono restare coerenti per tutto lo split.

Shard e manifest vengono scritti in modo atomico, così un'interruzione non lascia un file finale parziale. Una cartella di destinazione non vuota non viene modificata senza `--overwrite`; anche con quel flag lo script rimuove soltanto file riconosciuti come propri. La procedura non riprende uno split incompleto: per rigenerarlo occorre usare `--overwrite`.

## Artefatti prodotti

La destinazione segue la forma:

```text
precomputed_embeddings/<modello>/<variante>/<split>/
```

| Artefatto | Contenuto concettuale |
| --- | --- |
| `shard-*.pt` | Identificativi degli articoli e relativi embedding multimodali. |
| `manifest.json` | Configurazione, modello, fingerprint, dimensioni, precisione, conteggi e ordine degli shard. |
| Console | Dispositivo selezionato, avanzamento, prestazioni e riepilogo finale. |

Il manifest rende la cache verificabile: consente al training di controllare che dati, modello e struttura degli embedding siano quelli attesi.

## Contenuto degli shard

Ogni `shard-*.pt` è un dizionario PyTorch con tre campi:

| Campo | Tipo e forma | Significato |
| --- | --- | --- |
| `schema_version` | Intero, attualmente `1` | Versione del formato, usata per riconoscere cache compatibili. |
| `item_ids` | Tupla di `N` stringhe | Identificativi Polyvore degli articoli contenuti nello shard. |
| `embeddings` | Tensore CPU `[N, 1024]` | Una riga per ogni `item_id`, nello stesso ordine, salvata in `float32` o `float16`. |

Ogni riga di `embeddings` è composta da due parti concatenate:

| Posizione | Contenuto | Preparazione |
| --- | --- | --- |
| `0:512` | Rappresentazione dell'immagine | Prodotta dalla torre visiva FashionCLIP e normalizzata L2. |
| `512:1024` | Rappresentazione della descrizione | Prodotta dalla torre testuale FashionCLIP e normalizzata L2. |

`N` è al massimo pari a `--shard-size`: gli shard intermedi sono normalmente pieni, mentre l'ultimo può essere più piccolo. Gli shard non contengono immagini, descrizioni, categorie o pesi di FashionCLIP; conservano soltanto gli identificativi e le rappresentazioni numeriche necessarie al training.

## Uso nel training CP

La modalità CLIP del training CP richiede almeno le cache `train` e `validation` della stessa variante e dello stesso modello. Lo split `test` è facoltativo e serve per una valutazione separata. Le modalità classic e new classic non usano questi artefatti perché ricavano direttamente le feature da immagini e descrizioni.

`float16` riduce spazio su disco e memoria, con minore precisione; `float32` è la scelta predefinita. Il training converte comunque gli embedding nel tipo richiesto dal modello.

## Esempi

Precomputazione completa per il training predefinito `nondisjoint`:

```powershell
python -m scripts.precompute_embeddings --variant nondisjoint --split train
python -m scripts.precompute_embeddings --variant nondisjoint --split validation
```

Prova rapida su 100 articoli:

```powershell
python -m scripts.precompute_embeddings --variant nondisjoint --split train --limit 100 --device auto
```
