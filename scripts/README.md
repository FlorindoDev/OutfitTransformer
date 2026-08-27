# Script di supporto

Questa cartella contiene gli strumenti eseguibili che preparano i dati usati dal progetto senza partecipare direttamente all'addestramento.

## Indice

- [File](#file)
- [Download Polyvore](#download-polyvore)
- [Flag precomputazione](#flag-precomputazione)
- [Precomputazione multimodale](#precomputazione-multimodale)
- [Controlli e sicurezza](#controlli-e-sicurezza)
- [Artefatti prodotti](#artefatti-prodotti)
- [Contenuto degli shard](#contenuto-degli-shard)
- [Uso nel training CP](#uso-nel-training-cp)
- [Esempi](#esempi)

## File

| File | Funzione concettuale |
| --- | --- |
| `download_polyvore.py` | Scarica intero repository Polyvore nella cartella locale usata dal progetto. |
| `precompute_embeddings.py` | Trasforma immagini e descrizioni Polyvore in embedding FashionCLIP locali o OpenRouter remoti. |

## Download Polyvore

Scarica entrambe le varianti e tutti gli split nella destinazione predefinita:

PowerShell:

```powershell
python -m scripts.download_polyvore
```

Linux (Bash):

```bash
python -m scripts.download_polyvore
```

Destinazione predefinita: `datasets/polyvore-outfits`. File invariati già
presenti vengono riusati. Struttura del repository resta invariata, quindi
training, evaluation e precomputazione trovano subito risorse locali.

| Flag | Default | Cosa fa |
| --- | --- | --- |
| `--output-dir` | `datasets/polyvore-outfits` | Imposta cartella locale del dataset. |
| `--cache-dir` | non impostato | Indica cache Hugging Face personalizzata. |
| `--revision` | revisione predefinita | Blocca download a branch, tag o commit. |
| `--max-workers` | `8` | Imposta download paralleli. |
| `--force-download` | disattivato | Riscarica anche file già presenti o in cache. |
| `--token` | credenziali locali | Usa token Hugging Face esplicito. |
| `--no-token` | disattivato | Forza accesso anonimo. |

## Flag precomputazione

| Flag | Default | Cosa fa |
| --- | --- | --- |
| `-h`, `--help` | — | Mostra la guida dei comandi e termina. |
| `--dataset` | `polyvore` | Seleziona source registrata nell'API pubblica `data`. |
| `--subset` | `disjoint` | Seleziona il subset del dataset. |
| `--split` | `train` | Seleziona lo split da elaborare: `train`, `validation` oppure `test`. |
| `--model-name` | dipende dal backend | Sceglie il modello. Senza `--openrouter` usa `patrickjohncyh/fashion-clip`; con OpenRouter usa `google/gemini-embedding-2`. |
| `--openrouter` | disattivato | Usa API embedding OpenRouter. Senza flag resta sempre attivo FashionCLIP locale. |
| `--openrouter-dimensions` | `512` | Dimensione richiesta per ciascuna modalità; `512` produce cache finali da `1024` compatibili col training predefinito. |
| `--openrouter-request-batch-size` | `8` | Numero massimo di input inviati in una singola richiesta API. |
| `--openrouter-image-size` | `224` | Lato in pixel delle immagini quadrate inviate come PNG base64. |
| `--openrouter-timeout` | `60` | Timeout in secondi per ogni tentativo API; gli errori transitori vengono ritentati automaticamente. |
| `--output-dir` | `precomputed_embeddings` | Imposta la cartella radice in cui salvare cache e manifest. |
| `--dataset-root` | `datasets/polyvore-outfits` | Cerca qui dataset Polyvore prima di usare cache Hugging Face o download. |
| `--cache-dir` | non impostato | Indica una cache personalizzata per i file scaricati da Hugging Face. |
| `--batch-size` | `128` | Numero di articoli codificati insieme; influenza soprattutto memoria e velocità di inferenza. |
| `--num-workers` | `0` | Numero di processi usati per caricare i dati. `0` mantiene il caricamento nel processo principale. |
| `--shard-size` | `10000` | Numero massimo di articoli in ogni file di output; non modifica il batch di inferenza. |
| `--output-dtype` | `float32` | Precisione degli embedding salvati: `float32` o `float16`. |
| `--device` | `auto` | Sceglie il dispositivo. FashionCLIP prova CUDA, poi MPS e CPU; OpenRouter usa CPU in automatico perché inferenza è remota. Accetta anche un dispositivo esplicito. |
| `--limit` | non impostato | Limita il numero totale di articoli, utile per prove rapide; senza valore elabora tutto lo split. |
| `--overwrite` | disattivato | Sostituisce una cache già esistente, eliminando esclusivamente gli artefatti gestiti dallo script. |
| `--log-every` | `25` | Mostra l'avanzamento ogni N batch. |
| `--token` | credenziali locali | Usa un token Hugging Face esplicito; in assenza del flag vengono usate le credenziali locali disponibili. |
| `--no-token` | disattivato | Forza l'accesso anonimo a Hugging Face; è incompatibile con `--token`. |

## Precomputazione multimodale

`precompute_embeddings.py` prepara una rappresentazione multimodale per ogni articolo Polyvore:

1. chiede alla source pubblica item dello split e subset richiesti;
2. codifica l'immagine con FashionCLIP oppure con un modello embedding
   multimodale OpenRouter;
3. codifica la descrizione con lo stesso backend e modello;
4. normalizza separatamente le due rappresentazioni;
5. concatena `D` valori visivi e `D` testuali; con default `D=512` ottiene un vettore da `1024` valori;
6. salva progressivamente i risultati in shard, senza calcolare gradienti o aggiornare encoder.

Risoluzione dati segue ordine: `--dataset-root`, cache Hugging Face, download.
Se parquet e metadata sono già locali, precomputazione non contatta repository
Polyvore.

FashionCLIP resta backend predefinito e non richiede flag. OpenRouter è opt-in,
legge la chiave solo dalla variabile `OPENROUTER_API_KEY` e non la salva nel
manifest. Immagini e descrizioni vengono inviate al provider remoto; richieste
possono avere costi. Modello scelto deve accettare input immagine e testo e
restituire embedding nello stesso spazio.

La precomputazione separa costo degli encoder dal training: il modulo CP legge
vettori già pronti, riducendo tempo di calcolo e memoria richiesta durante le
epoche.

## Controlli e sicurezza

Prima del salvataggio vengono verificati dimensioni, tipo numerico, valori finiti, vettori non nulli e unicità degli identificativi. Le dimensioni devono restare coerenti per tutto lo split.

Shard e manifest vengono scritti in modo atomico, così un'interruzione non lascia un file finale parziale. Una cartella di destinazione non vuota non viene modificata senza `--overwrite`; anche con quel flag lo script rimuove soltanto file riconosciuti come propri. La procedura non riprende uno split incompleto: per rigenerarlo occorre usare `--overwrite`.

## Artefatti prodotti

La destinazione segue la forma:

```text
precomputed_embeddings/<modello>/<subset>/<split>/
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
| `schema_version` | Intero, attualmente `2` | Versione del formato, usata per riconoscere cache compatibili. |
| `item_ids` | Tupla di `N` stringhe | Identificativi Polyvore degli articoli contenuti nello shard. |
| `embeddings` | Tensore CPU `[N, 2D]` | Una riga per ogni `item_id`, nello stesso ordine, salvata in `float32` o `float16`. |

Ogni riga di `embeddings` è composta da due parti concatenate:

| Posizione | Contenuto | Preparazione |
| --- | --- | --- |
| `0:D` | Rappresentazione dell'immagine | Prodotta dall'encoder visuale e normalizzata L2. |
| `D:2D` | Rappresentazione della descrizione | Prodotta dall'encoder testuale e normalizzata L2. |

`D` vale `512` per default, quindi ogni riga contiene `1024` valori. Cambiare
`--openrouter-dimensions` richiede una configurazione Transformer con dimensioni
coerenti.

`N` è al massimo pari a `--shard-size`: gli shard intermedi sono normalmente pieni, mentre l'ultimo può essere più piccolo. Gli shard non contengono immagini, descrizioni, categorie, pesi o API key; conservano soltanto identificativi e rappresentazioni numeriche necessarie al training.

## Uso nel training CP

La modalità CLIP del training CP richiede almeno le cache `train` e `validation` dello stesso dataset, subset e modello. Lo split `test` è facoltativo e serve per una valutazione separata. Le modalità classic e new classic non usano questi artefatti perché ricavano direttamente le feature da immagini e descrizioni.

`float16` riduce spazio su disco e memoria, con minore precisione; `float32` è la scelta predefinita. Il training converte comunque gli embedding nel tipo richiesto dal modello.

## Esempi

Precomputazione completa per il training predefinito `nondisjoint`:

PowerShell:

```powershell
python -m scripts.precompute_embeddings --subset nondisjoint --split train
python -m scripts.precompute_embeddings --subset nondisjoint --split validation
python -m scripts.precompute_embeddings --subset nondisjoint --split test
```

Linux (Bash):

```bash
python -m scripts.precompute_embeddings --subset nondisjoint --split train
python -m scripts.precompute_embeddings --subset nondisjoint --split validation
python -m scripts.precompute_embeddings --subset nondisjoint --split test
```

Prova rapida su 100 articoli:

PowerShell:

```powershell
python -m scripts.precompute_embeddings --subset nondisjoint --split train --limit 100 --device auto
```

Linux (Bash):

```bash
python -m scripts.precompute_embeddings --subset nondisjoint --split train --limit 100 --device auto
```

Prova OpenRouter.

PowerShell:

```powershell
$env:OPENROUTER_API_KEY = "<chiave>"
python -m scripts.precompute_embeddings `
  --openrouter `
  --model-name google/gemini-embedding-2 `
  --subset nondisjoint `
  --split train `
  --limit 100
```

Linux (Bash):

```bash
export OPENROUTER_API_KEY="<chiave>"
python -m scripts.precompute_embeddings \
  --openrouter \
  --model-name google/gemini-embedding-2 \
  --subset nondisjoint \
  --split train \
  --limit 100
```

Cache viene salvata sotto
`precomputed_embeddings/openrouter-google-gemini-embedding-2/`. Passare questa
cartella a `--embedding-root` durante training CP.
