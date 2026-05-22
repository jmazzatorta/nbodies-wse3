# wse3-bench-multi — Guida pratica

Versione production multistep di N-body su WSE-3.
Senza simprint, con `<time>` library per misure di performance.
Parametrica su `side`, `N_LOCAL`, `N_STEPS`, `DT_NUM`, `DT_DEN`.

---

## File

| File | Cosa fa |
|---|---|
| `constants.csl` | Costanti, task IDs, params |
| `layout.csl` | Topologia ring serpentina, exports |
| `pe.csl` | Logica per-PE: pipeline + leapfrog + timing |
| `commands.sh` | Script di compilazione |
| `run.py` | Host launcher (push, launch, pull, decode timing) |
| `generate_bodies.py` | Genera bodies.npy |
| `validate_multistep.py` | Reference Python leapfrog f64 (per validazione esterna) |

---

## Workflow base

```bash
# 1. Genera particelle
python3 generate_bodies.py --n 1024

# 2. (Opzionale) Crea reference Python per validare
python3 validate_multistep.py --n_steps 10 --dt 0.0001

# 3. Compila
./commands.sh 32 1 10 1 10000

# 4. Esegui (simulator)
cs_python run.py --name out

# 5. Esegui (CS-3 reale)
export CS_FABRIC_DIMS=757,996         # da sostituire con valori reali del wafer
cs_python run.py --name out --cmaddr $CS_IP_ADDR:9000
```

---

## Parametri di compilazione (`commands.sh`)

```
./commands.sh SIDE N_LOCAL N_STEPS DT_NUM DT_DEN
```

| Arg | Valori tipici | Significato |
|---|---|---|
| `SIDE` | 32, 64, 128 | PE per lato. Totale PE = SIDE×SIDE |
| `N_LOCAL` | 1, 2, 4, 8, 16 | Particelle per PE |
| `N_STEPS` | 10, 100, 1000 | Step di leapfrog |
| `DT_NUM` | 1, 5 | Numeratore di dt |
| `DT_DEN` | 10000, 1000 | Denominatore di dt. dt = DT_NUM/DT_DEN |

**Calcolo N_TOTAL particelle**: `SIDE * SIDE * N_LOCAL`

Esempi:
- `./commands.sh 32 1 10 1 10000` → 1024 particelle, 10 step, dt=0.0001
- `./commands.sh 64 4 100 1 10000` → 16384 particelle, 100 step, dt=0.0001
- `./commands.sh 128 16 50 5 10000` → 262144 particelle, 50 step, dt=0.0005

---

## Parametri di esecuzione (`run.py`)

```
cs_python run.py --name out [opzioni]
```

| Opzione | Significato |
|---|---|
| `--name DIR` | Directory di compilazione (default `out`) |
| `--cmaddr IP:PORT` | IP del CS-3. Omettilo per simulator. |
| `--bodies FILE` | File bodies.npy (default `bodies.npy`) |
| `--results-json PATH` | File JSON con risultati strutturati (default `results.json`) |
| `--no-results-json` | Salta scrittura del JSON |

`bodies.npy` deve avere esattamente `SIDE*SIDE*N_LOCAL` righe. Se non corrisponde, errore.

---

## File generati

| File | Quando | Contenuto |
|---|---|---|
| `bodies.npy` | da `generate_bodies.py` | (N, 7) [x,y,z,vx,vy,vz,m] |
| `reference.npy` | da `validate_multistep.py` | (N, 10) reference Python f64 |
| `out/` | da `commands.sh` | Binari compilati per il device |
| `results.json` | da `run.py` | Configurazione + timing + throughput |
| `device_final.npy` | da `run.py` | (N, 10) stato finale del device |

---

## Output di `run.py`

### Stampa a video

1. **Header**: configurazione (SIDE, N_LOCAL, N_STEPS, DT, target).
2. **Per-step table**: per ogni step 0..N_STEPS-1, min/max/avg cycle counts di:
   - `t_self`: compute_self_forces + pack_own_chunk
   - `t_pipeline`: drain + emit + chunk_forces (cross-PE)
   - `t_total`: somma
3. **Aggregate**: stesse statistiche aggregate su tutti gli step.
4. **Wall-clock host**: tempi di push/launch/pull (host-side).
5. **Throughput stimato**: pair-interactions/sec.
6. **Sanity**: n bodies con forza non zero.

### `results.json`

Contiene tutti i numeri sopra in formato strutturato per script di analisi/plotting.

---

## Validazione esterna

Per verificare che il device sia corretto:

```bash
# 1. Genera reference Python (lo stesso bodies.npy, stesso N_STEPS/DT)
python3 validate_multistep.py --n_steps 10 --dt 0.0001

# 2. Compila e lancia (run.py salva il risultato in device_final.npy)
./commands.sh 32 1 10 1 10000
cs_python run.py --name out

# 3. Confronta device vs reference
python3 compare_results.py
```

`compare_results.py` stampa l'errore relativo su posizioni, velocità, forze.
Pass/fail vs tolleranza (default 1e-3).

---

## Decoding timing buffer

Per chi vuole capire come funziona internamente:

- Ogni PE esporta `time_buf_u16`: array di `9 * (N_STEPS+1)` u16.
- Per step `s` (0..N_STEPS):
  - `buf[9*s + 0..3]`: tsc_start[s] (3 u16, little-endian → u48)
  - `buf[9*s + 3..6]`: tsc_self[s]
  - `buf[9*s + 6..9]`: tsc_end[s]
- Per ogni PE: `t_self[s] = self[s] - start[s]`, `t_pipeline[s] = end[s] - self[s]`, `t_total[s] = end[s] - start[s]`.
- Host aggrega min/max/avg sui PE.

**Nota**: lo step `s = N_STEPS` ha solo `end[N_STEPS]` significativo (è solo half-kick finale, no pipeline). Le tabelle saltano questo step.

---

## CS-3 reale — checklist

1. **Imposta** `CS_FABRIC_DIMS` alle dimensioni reali del wafer (verifica con `cs_python -c "..."` se serve).
2. **Imposta** `CS_IP_ADDR` all'IP del CS-3.
3. **Compila**: `./commands.sh SIDE N_LOCAL N_STEPS 1 10000`.
4. **Lancia**: `cs_python run.py --name out --cmaddr $CS_IP_ADDR:9000`.
5. **Verifica** `results.json` per i timing.

Su CS-3 reale: tempi pull/launch trascurabili (~ms), `t_pipeline` è la vera misura.

---

## ⚠️ Note sul simulator

- `run.py` usa `suppress_simfab_trace=True` per evitare che il simulator generi file di trace giganteschi (decine/centinaia di GB per multistep).
- Se vedi errori tipo `fwrite assertion nmemb == 1`: probabilmente disco pieno per trace. Verifica `df -h` e `du -sh simfab_traces/`. Pulisci con `rm -rf simfab_traces/`.
- Su CS-3 reale questi problemi non esistono (non c'è simulator).
