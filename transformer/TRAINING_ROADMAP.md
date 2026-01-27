# Training Roadmap - Hybrid CNN-Transformer NILM

## Filosofia

> **Prima funziona, poi ottimizza.**

Approccio iterativo: baseline pulita → metriche → ablation informata.

---

## Fase 1: Baseline Training

**Obiettivo:** Stabilire reference point con architettura attuale.

```bash
cd transformer
python train.py --epochs 100 --batch_size 64
```

**Config:**
- 7M parametri
- Neighborhood regression (±5)
- Confidence-gated energy loss
- Causal stationarization

**Metriche da registrare:**
- [ ] Loss curve (train/val)
- [ ] F1 per appliance
- [ ] MAE, SAE per appliance
- [ ] Tempo training/epoch
- [ ] Tempo inference (samples/sec)

---

## Fase 2: Analisi Risultati

| Osservazione | Possibile Problema | Azione |
|--------------|-------------------|--------|
| F1 basso su pattern giornalieri | RoPE non cattura tempo | Aggiungi TimeRPE |
| MAE alto su transitori | CNN troppo piccola | Dilated conv / kernel larger |
| Energy loss non converge | Weight troppo alto | Riduci a 0.05 |
| Inference lenta | O(n²) attention | LinearAttention / Mamba |

---

## Fase 3: Ablation Study

| Esperimento | Modifica | Confronto |
|-------------|----------|-----------|
| **A** | RoPE → ALiBi | Stabilità su sequenze lunghe |
| **B** | Cumsum → EMA (α=0.001) | Adattamento a drift |
| **C** | Energy weight 0.1 → 0.05 | Convergenza |
| **D** | Neighborhood ±5 → ±10 | Robustezza streaming |

---

## Fase 4: Modello Finale

Dopo ablation, implementa solo i vincitori. Documenta:

```markdown
## Modello Produzione
- Ablation A: [risultato]
- Ablation B: [risultato]
- Config finale: ...
```

---

## Checklist Pre-Training

- [ ] Dati 1Hz preprocessati (`nilm_ready_1sec.parquet`)
- [ ] GPU disponibile
- [ ] Logging configurato
- [ ] Early stopping attivo

---

## Comandi Rapidi

```bash
# Training
python train.py --epochs 100 --batch_size 64

# Solo test
python test_model.py

# Benchmark velocità
python benchmark.py
```
