# SOTA 2025 Event-Driven NILM: Ricerca & Allineamento Flusso Dati

**Obiettivo:** Ricerca sistemi SOTA 2025 event-driven, verifica allineamento flusso **raw → preprocessing → pretraining → modello** e rimozione incongruenze.

---

## 1. Ricerca SOTA 2025 – Event-Driven NILM

### Riferimenti principali (2024–2025)

| Fonte | Metodo | Dataset | Risultati | Note |
|-------|--------|---------|-----------|------|
| **Xue et al. 2024** | Z-score + XGBoost + Confidence Gating | Reale (Edge) | 92.6% acc, F1 0.74, &lt;25 ms/evento | Baseline architetturale del progetto |
| **Framework 2025 (PLAID)** | Z-score + XGBoost + SHAP | PLAID | 90% acc, sub-second latency | Explainability |
| **TFED – Kaddour et al. 2024** | Tukey Fences + FFT | NILMPEds | 99% accuracy event start | Alternativa al Z-score |
| **Gerasimov et al. 2025** | XGBoost event-based | PLAID | 90% | Conforme a nostro stack |
| **NILMTK / NILMPEds** | Standard preprocessing | Multi | 23+ metriche, 47k modelli | Best practice pipeline |

### Trend SOTA

- **Event-driven > time-based** per NILM a 1 Hz: focus su transitori, ~99% riduzione predizioni.
- **Z-score adattivo** (multi-risoluzione, soglia dinamica) e **TFED** (Tukey + FFT) sono gli approcci più citati per event detection.
- **XGBoost + confidence gating + KNN fallback** è lo stack standard per classificazione eventi (Xue, Gerasimov, framework 2025).
- **Feature:** ΔP, rise time, steady-state stats, spectral (FFT, entropy), wavelet, temporali, cross-event. Il nostro set da 41 feature è **allineato SOTA**.
- **Explainability:** SHAP per interpretabilità è raccomandato (framework 2025).

### Raccomandazioni per il nostro sistema

1. **Event detection:** Z-score multi-risoluzione (implementato) è SOTA-aligned. Opzionale: valutare **TFED** (Tukey + FFT) per confronto su accuratezza event start.
2. **Classification:** XGBoost + gating + KNN (implementato) è production-ready. Mantenere.
3. **Preprocessing:** Allineare a NILMTK-style (resample, gap handling, filter) dove possibile; confermare unità (kW vs W) end-to-end.
4. **Pipeline:** Raw → preprocessing → **solo** event detection → features → XGBoost. Il **pretraining** (Transformer / DL) è un **flusso separato** (sequence-based), non usato dall’event-based classifier.

---

## 2. Mappa Flusso Dati (Raw → Preprocessing → Pretraining → Modello)

### 2.1 Schema generale

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  RAW                                                                             │
│  • data/raw/1sec (CSV)        → 1 Hz (periodi selezionati)                       │
│  • data/raw/15min (Excel→CSV) → 15 min                                           │
└─────────────────────────────────────────────────────────────────────────────────┘
                    │                                    │
                    ▼                                    ▼
┌───────────────────────────────────┐    ┌─────────────────────────────────────────┐
│  PREPROCESSING 1sec_new           │    │  PREPROCESSING 15min / 10sec             │
│  • data_preparation_1sec_new      │    │  • 15min: nilm_ready_dataset.parquet     │
│  • Output: nilm_ready_1sec_new    │    │  • 10sec: nilm_10sec_mar_may.parquet     │
│    .parquet (1 Hz, 37M rows)      │    │  (uso: model_exploration, pretrain 10s)  │
│  • Cols: Time, Aggregate, 11 app, │    │                                          │
│    hour/dow/month sin/cos         │    │                                          │
└───────────────────────────────────┘    └─────────────────────────────────────────┘
                    │                                    │
                    │                                    │
        ┌───────────┴───────────┐                        │
        ▼                       ▼                        ▼
┌───────────────────┐   ┌─────────────────────────────────────────────────────────┐
│  EVENT-BASED      │   │  PRETRAINING (Transformer / DL)                          │
│  CLASSIFICATION   │   │  • nilm_pretraining_1sec_new: nilm_ready_1sec_new        │
│  (SOTA 2025)      │   │    → resample (es. 5 s) → model_ready/ (X,y, scaler)     │
│                   │   │  • nilm_pretraining_15min: 15min parquet                 │
│  Input:           │   │    → model_ready/ (per appliance o heatpump)             │
│  nilm_ready_1sec_ │   │  • nilm_pretraining_1sec (10s): 10sec parquet            │
│  new.parquet      │   │    → 10sec model_ready/                                  │
│                   │   │  Output: model_ready/ (numpy, scaler) per Transformer,   │
│  01 Detection     │   │  NILMformer, CNN/LSTM notebook, ecc.                     │
│  02 Features      │   │  ⚠️ Non usato dalla pipeline event-based XGBoost         │
│  03 Train XGB     │   │                                                          │
└───────────────────┘   └─────────────────────────────────────────────────────────┘
```

### 2.2 Pipeline event-based (classification)

| Stage | Input | Output | Path / Note |
|-------|--------|--------|-------------|
| **01_event_detection** | `nilm_ready_1sec_new.parquet` (Time, Aggregate) | `detected_events.csv` | Output in `classification/` |
| **02_feature_extraction** | `detected_events.csv` + stesso parquet (Aggregate) | `event_features.csv` | Idem |
| **03_train_classifier** | `event_features.csv` + parquet (Time, 11 appliances) | `xgb_nilm_model*.json/pkl`, `model_metadata.json`, `labeled_events.csv` | Idem |
| **04_error_analysis** | `event_features.csv`, parquet, `model_metadata.json`, modello | Report, confusion matrix | Legge da `classification/` e `data/` |
| **05_inference** | Modello + parquet (demo) | Predizioni | Modello in `classification/` |
| **06_visualize** | Modello, `event_features`, parquet | Plot | Idem |

**Unità:** Aggregate (e delta) in **kW** in tutto il flusso event-based. Soglie (es. `min_power_diff=0.020`) in kW (= 20 W).

### 2.3 Pretraining vs classification

- **Pretraining** crea `model_ready/` (numpy, scaler) per modelli **sequence-based** (Transformer, CNN, LSTM, ecc.).
- **Classification event-based** usa **solo** `nilm_ready_1sec_new.parquet` e i CSV intermedi.
- **Nessun flusso** da pretraining → event-based classifier: sono pipeline **distinte** per design.

### 2.4 Risoluzioni e dataset

| Risoluzione | Dataset | Uso |
|-------------|---------|-----|
| **1 Hz** | `nilm_ready_1sec_new.parquet` | Event-based classification (01–06) |
| **5–10 s** | `model_ready` da 1sec_new | Pretraining Transformer / DL su 1 s |
| **10 s** | `nilm_10sec_mar_may.parquet` | Pretraining 10 s, unify script |
| **15 min** | `nilm_ready_dataset` (15min) | Model exploration, pretraining 15 min, visualize_15min |

---

## 3. Incongruenze Rilevate e Fix

### 3.1 Path degli output (run_full_pipeline vs script)

- **Problema:** `run_full_pipeline` esegue 01/02/03 con `cwd=classification/`; gli script scrivono in cwd (`detected_events.csv`, `event_features.csv`, modello, metadata). `check_outputs` e `print_summary` cercavano in `BASE_DIR` (root) → **NOT FOUND**.
- **Fix:** Cercare gli output in `CLASSIFICATION_DIR` (es. `BASE_DIR / "classification"`). `run_full_pipeline` usa `CLASSIFICATION_DIR` per `check_outputs` e `print_summary`.

### 3.2 validate_pipeline

- **Problema:** Cercava `detected_events.csv`, `event_features.csv`, modello e metadata nella **root** del progetto.
- **Fix:** Cercare gli output della pipeline in `classification/` (stesso `CLASSIFICATION_DIR`).

### 3.3 04_error_analysis

- **Problema:** Path misti (relative da root vs `classification/`): `data/processed/...`, `classification/model_metadata.json`, `event_features.csv`, `classification/xgb_nilm_model.json`. Fallisce se run da `classification/` o con struttura diversa.
- **Fix:** Usare `BASE_DIR` e `CLASSIFICATION_DIR` derivati da `Path(__file__)`, e path assoluti per parquet, metadata, modello, `event_features`, output (es. confusion matrix).

### 3.4 Unità (kW) e soglie

- **Stato:** Aggregate e delta in **kW**. `min_power_diff=0.020` ⇔ 20 W, soglie di match 0.030 / 0.050 kW, ecc. **Coerente** in 01/02/03/04/sota_2025_fixes.
- **Azione:** Nessun fix; documentare in README/ARCHITECTURE che i dati sono in kW.

### 3.5 04 vs 03 match_labels

- **04** usa `tolerance_sec=10`, `min_on_power=0.015`; **03** usa `tolerance_sec=5` (o 10 in alcune versioni), `min_on_power` 0.020 (20 W). Possibile discrepanza tra labeled set in 03 e 04.
- **Raccomandazione:** Allineare parametri di match tra 03 e 04 (stesso `tolerance_sec`, `min_on_power`, soglie di error) per coerenza evaluation vs training.

---

## 4. Checklist Allineamento SOTA

- [x] Event detection: Z-score multi-risoluzione, adattivo, debounce (SOTA-aligned).
- [x] Feature extraction: transienti, steady-state, spectral, wavelet, temporali, cross-event (41 feature).
- [x] Classificatore: XGBoost + confidence gating + KNN fallback (Xue-like).
- [x] Unità: kW end-to-end per power; soglie coerenti.
- [x] Input unico event-based: `nilm_ready_1sec_new.parquet`.
- [ ] **Output unificati in `classification/`** e script/validate/run_full_pipeline che li usano in modo coerente (fix path).
- [ ] **04_error_analysis:** path robusti da `BASE_DIR` / `CLASSIFICATION_DIR`.
- [ ] **Match labels:** stessi parametri in 03 e 04.
- [ ] (Opzionale) **TFED (Tukey Fences + FFT)**: Kaddour et al. 2024 (arXiv:2402.17809) raggiunge 99% accuracy sull’event start; confrontare con Z-score multi-risoluzione per possibili gain.
- [ ] (Opzionale) ONNX export + quantizzazione per edge (target &lt;50 MB, &lt;20 ms).

---

## 5. Riferimenti

1. Xue et al. 2024 – Edge-Cloud NILM, XGBoost, confidence gating.
2. Kaddour et al. 2024 – TFED, Tukey Fences + FFT, arXiv:2402.17809.
3. NILMTK – Preprocessing e metriche standard.
4. NILMPEds – Benchmark event detection, 23+ metriche.
5. Gerasimov et al. 2025 – XGBoost event-based, PLAID 90%.
6. docs: `NILM_EventBased_SOTA_2025.md`, `SOTA_Edge_NILM_XGBoost_2025.md`.

---

**Team:** MTS3-MCTE Energy G1 · **Data:** Gennaio 2026 · **Versione:** 1.0
