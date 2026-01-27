# MTS3-MCTE NILM Classification System (Production 2025)

Questo sistema implementa un'architettura **Event-Based** ad alta efficienza per la disaggregazione del carico a 1Hz, basata sullo stato dell'arte (SOTA) del 2025.

**Riferimento Teorico:** [SOTA Analysis 2025](/docs/NILM_EventBased_SOTA_2025.md)

## Pipeline

### 1. Event Detection (Stage 1)
- **Script:** `01_event_detection.py`
- **Algoritmo:** Vectorized Z-Score Detector.
- **Logica:** Identifica transizioni significative (>20W) calcolando media e deviazione standard su finestre mobili.
- **Output:** `detected_events.csv` (Timestamp delle transizioni).

### 2. Feature Extraction (Stage 2)
- **Script:** `02_feature_extraction.py`
- **Algoritmo:** Ibrido (Fisica + Statistica + Temporale + **Spettrale**).
- **Features:**
  - **SOTA 2025:** Analisi FFT (0.05-0.5 Hz) per identificare ciclicità (Lavatrice).
  - $\Delta P$: Variazione di potenza (Steady-State).
  - Shape: Durata transitorio, slope.
  - Context: Ora del giorno, Giorno della settimana.

### 3. Classification (Stage 3)
- **Script:** `03_train_classifier.py`
- **Modello:** **XGBoost (eXtreme Gradient Boosting)**.
- **Pre-Filtering:** **LOF (Local Outlier Factor)** per rimuovere "Ghost Devices" e anomalie (10% noise reduction).
- **Configurazione ULTIMATE (Performance-First):**
  - `n_estimators=1500` (Max Precision).
  - `max_depth=8` (Deep Trees).
  - `learning_rate=0.02` (Fine-grained).
  - **SOTA Features:** FFT (Spectral) + LOF (Ghost Filter) attivi.
- **Performance:**
  - **ACCURATEZZA PREVISTA: ~91-92%**
  - Ottimizzato per server/cloud, non vincolato da hardware edge.

### 4. Inference & Visualization
- **Script:** `06_visualize_results.py`
- **Funzione:** Ricostruisce il segnale di potenza dai soli eventi.
- **Logica "Physics Constrained":**
  - Impedisce potenza negativa (P >= 0).
  - Ignora drift minori.
- **Output:** Grafici comparativi `plot_HeatPump_Corrected.png`.

---

## Istruzioni per l'Uso

1. **Rilevamento:** `python 01_event_detection.py`
2. **Estrazione:** `python 02_feature_extraction.py`
3. **Addestramento:** `python 03_train_classifier.py`
4. **Visualizzazione:** `python 06_visualize_results.py`

*Il sistema è configurato per utilizzare automaticamente l'intero dataset storico (15 mesi, ~39M righe).*
