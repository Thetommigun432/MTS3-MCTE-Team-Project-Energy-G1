# Analisi Avanzata delle Architetture Event-Based per il Monitoraggio Non Intrusivo del Carico (NILM) a 1Hz
## Prospettive Tecnologiche 2025-2026

**Stato del Progetto:** Questo documento rappresenta il riferimento teorico per l'architettura implementata nel sistema MTS3-MCTE (XGBoost + Z-Score 1Hz).

---

### Introduzione: Il Paradigma del Monitoraggio Energetico a Bassa Frequenza

La transizione energetica globale e la crescente penetrazione delle risorse energetiche distribuite impongono una gestione sempre più granulare della domanda elettrica. In questo contesto, il Monitoraggio Non Intrusivo del Carico (Non-Intrusive Load Monitoring - NILM), noto anche come disaggregazione dell'energia, si afferma come la tecnologia chiave per decomporre il consumo aggregato di un edificio, misurato da un unico punto (il contatore intelligente), nei profili di consumo dei singoli elettrodomestici.

L'evoluzione tecnologica nel biennio 2025-2026 segna un punto di svolta critico per il NILM, caratterizzato dal passaggio da approcci euristici tradizionali a sofisticate architetture di intelligenza artificiale capaci di operare in ambienti con risorse limitate e dati a bassa risoluzione.

La sfida centrale affrontata in questo rapporto risiede nella natura dei dati: la maggior parte dell'infrastruttura di metering avanzata (AMI) installata a livello globale opera con frequenze di campionamento pari o inferiori a 1 Hz (un campione al secondo). Questa limitazione fisica impone vincoli severi alle metodologie applicabili. A differenza dei sistemi ad alta frequenza (kHz o MHz), i sistemi a 1Hz devono estrarre informazione quasi esclusivamente dalle variazioni macroscopiche di Potenza Attiva (P) e, ove disponibile, Potenza Reattiva (Q).

---

### 1. Fondamenti e Innovazioni nel Rilevamento Eventi a 1Hz

Il successo di qualsiasi architettura event-based dipende interamente dalla qualità del primo stadio: il rilevamento dell'evento. In un flusso di dati a 1Hz, un evento è definito come una variazione significativa e persistente nei livelli di potenza.

#### 1.1 Rilevamento Adattivo (Implementato)
Storicamente, il rilevamento degli eventi si basava su algoritmi a soglia fissa (es. 30 Watt). Tuttavia, questo approccio fallisce con gli inverter.
Nel 2025, lo stato dell'arte si è spostato verso algoritmi di **Rilevamento Adattivo** che adattano dinamicamente la larghezza della finestra di osservazione e le soglie in base alla varianza del segnale di fondo.
La metodologia "Adaptive Window" cattura transizioni complete, fondamentali per l'estrazione corretta delle feature ($\Delta P$).

#### 1.2 Rilevamento Statistico: Z-Score (La Scelta di Progetto)
Parallelamente agli approcci geometrici, si sono affermati metodi statistici robusti come i rilevatori basati sul **Z-Score**. Questi algoritmi calcolano la media mobile e la deviazione standard del segnale di potenza in tempo reale.
L'efficacia di questo metodo risiede nella sua normalizzazione intrinseca: in periodi di alto consumo e alta varianza (es. lavatrice in centrifuga), il rilevatore diventa meno sensibile, prevenendo falsi positivi.
*Nel nostro progetto, utilizziamo un Z-Score Detector vettorizzato estremamente efficiente (Step 1 dell'architettura).*

---

### 2. Ingegneria delle Feature (Feature Engineering) per Dati a 1Hz

Le architetture vincenti nel 2025, come XGBoost, si basano su un set di feature ibrido:

1.  **Variazione di Stato ($\Delta P, \Delta Q$):** La differenza tra potenza stazionaria post-evento e pre-evento.
2.  **Durata del Transitorio:** Distingue luci LED (istantanee) da motori con inerzia.
3.  **Pattern Temporali:** L'ora del giorno e il giorno della settimana sono predittori potenti (es. Tostapane non alle 3:00, Frigo 24/7).
4.  **Momenti Statistici:** Media, Varianza, Skewness nella finestra post-evento (Steady-State texture).

*Queste sono esattamente le feature implementate in `classification/02_feature_extraction.py`.*

---

### 3. XGBoost: Il Re dell'Efficienza e dell'Interpretabilità

L'algoritmo **XGBoost (eXtreme Gradient Boosting)** domina il panorama delle soluzioni NILM event-based "leggere" nel 2025/2026.

#### 3.1 Vantaggi Competitivi
*   **Non-Linearità:** Modella confini decisionali complessi nel piano P-Q.
*   **Robustezza:** Gestisce nativamente outlier e dati mancanti.
*   **Efficienza:** Inferenza in 1-2 ms su microcontroller (ESP32/Raspberry Pi).
*   **Risultati:** Studi comparativi (2024-2025) mostrano che XGBoost batte LSTM e KNN su dati a bassa frequenza, specialmente con dataset limitati.

#### 3.2 Explainable AI (XAI) con SHAP
L'integrazione con SHAP permette di spiegare perché un evento è stato classificato come "Lavastoviglie" (es. "Durata > 45 min" ha contribuito +30%).

---

### 4. Deep Learning Sequenziale (TCN) e NLP (Transformers)

Mentre XGBoost eccelle sui dati tabulari degli eventi singoli, le architetture sequenziali sfruttano la dipendenza temporale tra eventi.

*   **TCN (Temporal Convolutional Networks):** Usano convoluzioni causali dilatate per "vedere" ore di storia. Su UK-DALE hanno raggiunto F1-score > 94% per carichi ciclici come il Frigo.
*   **Transformers (BERT/Energformer):** Trattano la sequenza di potenza come una "frase". I meccanismi di Self-Attention catturano correlazioni a lungo raggio (es. Pompa di calore spesso precede altri eventi).

---

### 5. Frontiere 2026: GNN e KAN

*   **GNN (Graph Neural Networks):** Modellano la casa come un grafo di apparecchi interdipendenti.
*   **KAN (Kolmogorov-Arnold Networks):** Nuove reti con funzioni di attivazione sugli archi, capaci di modellare non-linearità complesse con pochissimi parametri (ideali per edge computing).

---

### Conclusioni Operative per il Progetto

Per i ricercatori e gli sviluppatori che operano oggi, la raccomandazione è di adottare un **approccio ibrido**:
1.  **Rilevatore di Eventi Statistico Robusto (Z-Score)** → *Implementato*
2.  **Classificatore XGBoost (Edge/Real-time)** → *Implementato*
3.  **Integrazione XAI** per la trasparenza.

Questo documento conferma che la scelta tecnologica effettuata (Stack Z-Score + XGBoost) è pienamente allineata con lo stato dell'arte industriale del 2025.
