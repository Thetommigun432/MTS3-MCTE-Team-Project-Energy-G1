# 📊 Analisi Completa Problemi Dataset 1-Second

> **Data analisi:** Gennaio 2026  
> **Dataset:** Dati energetici residenziali alta risoluzione (1sec/10sec)  
> **Periodo:** Marzo - Dicembre 2024  
> **Files analizzati:** 10  

---

## Indice
1. [Problemi Critici (Bloccanti)](#1-problemi-critici-bloccanti)
2. [Problemi di Qualità Dati](#2-problemi-di-qualità-dati)
3. [Problemi di Completezza](#3-problemi-di-completezza)
4. [Problemi di Schema/Struttura](#4-problemi-di-schemastruttura)
5. [Problemi di Risoluzione Temporale](#5-problemi-di-risoluzione-temporale)
6. [Problemi NILM-Specifici](#6-problemi-nilm-specifici)
7. [Riepilogo per File](#7-riepilogo-per-file)
8. [Raccomandazioni](#8-raccomandazioni)

---

## 1. Problemi Critici (Bloccanti)

### 1.1 🔴 Building (Aggregate) Mancante

**Severità:** CRITICA - Impedisce NILM  
**Files affetti:** 7/10 (Giugno-Dicembre 2024)

| File | Building NULL% | Status |
|------|---------------|--------|
| samengevoegd_2024-03.csv | 2.6% | ✅ Usabile |
| samengevoegd_2024-04.csv | 0.0% | ✅ Usabile |
| samengevoegd_2024-05.csv | 0.0% | ✅ Usabile |
| samengevoegd_2024-06.csv | **89.1%** | ❌ Non usabile |
| samengevoegd_2024-07.csv | **100.0%** | ❌ Non usabile |
| samengevoegd_2024-08.csv | **100.0%** | ❌ Non usabile |
| samengevoegd_2024-09.csv | **100.0%** | ❌ Non usabile |
| samengevoegd_2024-10.csv | **100.0%** | ❌ Non usabile |
| samengevoegd_2024-11.csv | **100.0%** | ❌ Non usabile |
| samengevoegd_2024-12.csv | **100.0%** | ❌ Non usabile |

**Dettaglio:** Il sensore Building ha smesso di registrare il **4 Giugno 2024 alle 06:42:17**.

**Impatto:** Senza il consumo aggregato della casa, NILM è impossibile. La disaggregazione energetica richiede:
- Input: Consumo totale (Building)
- Output: Consumo per singolo apparecchio

**Workaround tentato:** Ricostruire Building = Σ(apparecchi)  
**Risultato:** ❌ Non praticabile - Ghost load 47-70% (vedi sezione 6.1)

---

### 1.2 🔴 Smappee_laadpaal (Caricatore EV) NULL

**Severità:** ALTA  
**Files affetti:** Marzo-Agosto 2024

| File | Smappee_laadpaal NULL% |
|------|------------------------|
| samengevoegd_2024-03.csv | 100.0% |
| samengevoegd_2024-04.csv | 100.0% |
| samengevoegd_2024-05.csv | 100.0% |
| samengevoegd_2024-06.csv | 100.0% |
| samengevoegd_2024-07.csv | 100.0% |
| samengevoegd_2024-08.csv | 72.2% |
| samengevoegd_2024-09.csv | 0.0% ✅ |
| samengevoegd_2024-10.csv | 0.0% ✅ |
| samengevoegd_2024-11.csv | 0.0% ✅ |
| samengevoegd_2024-12.csv | 0.0% ✅ |

**Impatto:** Per i mesi usabili (Mar-Mag), questo dispositivo non è disponibile. Riduce gli apparecchi disaggregabili da 9 a 8.

---

## 2. Problemi di Qualità Dati

### 2.1 🟠 Valori Negativi Sistematici

**Severità:** MEDIA - Risolvibile con clipping  
**Causa probabile:** Offset del sensore CT (Current Transformer)

| Dispositivo | % Valori Negativi | Media (kW) | Interpretazione |
|-------------|-------------------|------------|-----------------|
| **Fornuis** | 97-99% | -0.005 a -0.010 | Offset CT ~10W |
| **Oven** | 71-99% | -0.004 a -0.012 | Offset CT ~10W |
| **Vaatwasser** | 86-95% | +0.002 a +0.005 | Offset minore |
| **Wasmachine** | 95-97% | +0.009 a +0.022 | Offset + consumo standby |
| **Regenwaterpomp** | 0.2-46% | ~0 | Variabile per mese |

**Soluzione:** Applicare `clip(lower=0)` durante preprocessing.

**Nota:** Questo pattern è identico ai dati 15min, dove la stessa correzione è stata applicata con successo.

---

### 2.2 🟠 Valori Costanti Prolungati

**Severità:** BASSA - Comportamento normale  
**Osservazione:** Molti dispositivi mostrano valori identici per periodi prolungati.

| Dispositivo | % Valori Identici al Precedente | Interpretazione |
|-------------|--------------------------------|-----------------|
| Dampkap | 99.2% | Quasi sempre OFF |
| Warmtepomp | 94.1% | Consumo stabile quando ON |
| Wasmachine | 96.3% | Fasi a potenza costante |
| Droogkast | 99.8% | Quasi sempre OFF |

**Interpretazione:** NON è un problema di campionamento. I dispositivi:
- Sono spesso OFF (consumo = 0)
- Hanno fasi a potenza costante (es. resistenza riscaldamento)
- Cambiano stato raramente in 1 secondo

**Conclusione:** I dati sono corretti, il campionamento a 1sec cattura la realtà fisica.

---

### 2.3 🟡 Picchi Anomali

**Severità:** BASSA  
**Osservazione:** Alcuni dispositivi mostrano picchi molto alti.

| Dispositivo | Max (kW) | Plausibile? |
|-------------|----------|-------------|
| Fornuis | 3.66 | ✅ Sì (piano induzione) |
| Wasmachine | 2.04 | ✅ Sì (resistenza) |
| Warmtepomp | 1.77 | ✅ Sì (compressore) |
| Vaatwasser | 1.32 | ✅ Sì (resistenza) |
| Droogkast | 1.40 | ✅ Sì (resistenza) |
| Smappee_laadpaal | 10.97 | ⚠️ Alto ma plausibile (11kW charger) |

**Conclusione:** Nessun picco anomalo rilevato.

---

## 3. Problemi di Completezza

### 3.1 🟠 Completezza Temporale Variabile

**Severità:** MEDIA

| File | Righe Presenti | Righe Attese | Completezza | Gap Massimo |
|------|---------------|--------------|-------------|-------------|
| 2024-03 | 154,236 | 153,029 | 100.8% | 2.23 ore |
| 2024-04 | 218,819 | 259,199 | **84.4%** | **43.92 ore** |
| 2024-05 | 1,080,769 | 2,678,398 | **40.4%** | 0.02 ore |
| 2024-06 | 2,590,063 | 2,591,999 | 99.9% | 0.35 ore |
| 2024-07 | 2,673,691 | 2,678,399 | 99.8% | 1.10 ore |
| 2024-08 | 2,677,884 | 2,678,399 | 100.0% | 0.10 ore |
| 2024-09 | 2,590,681 | 2,591,999 | 99.9% | 0.07 ore |
| 2024-10 | 2,046,753 | 2,678,399 | **76.4%** | **172.74 ore** |
| 2024-11 | 2,589,683 | 2,591,999 | 99.9% | 0.34 ore |
| 2024-12 | 2,675,505 | 2,678,399 | 99.9% | 0.16 ore |

**Problemi specifici:**
- **Aprile 2024:** Gap di 44 ore (6-8 Aprile) e 42 ore (13-15 Aprile)
- **Maggio 2024:** Solo 40% dei dati presenti (risoluzione mista 1sec/10sec)
- **Ottobre 2024:** Gap di 173 ore (2-9 Ottobre)

---

### 3.2 🟡 Gap Temporali Significativi

**Severità:** BASSA - Gestibile con interpolazione o esclusione

| File | # Gap > 1 ora | Gap Massimo | Periodo Mancante |
|------|--------------|-------------|------------------|
| 2024-04 | 2 | 43.9 ore | 6-8 Apr, 13-15 Apr |
| 2024-07 | 1 | 1.1 ore | 31 Lug mattina |
| 2024-10 | 1 | 172.7 ore | 2-9 Ott |

**Soluzione:** Durante il preprocessing, questi gap possono essere:
1. Interpolati (se < 1 ora)
2. Esclusi dalla finestra di training
3. Marcati come boundary per sequenze

---

## 4. Problemi di Schema/Struttura

### 4.1 🟠 Dispositivi Mancanti rispetto a 15min

**Severità:** ALTA - Riduce capacità di disaggregazione

| Dispositivo | Presente in 15min | Presente in 1sec | Consumo Medio |
|-------------|-------------------|------------------|---------------|
| Grid | ✅ | ❌ | N/A (non apparecchio) |
| Zonne-energie | ✅ | ❌ | N/A (produzione) |
| **Kast garage** | ✅ | ❌ | **~0.27 kW** |
| **Laadpaal_stopcontact** | ✅ | ❌ | ~0.01 kW |
| **Warmtepomp-Sturing** | ✅ | ❌ | ~0.01 kW |

**Impatto:** 
- 3 apparecchi consumatori non sono misurati
- Contribuiscono a ~0.29 kW di "ghost load"
- Impossibile disaggregarli nel modello 1sec

**Confronto:**
- **15min:** 12 colonne (Building + 11 apparecchi)
- **1sec:** 11 colonne (Building + 9 apparecchi + Smappee_laadpaal)

---

### 4.2 🟡 Formato Dati Diverso

**Severità:** BASSA - Solo preprocessing diverso

| Aspetto | 15min | 1sec |
|---------|-------|------|
| Formato | Long (msr_subject) | Wide (colonne) |
| Unità | W | kW |
| Timezone | UTC+0 | UTC+0 |
| Separatore | , | , |

**Impatto:** Il preprocessing per 1sec è più semplice (già pivotato), ma serve conversione unità.

---

## 5. Problemi di Risoluzione Temporale

### 5.1 🟠 Risoluzione Mista tra File

**Severità:** MEDIA

| File | Risoluzione Principale | % Campioni |
|------|------------------------|------------|
| 2024-03 | **10sec** | 98.1% |
| 2024-04 | **10sec** | 99.9% |
| 2024-05 | **1sec** (misto) | 83.5% 1sec, 16.3% 10sec |
| 2024-06 | 1sec | 100.0% |
| 2024-07 | 1sec | 100.0% |
| 2024-08 | 1sec | 100.0% |
| 2024-09 | 1sec | 100.0% |
| 2024-10 | 1sec | 99.9% |
| 2024-11 | 1sec | 100.0% |
| 2024-12 | 1sec | 100.0% |

**Problema:** Marzo e Aprile sono a 10sec, il resto a 1sec.

**Soluzione:** Ricampionare tutto a 10sec per uniformità:
- 1sec → 10sec: aggregazione (mean)
- 10sec → 10sec: nessuna modifica

---

### 5.2 🟡 Transizione Risoluzione in Maggio

**Severità:** BASSA

Il file di Maggio contiene un mix:
- 83.5% campioni a 1sec
- 16.3% campioni a 10sec

Questo suggerisce che il sistema di acquisizione è cambiato durante il mese.

---

## 6. Problemi NILM-Specifici

### 6.1 🔴 Ghost Load Elevato

**Severità:** CRITICA per ricostruzione Building

**Definizione:** Ghost Load = Building - Σ(apparecchi misurati)

| File | Building Medio | Σ Apparecchi | Ghost Load | Ghost % |
|------|---------------|--------------|------------|---------|
| 2024-03 | 0.606 kW | 0.320 kW | 0.286 kW | **47.1%** |
| 2024-04 | 0.531 kW | 0.276 kW | 0.255 kW | **48.1%** |
| 2024-05 | 0.373 kW | 0.112 kW | 0.261 kW | **70.0%** |

**Causa:** Apparecchi non misurati (Kast garage ~0.27 kW + altri)

**Implicazione:** NON è possibile ricostruire Building da Σ(apparecchi):
- Mancherebbe 47-70% del consumo
- Il modello imparerebbe relazioni errate

---

### 6.2 🟠 Correlazione Building vs Σ Apparecchi

**Severità:** MEDIA

| File | Correlazione | RMSE | Interpretazione |
|------|-------------|------|-----------------|
| 2024-03 | 0.834 | - | Discreta |
| 2024-04 | 0.850 | - | Discreta |
| 2024-05 | 0.803 | - | Bassa |

**Target minimo per NILM:** Correlazione > 0.90

**Conclusione:** La correlazione non è sufficiente per ricostruzione affidabile.

---

### 6.3 🟡 Pattern Temporale Ghost Load

**Severità:** INFORMATIVO

| Ora | Building | Σ Appar. | Ghost | Ghost % |
|-----|----------|----------|-------|---------|
| 00-05 | 0.16 kW | 0.01 kW | 0.15 kW | **95-96%** |
| 08-12 | 0.40-1.38 kW | 0.05-1.09 kW | 0.21-0.39 kW | 20-80% |
| 18-23 | 0.20-0.35 kW | 0.01-0.05 kW | 0.19-0.34 kW | **96-98%** |

**Interpretazione:**
- Di notte: consumo base non misurato (Kast garage, standby vari)
- Di giorno: apparecchi misurati più attivi, ghost % più basso
- Sera: ritorno a consumo base

---

### 6.4 🟡 Perdita Informazione con 15min

**Severità:** MEDIA - Dipende dall'obiettivo

| Dispositivo | Max 1sec | Max 15min | Perdita % | Picchi Persi |
|-------------|----------|-----------|-----------|--------------|
| Fornuis | 3.66 kW | 1.23 kW | **66.5%** | 20 |
| Wasmachine | 2.04 kW | 1.33 kW | **34.9%** | 588 |
| Vaatwasser | 1.32 kW | 0.88 kW | **33.2%** | 23 |
| Warmtepomp | 1.65 kW | 1.49 kW | 9.3% | 94 |
| Oven | 0.80 kW | 0.76 kW | 5.4% | 0 |

**Conclusione:** Per Fornuis, Wasmachine, Vaatwasser, l'alta risoluzione cattura picchi significativi che 15min perde.

---

## 7. Riepilogo per File

### File Usabili ✅

| File | Righe | Giorni | Risoluzione | Building | Problemi |
|------|-------|--------|-------------|----------|----------|
| **2024-03** | 154K | 17.7 | 10sec | 97% | Gap 2h, Smappee NULL |
| **2024-04** | 219K | 30 | 10sec | 100% | Gap 44h, Smappee NULL |
| **2024-05** | 1.08M | 31 | 1sec | 100% | 40% completezza, Smappee NULL |

**Totale usabile:** 1.45M righe, ~79 giorni

### File Non Usabili ❌

| File | Motivo Principale |
|------|-------------------|
| 2024-06 | Building 89% NULL |
| 2024-07 | Building 100% NULL |
| 2024-08 | Building 100% NULL |
| 2024-09 | Building 100% NULL |
| 2024-10 | Building 100% NULL |
| 2024-11 | Building 100% NULL |
| 2024-12 | Building 100% NULL |

---

## 8. Raccomandazioni

### 8.1 Per il Progetto Scolastico

**OPZIONE RACCOMANDATA: Approccio Comparativo**

```
┌─────────────────────────────────────────────────────────────┐
│  PIPELINE 1: 15min (Seq2Seq)                                │
│  ─────────────────────────────                              │
│  • Dataset: 1 anno completo (35,040 righe)                  │
│  • Apparecchi: 12                                           │
│  • Pro: Dati completi, pipeline già funzionante             │
│  • Architettura: Seq2Seq (già implementato)                 │
└─────────────────────────────────────────────────────────────┘
                              +
┌─────────────────────────────────────────────────────────────┐
│  PIPELINE 2: 10sec (Transformer)                            │
│  ────────────────────────────────                           │
│  • Dataset: Mar-Apr-Mag (~80 giorni)                        │
│  • Apparecchi: 8 (senza Smappee)                            │
│  • Pro: Maggiore dettaglio temporale                        │
│  • Architettura: Transformer (nuovo)                        │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  CONFRONTO RISULTATI                                        │
│  ───────────────────                                        │
│  • MAE, RMSE, F1-score per apparecchio                      │
│  • Analisi: quando 10sec è migliore di 15min?               │
│  • Dispositivi: Fornuis, Wasmachine beneficiano di 10sec    │
│  • Dispositivi: Warmtepomp non beneficia                    │
└─────────────────────────────────────────────────────────────┘
```

### 8.2 Preprocessing 1sec/10sec

1. **Caricare solo file usabili:** Mar, Apr, Mag
2. **Unificare risoluzione a 10sec** (per consistenza)
3. **Clip valori negativi** a 0
4. **Rimuovere Smappee_laadpaal** (NULL)
5. **Gestire gap:** Escludere finestre che attraversano gap > 1 ora
6. **Normalizzare:** Same approach as 15min

### 8.3 Email per Jeroen

Punti da chiedere:
1. Il sensore Building ha smesso di registrare il 4 Giugno 2024 - è possibile recuperare i dati?
2. Esistono altri dataset con Building valido per periodi più lunghi?
3. Perché Kast garage, Laadpaal_stopcontact, Warmtepomp-Sturing non sono nei dati 1sec?

---

## Appendice: Statistiche Dispositivi (Maggio 2024)

| Dispositivo | Media (kW) | Max (kW) | ON% | Accensioni | Durata Media |
|-------------|------------|----------|-----|------------|--------------|
| Dampkap | 0.0005 | 0.23 | 0.2% | 8 | 4.7 min |
| Droogkast | 0.0038 | 1.40 | 0.1% | 46 | 0.2 min |
| Fornuis | 0.0060 | 3.66 | 1.1% | 168 | 1.2 min |
| Oven | 0.0121 | 0.80 | 1.6% | 79 | 3.7 min |
| Regenwaterpomp | 0.0002 | 0.53 | 0.0% | 42 | 0.2 min |
| Vaatwasser | 0.0053 | 1.32 | 1.7% | 1,512 | 0.2 min |
| Warmtepomp | 0.0655 | 1.65 | 5.1% | 57 | 16.0 min |
| Wasmachine | 0.0183 | 2.04 | 3.2% | 3,244 | 0.2 min |

---

*Report generato automaticamente da analisi Python*
