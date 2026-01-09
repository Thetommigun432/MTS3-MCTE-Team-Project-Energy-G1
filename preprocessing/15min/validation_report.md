# Validation Report - NILM Preprocessing Pipeline

**Date**: January 8, 2026

---

## 0. Field Mapping (Requirements → Dataset)

| Requirement Name | Dataset Field(s) | Filter Applied | Status |
|------------------|------------------|----------------|--------|
| `msr_subject` | `msr_subject` column | N/A | ✅ Used for appliance identification |
| `msr_category` | `msr_category` column | `Power_active` only | ✅ Reactive power filtered (6%) |
| Grid | `msr_subject='Grid'` | Dropped | Redundant (Grid = Building - Solar) |
| Building - consumption - Total - Smappee | `msr_subject='Building'` + `msr_device='Smappee'` + `msr_spec='Consumption'` | Used as **Aggregate** | ✅ Main meter |
| Building - consumption - Total - Calculated | `msr_device='Calculated'` | Filtered out | Derived value, Smappee used instead |
| Building - production | `msr_subject='Zonne-energie'` (Solar) | Dropped | Pattern captured by hour_sin/cos |
| Building - power_reactive | `msr_category='Power_reactive'` | Filtered out | Not relevant for NILM |

---

## 1. Dataset Overview

| Metric | Value |
|--------|-------|
| Raw data rows | 630,662 |
| Output rows | 35,040 |
| Output columns | 20 (Time + Aggregate + 12 appliances + 6 temporal features) |
| Time range | 2024-10-20 → 2025-10-20 |
| Resolution | 15 minutes |
| Time gaps after interpolation | 0 |

---

## 2. Phase Structure (L1/L2/L3)

| Device | Rows/timestamp | Phases |
|--------|----------------|--------|
| Building | 3 | Total |
| Dampkap | 1 | L2 |
| Droogkast | 1 | L1 |
| Fornuis | 1 | L1_L2_L3 |
| Grid | 1 | L1_L2_L3 |
| Kast garage | 1 | L1_L2_L3 |
| Laadpaal_stopcontact | 1 | L1 |
| Oven | 1 | L1 |
| Regenwaterpomp | 1 | L1 |
| Smappee_laadpaal | 1 | L1_L2_L3 |
| Vaatwasser | 1 | L3 |
| Warmtepomp | 1 | L1 |
| Warmtepomp - Sturing | 1 | L2 |
| Wasmachine | 1 | L3 |
| Zonne-energie | 1 | L1 |

**Observation**: Building has 3 rows per timestamp due to multiple msr_device/msr_spec combinations. Data Preparation filters to `msr_device='Smappee'` and `msr_spec='Consumption'`.

**Observation**: 3-phase devices (L1_L2_L3) are pre-summed by Smappee hardware, resulting in 1 row per timestamp.

---

## 3. Energy Flow Equation

Measured relationship between components:

| Component | Mean (kW) |
|-----------|-----------|
| Building | 0.8089 |
| Grid | 0.1643 |
| Solar | 0.6445 |
| Grid + Solar | 0.8088 |
| Residual (Building - Grid - Solar) | 0.000077 |

**Finding**: `Building = Grid + Solar` holds with residual < 0.001 kW.

**Interpretation**: Building measures total consumption (GROSS). Solar correction is not applied.

---

## 4. Corrections Applied

### 4.1 Negative Values

| Category | Count (raw) | Action |
|----------|-------------|--------|
| Appliances negative values | 119,671 | Clipped to 0 |
| Aggregate negative values | 3,851 (11%) | Clipped to 0 |

### 4.2 Double Counting

**Physical setup identified**:
```
Kast garage (CT on garage circuit)
├── Smappee_laadpaal (EV Charger) - has dedicated CT
├── Laadpaal_stopcontact (Charging Socket) - has dedicated CT
└── Residual garage equipment
```

**Correction applied**: `Kast garage = Kast garage - Smappee_laadpaal - Laadpaal_stopcontact`

| Metric | Before | After |
|--------|--------|-------|
| Kast garage mean | 0.5118 kW | 0.2717 kW |
| Ghost Load | -29.9% | -0.2% |

### 4.3 Time Gaps

| Metric | Value |
|--------|-------|
| Gaps found (raw) | 3 |
| Gaps after interpolation | 0 |
| Interpolation method | Linear |

---

## 5. EDA ↔ Data Preparation Comparison

| Aspect | EDA | Data Prep |
|--------|-----|-----------|
| Energy flow equation | Building=Grid+Solar verified | Same assumption used |
| Building interpretation | GROSS consumption | Used directly as Aggregate |
| Negative appliances | Identified | Clipped to 0 |
| Double counting | Identified (Kast garage) | Subtracted EV chargers |
| Ghost Load (raw) | -23.1% | -29.9% |
| Ghost Load (final) | 0.3% | 0.3% |
| Aggregate negatives | Identified (11%) | Clipped to 0 |
| Time gaps | 3 gaps, 99.98% coverage | Interpolated to 100% |
| Solar column | Kept | Dropped (pattern captured by hour_sin/cos) |
| Grid column | Kept | Dropped (redundant: Grid = Building - Solar) |

**Note**: Ghost Load difference (-23.1% vs -29.9%) is due to different aggregation timing (before/after Building clipping).

---

## 6. Output File

### 6.1 Columns

| # | Column Name | Original (Dutch) |
|---|-------------|------------------|
| 1 | Time | _time |
| 2 | Aggregate | Building |
| 3 | RangeHood | Dampkap |
| 4 | Dryer | Droogkast |
| 5 | Stove | Fornuis |
| 6 | GarageCabinet | Kast garage |
| 7 | ChargingStation_Socket | Laadpaal_stopcontact |
| 8 | Oven | Oven |
| 9 | RainwaterPump | Regenwaterpomp |
| 10 | SmappeeCharger | Smappee_laadpaal |
| 11 | Dishwasher | Vaatwasser |
| 12 | HeatPump | Warmtepomp |
| 13 | HeatPump_Controller | Warmtepomp - Sturing |
| 14 | WashingMachine | Wasmachine |
| 15 | hour_sin | - (derived) |
| 16 | hour_cos | - (derived) |
| 17 | dow_sin | - (derived) |
| 18 | dow_cos | - (derived) |
| 19 | month_sin | - (derived) |
| 20 | month_cos | - (derived) |

### 6.2 Temporal Features

| Feature | Cycle | Range | Purpose |
|---------|-------|-------|---------|
| hour_sin | 24h | [-1, 1] | Daily pattern (sin component) |
| hour_cos | 24h | [-1, 1] | Daily pattern (cos component) |
| dow_sin | 7 days | [-0.97, 0.97] | Weekly pattern (sin component) |
| dow_cos | 7 days | [-0.90, 1] | Weekly pattern (cos component) |
| month_sin | 12 months | [-1, 1] | Seasonal pattern (sin component) |
| month_cos | 12 months | [-1, 1] | Seasonal pattern (cos component) |

**Why cyclical encoding?**
- Continuity at boundaries (23:59→00:00, Dec→Jan)
- Equal distance between consecutive values
- No artificial ordinal relationships (e.g., Sunday=6 not "bigger" than Monday=0)

### 6.3 Value Ranges

| Column | Min | Max | Mean |
|--------|-----|-----|------|
| Aggregate | 0.000 | 7.622 | 0.813 |
| RangeHood | 0.000 | 0.218 | 0.003 |
| Dryer | 0.002 | 1.400 | 0.004 |
| Stove | 0.000 | 1.761 | 0.007 |
| GarageCabinet | 0.000 | 1.932 | 0.272 |
| ChargingStation_Socket | 0.000 | 3.102 | 0.078 |
| Oven | 0.000 | 1.080 | 0.011 |
| RainwaterPump | 0.000 | 0.369 | 0.000 |
| SmappeeCharger | 0.002 | 7.350 | 0.162 |
| Dishwasher | 0.000 | 1.416 | 0.020 |
| HeatPump | 0.003 | 1.716 | 0.212 |
| HeatPump_Controller | 0.000 | 0.093 | 0.018 |
| WashingMachine | 0.000 | 1.194 | 0.024 |

**Observation**: All columns have minimum ≥ 0 (no negative values).

### 6.3 Energy Balance

| Metric | Value |
|--------|-------|
| Aggregate total | 28,486.55 kWh |
| Sum of Appliances total | 28,392.54 kWh |
| Difference | -94.01 kWh |
| Difference % | -0.33% |

---

## 7. Appliance Statistics

### 7.1 Activity Rates (% time > 10W)

| Appliance | Activity % |
|-----------|------------|
| GarageCabinet | 100.0% |
| HeatPump_Controller | 34.0% |
| HeatPump | 26.7% |
| SmappeeCharger | 7.4% |
| ChargingStation_Socket | 6.2% |
| Dishwasher | 6.2% |
| WashingMachine | 6.2% |
| Oven | 2.3% |
| Stove | 1.9% |
| RangeHood | 0.6% |
| Dryer | 0.1% |
| RainwaterPump | 0.1% |

### 7.2 Power Signatures (when ON > 10W)

| Appliance | Mean (ON) kW | Max kW |
|-----------|--------------|--------|
| SmappeeCharger | 2.163 | 7.350 |
| ChargingStation_Socket | 1.259 | 3.102 |
| HeatPump | 0.783 | 1.716 |
| Dryer | 0.628 | 1.400 |
| Oven | 0.454 | 1.080 |
| WashingMachine | 0.388 | 1.194 |
| Stove | 0.378 | 1.761 |
| Dishwasher | 0.315 | 1.416 |
| GarageCabinet | 0.272 | 1.932 |
| RainwaterPump | 0.221 | 0.369 |
| RangeHood | 0.052 | 0.218 |
| HeatPump_Controller | 0.051 | 0.093 |

---

## 8. Format Comparison with NILM Benchmarks

| Dataset | Houses | Duration | Resolution | Appliances | Format |
|---------|--------|----------|------------|------------|--------|
| REDD (2011) | 6 | 3-19 days | 1s / 15min | 9-24 | Time, Main, Apps |
| UK-DALE (2015) | 5 | 2-4 years | 6s / 1min | 5-54 | Time, Main, Apps |
| REFIT (2017) | 20 | 2 years | 8s | 9 | Time, Agg, Apps |
| Pecan Street | 1000+ | varies | 1min/15min | varies | Time, Grid, Apps |
| ENERTALK (2019) | 22 | varies | 15s | 7 | Time, Main, Apps |
| This dataset | 1 | 1 year | 15min | 12 | Time, Agg, Apps |

**Structure comparison**:

REFIT format:
```
Time, Aggregate, Appliance1, Appliance2, ...
```

This dataset:
```
Time, Aggregate, RangeHood, Dryer, Stove, GarageCabinet, ...
```

---

## 9. Resolution Context

| Resolution | NILM Approaches |
|------------|-----------------|
| 1 second | Event-based, transient features |
| 1 minute | State-based HMM, neural networks |
| 15 minutes | Energy-based, daily patterns, sequence models |
| 1 hour | Statistical methods |

Referenced approaches for 15-min resolution:
- LSTM Sequence-to-Point (Zhang et al., 2018)
- CNN-based disaggregation (Kelly & Knottenbelt, 2015)
- Temporal Pooling (Faustine et al., 2020)
- Attention-based models (Yue et al., 2020)

---

## 10. Output Files

| File | Size |
|------|------|
| `data/processed/15min/nilm_ready_dataset.parquet` | 1.44 MB |
| `data/processed/15min/nilm_ready_dataset.csv` | 8.88 MB |
| `data/processed/15min/model_ready/scaler.pkl` | ~2 KB |
| `data/processed/15min/model_ready/heatpump/X_train.npy` | 63.69 MB |
| `data/processed/15min/model_ready/heatpump/y_train.npy` | 9.10 MB |

---

## 11. Pretraining Pipeline

### 11.1 Split Strategy (Block Time-Series Interleaved)

| Parameter | Value |
|-----------|-------|
| Block size | 7 days (672 samples) |
| Pattern | [Train, Train, Train, Train, Val, Test] |
| Train | 24,192 rows (69.0%) |
| Validation | 5,472 rows (15.6%) |
| Test | 5,376 rows (15.3%) |

**Seasonal Coverage**: Each set contains ~25% from each season (Winter, Spring, Summer, Autumn).

### 11.2 Windowing Configuration

| Parameter | Value |
|-----------|-------|
| Window size | 96 samples (24 hours) |
| Stride | 1 |
| Architecture | Seq2Seq |
| Input shape | (n_samples, 96, 7) |
| Output shape | (n_samples, 96, 1) |

### 11.3 Input Features

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | Aggregate | Total power consumption (kW) |
| 1 | hour_sin | Daily cycle (sin) |
| 2 | hour_cos | Daily cycle (cos) |
| 3 | dow_sin | Weekly cycle (sin) |
| 4 | dow_cos | Weekly cycle (cos) |
| 5 | month_sin | Yearly cycle (sin) |
| 6 | month_cos | Yearly cycle (cos) |

### 11.4 Scaling

- Method: MinMax scaling [0, 1]
- Fitted on: Train set only (no data leakage)
- Features scaled: 19 (Aggregate + 12 appliances + 6 temporal)

---

## 12. Summary

| Item | Value |
|------|-------|
| Energy flow equation | Building = Grid + Solar (residual: 0.000077 kW) |
| Phase handling | Smappee pre-sums 3-phase; Building filtered to Smappee/Consumption |
| Negative values | Clipped to 0 |
| Double counting correction | EV chargers subtracted from Kast garage |
| Time gaps | Interpolated (3 → 0) |
| Energy balance | -0.33% |
| Temporal features | 6 cyclical features (hour, dow, month sin/cos) |
| Format | Time, Aggregate, 12 Appliances, 6 Temporal Features |
| Total columns | 20 |
| Resolution | 15 minutes |
| Duration | 1 year (35,040 rows) |
