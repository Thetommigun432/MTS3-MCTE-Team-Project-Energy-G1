# Integration Failure Report

## Summary

The frontend-backend integration is broken due to schema mismatches between the prediction persister and backend API queries.

---

## 1. Measurement Name Mismatch

| Component | Measurement Name |
|-----------|-----------------|
| `persister.py` (writes) | `nilm_predictions` |
| Backend queries.py | `prediction` |
| Backend client.py | `prediction` (writes) |

**Impact**: Persister data is invisible to backend queries.

---

## 2. Tag/Field Schema Mismatch

### Persister Writes (narrow format)
```
nilm_predictions
├── Tags: building_id, appliance, model_version
└── Fields: power_watts, probability, confidence, is_on
```

### Backend Expects (wide format)
```
prediction
├── Tags: building_id, model_version (NO appliance_id tag)
└── Fields: predicted_kw_{appliance}, confidence_{appliance}
```

**Impact**: `get_unique_appliances()` queries `appliance_id` tag which doesn't exist in wide format.

---

## 3. Building ID Mismatch

| Component | Building ID Example |
|-----------|---------------------|
| Persister | `building_1` (stream key) |
| Backend API | UUID from Supabase |
| Frontend | UUID from Supabase buildings table |

**Impact**: Queries with UUID return empty data because Influx has `building_1`.

---

## 4. Appliances Discovery Broken

Backend `get_unique_appliances()` uses:
```flux
from(bucket: "predictions")
  |> filter(fn: (r) => r.building_id == "...")
  |> keep(columns: ["appliance_id"])  // FAILS: column doesn't exist
  |> distinct(column: "appliance_id")
```

**Fix needed**: Parse `predicted_kw_*` field keys instead of relying on tags.

---

## Resolution Plan

1. **Standardize measurement name**: Use `prediction` everywhere
2. **Adopt wide format**: One point per timestamp with all appliances as fields
3. **Bridge building IDs**: Add `stream_key` column to buildings table
4. **Fix appliance discovery**: Use `schema.measurementFieldKeys()` to list fields and parse `predicted_kw_*` prefixes
5. **Update persister**: Write canonical wide format with consistent tags
