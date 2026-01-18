# Energy System Simulation Module

Shared module (`src/shared/energy_system.py`) providing intra-day energy system simulation
with battery constraints. Used by Phase 3 extended decomposition and Phase 4 optimization.

## Battery Model

```python
BATTERY_PARAMS = {
    'capacity_kwh': 13.8,        # Total capacity (usable: 13.8 x 0.8 = 11.04 kWh)
    'max_charge_kw': 5.0,        # Max charging rate
    'max_discharge_kw': 5.0,     # Max discharging rate
    'efficiency': 0.77,          # Round-trip efficiency (post-degradation, was 0.84)
    'initial_soc_pct': 50.0,     # Default starting SoC
    'min_soc_pct': 20.0,         # Minimum SoC (battery protection since Mar 2025)
    'max_soc_pct': 100.0,        # Maximum SoC
    'discharge_start_hour': 15.0,   # Preferred discharge window start
    'discharge_end_hour': 22.0,     # Preferred discharge window end
    'allow_overnight_discharge': False,  # Block 00:00-06:00 discharge
}
```

## Battery Model Improvements (Jan 2026)

Analysis comparing model vs observed battery behavior revealed:
- **Total capacity**: 13.8 kWh with 20% min SoC -> usable capacity 11.04 kWh
- **Post-degradation efficiency**: 77% round-trip (down from 84% after Feb-Mar 2025 event)
- **Time-of-use strategy**: Discharge concentrated 15:00-22:00, minimal overnight
- **Result**: Model grid import now within +0.8% of observed (was -2.1%)

## COP Model

```
COP = 5.93 + 0.13*T_outdoor - 0.08*T_HK2
```
Clipped to [1.5, 8.0] for physical limits.

## Heating Curve Model

```
T_HK2 = setpoint + curve_rise * (T_ref - T_outdoor)
```
Where T_ref = 21.32C (comfort) or 19.18C (eco).

## Key Functions

| Function | Description |
|----------|-------------|
| `simulate_battery_soc()` | SoC tracking with capacity constraints |
| `predict_cop()` | Intra-day COP from T_outdoor and T_HK2 |
| `predict_t_hk2_variable_setpoint()` | Heating curve with comfort/eco modes |
| `is_high_tariff()` | Tariff period detection |
| `calculate_electricity_cost()` | Tariff-aware cost calculation |
| `simulate_energy_system()` | Full system simulation |

## Energy Flow Logic

1. Net = PV - consumption
2. If Net > 0: charge battery (up to capacity), excess to grid
3. If Net < 0: discharge battery (if available), deficit from grid

## Used By

- `src/phase3/06_extended_decomposition.py` - Panels 5-8, 10
- `src/phase4/04_pareto_optimization.py` - Strategy evaluation
- `src/phase4/02_strategy_simulation.py` - Strategy comparison

---

## Grey-Box Thermal Model (Abandoned)

Physics-based two-state discrete-time model for room temperature prediction.
**Status:** Tried but did not work well - poor predictive accuracy on validation data.

**Model Formulation (dt = 15 min):**
```
T_buffer[k+1] = T_buffer[k] + (dt/tau_buf) * [(T_HK2[k] - T_buffer[k]) - r_emit * (T_buffer[k] - T_room[k])]
T_room[k+1] = T_room[k] + (dt/tau_room) * [r_heat * (T_buffer[k] - T_room[k]) - (T_room[k] - T_out[k])] + k_solar * PV[k]
```

Parameters: `tau_buf` (buffer time constant), `tau_room` (building time constant),
`r_emit`/`r_heat` (coupling ratios), `k_solar` (solar gain), `c_offset` (bias).

Script: `src/phase3/01b_greybox_thermal_model.py`

---

## Transfer Function Thermal Model

Linear transfer function model using low-pass filtered inputs:
```
T_room = offset + g_outdoor*LPF(T_outdoor, 24h) + g_effort*LPF(Effort, 2h) + g_pv*LPF(PV, 12h)
```

Where `Effort = T_HK2 - baseline_curve` (deviation from heating curve).

### Key Parameters

- `g_effort = 0.208` (STABLE: coefficient of variation = 9%)
- `g_outdoor = 0.442` (UNSTABLE: CV = 95%)
- Model R2 = 0.68 (captures 68% of variance)

### Causal Coefficients (for Phase 4 optimization)

The transfer function provides causal estimates of how heating parameters affect room temperature:

| Parameter | Phase 2 Regression | Phase 3 Causal | Ratio |
|-----------|-------------------|----------------|-------|
| comfort_setpoint | +1.22C/C | **+0.21C/C** | 5.9x |
| curve_rise | +9.73C/unit | **+2.92C/unit** | 3.3x |

**Important:** Phase 2 regression coefficients overestimate effects by 3-6x because they
capture associations, not causal effects. Phase 4 now uses the causal coefficients from
`output/phase3/causal_coefficients.json`.

### Causal Chain

```
setpoint +1C -> T_HK2 +1C -> Effort +1C -> LPF(Effort) +1C -> T_room +0.21C
```

### Scripts

- `src/phase3/01_thermal_model.py` - Main thermal model
- `src/phase3/01e_adaptive_thermal_model.py` - Time-varying parameters (RLS)
- `src/phase3/01f_transfer_function_integration.py` - Causal coefficient derivation
