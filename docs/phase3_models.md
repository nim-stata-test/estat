# Phase 3: System Models Documentation

This document provides detailed documentation of the models developed in Phase 3, including their theoretical basis, assumptions, limitations, and interpretation guidelines.

---

## Table of Contents

1. [Thermal Model (3.1)](#1-thermal-model) — Transfer function model with low-pass filtered inputs
   - 1a. [Adaptive Thermal Model](#1a-adaptive-thermal-model) — Time-varying parameters via RLS
   - 1b. [Transfer Function Integration](#1b-transfer-function-integration) — Causal coefficients for optimization
2. [Heat Pump Model (3.2)](#2-heat-pump-model)
   - 2b. [Thermodynamic COP Model](#2b-thermodynamic-cop-model) — Physics-based COP from pressures
3. [Energy System Model (3.3)](#3-energy-system-model)
4. [Model Integration for Optimization](#4-model-integration)
5. [Tariff Cost Model (3.4)](#5-tariff-cost-model)
6. [Known Limitations](#6-known-limitations)

---

## 1. Thermal Model

### 1.1 Theoretical Background

The building is modeled as a first-order RC (resistance-capacitance) thermal network, analogous to an electrical circuit where:

- **Thermal resistance (R)** = 1/UA, where UA is the heat loss coefficient (W/K)
- **Thermal capacitance (C)** = thermal mass of the building (J/K)
- **Time constant (τ)** = R × C = C/UA (seconds or hours)

### 1.2 Key Insight: Continuous Heating

**Important**: The heat pump operates continuously, not just during daytime. Analysis shows:
- Heating is ON 38-77% of the time, including nights (44% at midnight)
- Heating circuit temperature (T_hk2) varies from ~30°C at night to ~36°C in morning
- This is NOT "heating off at night" - the system runs 24/7

Therefore, we use **T_hk2 (heating circuit 2 temperature)** as a proxy for heating effort.

### 1.3 Model Formulation

The governing equation with continuous heating:

```
dT_room/dt = a × (T_hk2 - T_room) - b × (T_room - T_outdoor) + c × PV
```

Where:
- `T_room` = indoor temperature (°C)
- `T_outdoor` = outdoor temperature (°C)
- `T_hk2` = heating circuit 2 temperature (proxy for heating effort, ~30-36°C)
- `PV` = solar gain proxy (from PV generation)
- `a` = heating coefficient (response to heating circuit)
- `b` = loss coefficient (heat loss to outdoor)
- `c` = solar gain coefficient
- `τ = 1/b` = thermal time constant

### 1.4 Discrete-Time Formulation

For 15-minute time steps:

```
ΔT = a × (T_hk2 - T_room) - b × (T_room - T_outdoor) + c × PV
```

Where ΔT = T[k+1] - T[k] is the temperature change over one time step.

### 1.5 Assumptions

| Assumption | Justification | Impact if Violated |
|------------|---------------|-------------------|
| T_hk2 as heating proxy | HK2 temp varies with heating demand | Good proxy, but not exact power measurement |
| Single thermal zone | Simplifies to one representative temperature | Under-predicts variation between rooms |
| First-order dynamics | Building has dominant single time constant | Misses fast/slow thermal responses |
| Linear relationships | Simplifies regression | May miss nonlinearities at extremes |
| PV as solar proxy | PV generation correlates with irradiance | Imperfect due to panel orientation/shading |
| Continuous heating | Heat pump runs 24/7 | Validated by data (38-77% duty cycle) |

### 1.6 Parameter Estimation Method

1. Extract T_room, T_out, T_hk2, and PV at 15-min intervals
2. Compute temperature change: `ΔT = T_room[k+1] - T_room[k]`
3. Compute driving factors: `(T_hk2 - T_room)`, `(T_room - T_outdoor)`, `PV`
4. Linear regression: `ΔT = a × (T_hk2 - T_room) - b × (T_room - T_out) + c × PV`
5. Time constant: `τ = dt / |b|` where dt = 0.25 hours

### 1.7 Results Interpretation

| Parameter | Value | Physical Meaning |
|-----------|-------|------------------|
| **Time constant (office1)** | ~17.5 hours | Building takes ~17h to lose 63% of temperature difference. Moderate thermal mass. |
| **Average time constant** | ~30.6 hours | Varies by room (16-56 hours depending on insulation, exposure) |
| **Heating coefficient** | ~0.013 K/(15min)/K | For every 1K difference between HK2 and room temp, room warms by 0.013K per 15 minutes. |
| **Loss coefficient** | ~-0.014 K/(15min)/K | For every 1K difference between room and outdoor, room cools by 0.014K per 15 minutes. |
| **R² = 0.10-0.24** | Low-moderate | Temperature is tightly controlled, limiting variation to explain. |

**Practical Implications:**

- **Pre-heating time**: With τ ≈ 17-30h, practical pre-heating is 1-2 hours before comfort period.
- **Setback recovery**: After night setback (eco mode), recovery to comfort takes 30-90 minutes depending on outdoor temperature.
- **Response to HK2 changes**: When HK2 increases by 5°C (comfort mode starts), room warms by ~0.065°C per 15 min = ~0.26°C/hour initially.

### 1.6 Model Limitations

1. **Low explanatory power (R²)**: The heating system maintains stable temperatures, so there's little variation to model. The model captures the physics but can't predict much variation.

2. **Simplified zone model**: Different rooms have different thermal characteristics. The model averages across zones.

3. **No wind/infiltration effects**: Wind increases heat loss through infiltration and convection. Not captured.

4. **Simulation instability**: Forward simulation can diverge due to small coefficient errors accumulating. Requires periodic resetting.

---

## 1a. Adaptive Thermal Model

> **Script**: `src/phase3/01e_adaptive_thermal_model.py`
> **Output**: `output/phase3/fig3.07_adaptive_thermal_model.png`

### 1a.1 Purpose

The fixed-parameter transfer function model achieves R² = 0.68, which is limited by:
- Time-varying building dynamics (weather, occupancy, wind)
- Seasonal changes in thermal characteristics
- Non-stationary heating patterns

The adaptive model addresses this by allowing parameters to vary over time using Recursive Least Squares (RLS).

### 1a.2 Model Formulation

Same transfer function structure with low-pass filtered inputs:

```
T_room = offset + g_outdoor×LPF(T_outdoor, 24h) + g_effort×LPF(Effort, 2h) + g_pv×LPF(PV, 12h)
```

Where `Effort = T_HK2 - baseline_curve` (deviation from heating curve).

Parameters `g_outdoor`, `g_effort`, `g_pv`, and `offset` are updated at each timestep via RLS.

### 1a.3 Recursive Least Squares (RLS)

The RLS algorithm updates parameter estimates as new data arrives:

```
K[k] = P[k-1]×x[k] / (λ + x[k]'×P[k-1]×x[k])  # Kalman gain
θ[k] = θ[k-1] + K[k]×(y[k] - x[k]'×θ[k-1])    # Parameter update
P[k] = (1/λ)×(P[k-1] - K[k]×x[k]'×P[k-1])     # Covariance update
```

Where λ = 0.995 (forgetting factor) balances adaptation speed vs stability.

### 1a.4 Key Results

**Performance improvement:**
- Fixed model: R² = 0.68
- Adaptive model: R² = 0.86+

**Parameter stability analysis:**
| Parameter | Mean | Std Dev | CV | Interpretation |
|-----------|------|---------|-----|----------------|
| g_effort | 0.208 | 0.019 | 9% | STABLE - reliable for optimization |
| g_outdoor | 0.442 | 0.420 | 95% | UNSTABLE - confounded with unmeasured factors |
| g_pv | 0.012 | 0.008 | 67% | Moderate variability |

**Key insight**: The heating effort coefficient `g_effort` is stable (CV=9%), meaning we can reliably predict how changes in heating parameters affect room temperature. The outdoor coefficient is highly variable, likely because it captures unmeasured factors (wind, infiltration, occupancy).

### 1a.5 Implications for Optimization

The stable `g_effort = 0.208` coefficient enables causal reasoning:
- +1°C increase in heating effort → +0.21°C room temperature (after low-pass filter lag)
- This chain: `setpoint → T_HK2 → Effort → T_room` is now quantified

---

## 1b. Transfer Function Integration

> **Script**: `src/phase3/01f_transfer_function_integration.py`
> **Output**: `output/phase3/fig3.08_transfer_function_integration.png`, `causal_coefficients.json`

### 1b.1 Purpose

Phase 2 regression analysis found strong associations between heating parameters and room temperature:
- Comfort setpoint: +1.22°C per 1°C increase
- Curve rise: +9.73°C per unit increase

However, these are **observational associations**, not causal effects. The transfer function integration derives **causal coefficients** that Phase 4 optimization can use.

### 1b.2 The Confounding Problem

Phase 2 regression captured correlations, but:
- Warmer outdoor temps correlate with both higher setpoints and higher room temps
- Users may adjust setpoints in response to comfort (reverse causality)
- Seasonal patterns affect both heating parameters and room temperature

### 1b.3 Causal Chain Derivation

The transfer function provides a causal pathway:

```
Parameters → T_HK2 (heating curve) → Effort → T_room (via g_effort)
```

**Step 1: Heating Curve Model**
```
T_HK2 = setpoint + curve_rise × (T_ref - T_outdoor)
```

**Step 2: Effort Calculation**
```
Effort = T_HK2 - baseline_curve
       = setpoint + curve_rise×(T_ref - T_outdoor) - [baseline_setpoint + baseline_rise×(T_ref - T_outdoor)]
       = Δsetpoint + Δcurve_rise × (T_ref - T_outdoor)
```

**Step 3: Room Temperature Response**
```
ΔT_room = g_effort × LPF(ΔEffort)
```

### 1b.4 Causal Coefficients

| Parameter | Phase 2 Regression | Phase 3 Causal | Ratio |
|-----------|-------------------|----------------|-------|
| comfort_setpoint | +1.22°C/°C | **+0.21°C/°C** | 5.9× |
| curve_rise | +9.73°C/unit | **+2.92°C/unit** | 3.3× |

**Key finding**: Phase 2 regression overestimates effects by 3-6× because it captures associations, not causal effects.

### 1b.5 Output File

The causal coefficients are saved to `output/phase3/causal_coefficients.json`:

```json
{
  "setpoint_effect": 0.208,
  "curve_rise_effect_per_degree": 0.208,
  "curve_rise_effect_at_ref": 2.92,
  "g_effort": 0.208,
  "source": "transfer_function_integration"
}
```

### 1b.6 Usage in Phase 4

Phase 4 optimization uses causal coefficients for realistic strategy evaluation:
- Temperature adjustments based on `g_effort × Δsetpoint` instead of regression coefficients
- More conservative (and realistic) predictions of comfort impacts
- Loaded from `causal_coefficients.json` by optimization scripts

---

## 2. Heat Pump Model

### 2.1 Theoretical Background

Heat pump efficiency is characterized by the Coefficient of Performance (COP):

```
COP = Q_heat / W_elec = Heat delivered / Electricity consumed
```

For an ideal (Carnot) heat pump:
```
COP_Carnot = T_hot / (T_hot - T_cold)
```

Real heat pumps achieve 40-60% of Carnot efficiency. COP depends on:
- **Source temperature** (outdoor air for ASHP)
- **Sink temperature** (flow/delivery temperature)
- **Part-load operation** (modulation)

### 2.2 Empirical COP Model

Based on daily data, a linear regression model was fitted:

```
COP = β₀ + β₁ × T_outdoor + β₂ × T_flow
```

**Fitted parameters:**
- β₀ = 6.52 (base COP)
- β₁ = +0.1319 (COP increase per °C outdoor)
- β₂ = -0.1007 (COP decrease per °C flow)

**Model quality:** R² = 0.95 (excellent fit)

### 2.3 Assumptions

| Assumption | Justification | Impact if Violated |
|------------|---------------|-------------------|
| Linear COP relationship | Empirically validated, common in literature | May miss nonlinearities at extremes |
| Daily averaging | Smooths out cycling effects | Loses detail on part-load efficiency |
| Steady-state operation | Ignores startup/defrost cycles | Underestimates losses during cycling |
| Constant auxiliary power | Pumps, controls assumed constant | Minor error in electricity consumption |
| No degradation over time | Short study period | Long-term: COP may decline |

### 2.4 Results Interpretation

| Parameter | Value | Physical Meaning |
|-----------|-------|------------------|
| **Mean COP** | 4.01 | For every 1 kWh electricity, 4.01 kWh heat delivered. Excellent efficiency. |
| **COP range** | 2.53 - 5.80 | Varies significantly with conditions. Low COP on cold days with high flow temps. |
| **Outdoor sensitivity** | +0.13/°C | Each +1°C outdoor temp improves COP by 0.13. Warmer is better. |
| **Flow sensitivity** | -0.10/°C | Each +1°C flow temp reduces COP by 0.10. Lower delivery temps are better. |

**Practical Implications:**

1. **Run during warmest hours**: If outdoor temp is 5°C higher at midday vs midnight, COP improves by 0.65 (e.g., from 3.5 to 4.15 = 19% more efficient).

2. **Lower flow temperatures**: Reducing flow temp by 5°C improves COP by 0.5. This can be achieved by:
   - Lowering curve rise parameter
   - Accepting slightly lower room temps
   - Using larger radiators (not practical short-term)

3. **Combined effect**: Running at midday (+5°C outdoor) with reduced flow (-3°C) could improve COP by ~1.0 (from 3.5 to 4.5 = 29% improvement).

### 2.5 COP Prediction Examples

Using `COP = 6.52 + 0.1319×T_outdoor - 0.1007×T_flow`:

| Scenario | T_outdoor | T_flow | Predicted COP |
|----------|-----------|--------|---------------|
| Cold night, high flow | -5°C | 45°C | 6.52 - 0.66 - 4.53 = **1.33** |
| Cold day, normal flow | 0°C | 35°C | 6.52 + 0 - 3.52 = **3.00** |
| Mild day, normal flow | 8°C | 35°C | 6.52 + 1.06 - 3.52 = **4.06** |
| Mild day, low flow | 8°C | 30°C | 6.52 + 1.06 - 3.02 = **4.56** |
| Warm day, low flow | 12°C | 28°C | 6.52 + 1.58 - 2.82 = **5.28** |

### 2.6 Buffer Tank Dynamics

The buffer tank serves as thermal storage between the heat pump and heating circuits.

**Observed characteristics:**
- Mean temperature: 35.8°C
- Range: 22.3°C - 55.5°C
- Max charging rate: ~30°C/hour
- Max discharging rate: ~35°C/hour

**Correlation with outdoor temp:** -0.37 (moderately negative)
- Buffer runs hotter when outdoor is colder (more heating demand)

**Implications for optimization:**
- Buffer can store ~2-4 hours of heating energy
- Can be pre-charged during solar hours for evening use
- Higher buffer temps reduce COP (acts like higher flow temp)

### 2.7 Model Limitations

1. **Daily resolution**: Misses intra-day variations and cycling effects.

2. **No defrost modeling**: In cold, humid conditions, ASHP must defrost periodically, reducing effective COP. Not explicitly modeled.

3. **Part-load effects**: COP varies with modulation level. Running at 50% capacity may have different efficiency than 100%.

4. **Hot water production**: Model focuses on space heating. Hot water production (different temps) not separately modeled.

---

## 2b. Thermodynamic COP Model

> **Script**: `src/phase3/02b_pressure_cop_model.py`
> **Output**: `output/phase3/fig3.06_pressure_cop_model.png`, `pressure_cop_daily.csv`

### 2b.1 Purpose

The empirical COP model (Section 2) uses outdoor and flow temperatures as inputs. The thermodynamic model provides a physics-based alternative using refrigerant pressure sensors, which gives:
- Theoretical upper bound on COP (Carnot efficiency)
- Validation of the empirical model
- Insight into compressor and system losses

### 2b.2 Theoretical Background

For an ideal (Carnot) heat pump:

$$COP_{Carnot} = \frac{T_{cond}}{T_{cond} - T_{evap}}$$

Where temperatures are in Kelvin. Real heat pumps achieve:

$$COP_{actual} = \eta \times COP_{Carnot}$$

Where η (eta) is the system efficiency factor (typically 0.4-0.6).

### 2b.3 Pressure-Temperature Relationship

The heat pump uses R410A refrigerant. Saturation temperatures are derived from measured pressures using refrigerant property tables:

| Pressure (bar) | T_sat R410A (°C) |
|----------------|------------------|
| 5 | -15.5 |
| 10 | 2.6 |
| 15 | 15.0 |
| 20 | 25.0 |
| 25 | 33.5 |
| 30 | 40.9 |

### 2b.4 Sensors Used

| Sensor | Description | Typical Range |
|--------|-------------|---------------|
| `wp_hochdruck` | High pressure (condenser) | 20-30 bar |
| `wp_niederdruck` | Low pressure (evaporator) | 5-12 bar |

### 2b.5 Results

**Carnot COP analysis:**
- Mean Carnot COP: 6.2
- Mean actual COP: 4.0
- Mean efficiency factor: η = 0.65

**Validation:**
- The thermodynamic model confirms the empirical model's sensitivity to temperature lift (T_cond - T_evap)
- Lower outdoor temps → lower evaporator pressure → larger temperature lift → lower COP

### 2b.6 Implications

The thermodynamic model validates that:
1. The empirical COP model captures the correct physical relationships
2. System efficiency is ~65% of Carnot ideal (reasonable for modern ASHPs)
3. Reducing flow temperature improves COP because it reduces condenser pressure

---

## 3. Energy System Model

### 3.1 Components Modeled

1. **PV Generation**: Solar panels producing electricity
2. **Battery Storage**: Stores excess PV for later use
3. **Grid Interaction**: Import when needed, export excess
4. **Self-consumption**: Direct use of PV without storage

### 3.2 PV Generation Model

**Empirical patterns from 3 years of data:**

| Month | Mean Daily Generation (kWh) |
|-------|----------------------------|
| January | 15.6 |
| February | 26.4 |
| March | 51.2 |
| April | 70.0 |
| May | 85.7 |
| June | 101.6 |
| July | 90.4 |
| August | 82.3 |
| September | 58.0 |
| October | 34.7 |
| November | 18.4 |
| December | 13.5 |

**Peak hours**: 10:00 - 16:00 (>50% of max generation)

**Yearly trend**: -973 kWh/year decline (R² = 0.99)
- Possible panel degradation (~1-2% per year typical)
- Or weather variation between years

### 3.3 Battery Model

**Round-trip efficiency**: 83.7%
- For every 1 kWh charged, 0.837 kWh can be discharged
- Below typical ~90% for new lithium batteries
- Consistent with degradation from Feb-Mar 2025 deep-discharge event

**Charging pattern**: Primarily 7:00 - 13:00 (during PV peak)
**Discharging pattern**: 16:00 - 6:00 (evening through night)

**Daily throughput:**
- Mean charge: 7.5 kWh
- Mean discharge: 6.3 kWh
- Difference (~1.2 kWh) = efficiency losses

### 3.4 Self-Sufficiency Calculation

```
Self-sufficiency = 1 - (Grid Import / Total Consumption)
                 = (Direct PV + Battery Discharge) / Total Consumption
```

**Current performance:**
- Self-sufficiency: 58.1%
- Grid import: 10.2 kWh/day
- Grid export: 41.0 kWh/day
- Net exporter: +30.8 kWh/day average

### 3.5 Optimization Scenarios

Three scenarios were modeled to estimate improvement potential:

**Scenario 1: Load Shifting (20%)**
- Assumption: 20% of evening/night consumption shifted to solar hours
- Method: 70% of shifted load avoids grid (rest still needs battery/grid)
- Result: Self-sufficiency improves to 63.6% (+5.5 percentage points)

**Scenario 2: Double Battery Capacity**
- Assumption: 2× current battery storage
- Method: Additional discharge capacity × efficiency added to self-consumption
- Result: Self-sufficiency improves to 79.9% (+21.8 percentage points)

**Scenario 3: Combined**
- Both load shifting and larger battery
- Result: Self-sufficiency improves to 85.3% (+27.2 percentage points)

### 3.6 Assumptions

| Assumption | Justification | Impact if Violated |
|------------|---------------|-------------------|
| Linear load shifting | Simple approximation | May over/underestimate flexibility |
| 70% grid avoidance | Conservative estimate | Actual depends on timing |
| Battery efficiency constant | Short-term approximation | Efficiency varies with SOC, rate |
| No demand response | Loads assumed inflexible beyond 20% | More flexibility = more gains |
| Grid always available | No outage modeling | Doesn't affect self-sufficiency calc |

### 3.7 Results Interpretation

**Grid Interaction:**
- System is a **net exporter** (41 kWh export vs 10 kWh import daily)
- Export mostly midday, import mostly morning/evening
- Financial optimization would consider tariff differentials

**Self-sufficiency drivers:**
1. PV generation timing vs consumption timing (mismatch)
2. Battery capacity (can bridge ~6-8 hours currently)
3. Battery efficiency (83.7% losses reduce stored energy value)

**Optimization priority:**
- Load shifting offers modest gains (5.5pp) but is low-cost
- Battery expansion offers large gains (21.8pp) but requires investment
- Combined approach achieves diminishing returns (not additive)

---

## 4. Model Integration for Optimization

> **Note**: Phase 4 optimization uses the **Transfer Function Thermal Model** with causal coefficients
> from Section 1b. The heating curve model determines T_HK2, which feeds into the thermal and COP models.
> See `src/shared/energy_system.py` for the shared implementation.

### 4.1 How Models Connect

```
                    ┌─────────────────┐
                    │  Weather/Solar  │
                    │   (T_outdoor,   │
                    │    PV forecast) │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
      ┌───────────┐  ┌───────────┐  ┌───────────┐
      │  Transfer │  │ Heat Pump │  │  Energy   │
      │  Function │  │   Model   │  │  System   │
      │   Model   │  │           │  │   Model   │
      │ T_room(t) │  │  COP(t)   │  │  PV(t)    │
      │           │  │           │  │ Battery(t)│
      └─────┬─────┘  └─────┬─────┘  └─────┬─────┘
            │              │              │
            └──────────────┼──────────────┘
                           │
                           ▼
                   ┌───────────────┐
                   │  Optimization │
                   │   Decision:   │
                   │ - When to heat│
                   │ - What setpoint│
                   │ - Curve rise  │
                   └───────────────┘
```

### 4.2 Decision Variables

| Variable | Range | Model Used |
|----------|-------|------------|
| Comfort schedule start | 06:00 - 12:00 | Thermal (pre-heat time needed) |
| Comfort schedule end | 18:00 - 22:00 | Thermal (coast time available) |
| Room setpoint (comfort) | 19 - 22°C | Heat Pump (affects flow temp) |
| Curve rise | 0.9 - 1.2 | Heat Pump (affects flow temp → COP) |
| Buffer target | 30 - 50°C | Heat Pump (thermal storage vs COP) |

### 4.3 Optimization Objective

**Primary**: Maintain comfortable temperature in key rooms
**Secondary**: Minimize energy expenditure

```
Maximize: Comfort_Compliance (% time T_room in comfort band)
Subject to:
  - T_room ∈ [18.5°C, 23°C] for comfort hours (primary constraint)
  - T_room ∈ [16°C, 25°C] for eco hours
  - Equipment capacity limits
  - Solar priority (prefer PV over grid)

Secondary objective (minimize when comfort is satisfied):
  Grid_Import × Cost_import - Grid_Export × Value_export
```

### 4.4 Example Optimization Logic

**Rule-based heuristics derived from models:**

1. **Morning pre-heat timing**:
   - Check weather forecast for outdoor temp at sunrise
   - If T_outdoor < 5°C: Start comfort at 08:00 (before PV)
   - If T_outdoor ≥ 5°C: Delay comfort to 10:00 (wait for PV)
   - Rationale: COP improves +0.65 by waiting for +5°C warmer temps

2. **Flow temperature reduction**:
   - If PV generation > consumption: Reduce curve rise by 0.05
   - Rationale: -0.05 curve rise ≈ -0.8°C flow temp ≈ +0.08 COP

3. **Buffer pre-charging**:
   - At 12:00-14:00 (peak PV): Increase buffer target to 45°C
   - At 16:00 (PV declining): Return to normal operation
   - Rationale: Store thermal energy for evening heating

4. **Evening setback timing**:
   - With 54h time constant, room cools slowly
   - Can start eco mode 1-2 hours before bedtime
   - Room temp drops only ~0.5-1°C before sleep

---

## 5. Tariff Cost Model

### 5.1 Overview

The tariff cost model analyzes electricity costs using time-of-use tariff data from Primeo Energie and builds a forecasting model for cost prediction.

### 5.2 Data Sources

| Source | Data | Time Range |
|--------|------|------------|
| Primeo Energie | Purchase and feed-in tariffs | 2023-2025 |
| ElCom LINDAS | Official Swiss tariff database | 2023-2025 |

### 5.3 Tariff Structure

**Time-of-Use Windows:**
- **High tariff (Hochtarif)**: Mon-Fri 06:00-21:00, Sat 06:00-12:00
- **Low tariff (Niedertarif)**: Mon-Fri 21:00-06:00, Sat 12:00 - Mon 06:00, Federal holidays

**Current Rates (2025):**
| Tariff Type | High (Rp/kWh) | Low (Rp/kWh) | Spread |
|-------------|---------------|--------------|--------|
| Purchase | 35.9 | 27.7 | 8.2 |
| Feed-in (with HKN) | 13.0-15.5 | 13.0-15.5 | — |

### 5.4 Cost Calculation

**Daily net cost:**
```
Net_Cost = Grid_Import × Purchase_Rate - PV_Export × Feedin_Rate
```

**Where rates are time-dependent:**
```
Purchase_Rate(t) = High_Rate if is_high_tariff(t) else Low_Rate
```

### 5.5 Key Findings

**Historical Analysis:**
- Household is typically a **net producer** (revenue exceeds costs)
- Annual net income: ~CHF 1,048 (average over analysis period)
- High-tariff hours account for ~60% of grid purchase costs

**Cost Optimization Potential:**
- Shifting heating from high-tariff morning (06:00-10:00) to solar hours (10:00-16:00) can reduce costs by ~2-4%
- Low-tariff evening heating (21:00+) provides additional savings potential
- Battery should discharge during high-tariff peak demand (17:00-21:00)

### 5.6 Forecasting Model

A regression model predicts daily costs from weather and seasonal factors:

```
Daily_Cost = β₀ + β₁×HDD + β₂×Month + β₃×Weekday
```

Where:
- HDD = Heating Degree Days (max(18 - T_outdoor, 0))
- Month = categorical for seasonality
- Weekday = binary indicator for weekday vs weekend

### 5.7 Optimization Implications

The tariff cost model enables the **Cost-Optimized** strategy in Phase 4:

| Strategy Element | Implementation | Expected Impact |
|-----------------|----------------|-----------------|
| Avoid morning high-tariff | Comfort start at 11:00 vs 06:30 | -10% morning grid cost |
| Maximize solar self-consumption | Shift heating to 10:00-16:00 | Direct use > export |
| Aggressive grid fallback | Curve rise 0.85 when grid-dependent | Lower energy per heating event |
| Pre-heat before high-tariff | Buffer charging at 21:00 | Avoid morning grid peaks |

---

## 6. Known Limitations

### 6.1 Data Limitations

| Issue | Impact | Mitigation |
|-------|--------|------------|
| 64-day sensor overlap | Limited seasonal coverage | Use full 3-year energy data for seasonal patterns |
| 15-min resolution | Misses fast dynamics | Acceptable for building thermal modeling |
| Sparse room sensor data | Some rooms have gaps | Focus on rooms with best coverage |
| No solar irradiance sensor | Use PV as proxy | Imperfect correlation |

### 6.2 Model Limitations

| Model | Limitation | Impact |
|-------|------------|--------|
| Thermal | First-order only | Misses fast/slow dynamics |
| Thermal | Single zone | Ignores room-to-room variation |
| Heat Pump | Daily resolution | Misses cycling effects |
| Heat Pump | No defrost modeling | Underestimates cold-weather losses |
| Energy | Static scenarios | Real optimization is dynamic |
| Tariff | No dynamic simulation | Uses historical data, not strategy-specific costs |

### 6.3 Validation Status

| Model | Validation Method | Status |
|-------|-------------------|--------|
| Thermal (fixed) | R² of transfer function | 0.68 (good) |
| Thermal (adaptive) | R² with RLS parameters | 0.86+ (excellent) |
| Heat Pump | R² of COP regression | 0.95 (excellent) |
| Energy | Compare calculated vs actual self-sufficiency | Within 5% (good) |

### 6.4 Recommended Improvements

1. **Higher-resolution COP model**: Use 15-min or hourly data for better part-load characterization.

2. **Multi-zone thermal model**: Model key rooms separately for better comfort prediction.

3. **Weather forecast integration**: Use forecasts for predictive optimization.

4. **Dynamic cost simulation**: Simulate energy consumption changes under different strategies to predict actual cost savings.

5. **Longer sensor coverage**: Continue data collection through full heating season.

---

## Appendix: Key Equations Summary

### Transfer Function Thermal Model

**Model formulation:**
```
T_room = offset + g_outdoor×LPF(T_outdoor, 24h) + g_effort×LPF(Effort, 2h) + g_pv×LPF(PV, 12h)
```

Where `Effort = T_HK2 - baseline_curve` (deviation from heating curve).

**Key Parameters:**
| Symbol | Value | CV | Interpretation |
|--------|-------|-----|----------------|
| g_effort | 0.208 | 9% | Stable - reliable for optimization |
| g_outdoor | 0.442 | 95% | Unstable - confounded |
| g_pv | 0.012 | 67% | Moderate variability |

**Heating Curve:**

$$T_{HK2} = T_{setpoint} + curve\_rise \times (T_{ref} - T_{outdoor})$$

Where $T_{ref} = 21.32$°C (comfort) or $19.18$°C (eco).

**Causal Coefficients (for optimization):**
| Parameter | Causal Effect |
|-----------|---------------|
| setpoint | +0.21°C per 1°C increase |
| curve_rise | +2.92°C per unit increase |

### Heat Pump Model
```
COP = 6.52 + 0.1319×T_outdoor - 0.1007×T_flow
Heat produced = Electricity consumed × COP
```

### Energy System
```
Self-sufficiency = 1 - (Grid Import / Total Consumption)
Battery losses = Charge × (1 - Efficiency) = Charge × 0.163
```

### Combined Optimization Target
```
Primary: Maximize Comfort_Compliance (% time T_room in comfort band ≥ 95%)
Secondary: Minimize Σ [Grid_Import(t) × Price(t)] - [Grid_Export(t) × FeedIn_Tariff(t)]
```
