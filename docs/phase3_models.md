# Phase 3: System Models Documentation

This document provides detailed documentation of the models developed in Phase 3, including their theoretical basis, assumptions, limitations, and interpretation guidelines.

---

## Table of Contents

1. [Thermal Model (3.1)](#1-thermal-model) — *Reference only, superseded by Grey-Box*
   - 1b. [Grey-Box Thermal Model (3.1b)](#1b-grey-box-thermal-model) — **Primary model for optimization**
   - 1c. [Model Comparison](#1c-model-comparison)
2. [Heat Pump Model (3.2)](#2-heat-pump-model)
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

## 1b. Grey-Box Thermal Model

> **Primary Model**: This grey-box model is used for Phase 4 optimization forward simulation.
> It supersedes the transfer function model (Section 1) which is retained for reference.

### 1b.1 Theoretical Background

The grey-box model is a physics-based two-state discrete-time model that explicitly models:
1. **Buffer tank dynamics**: Thermal storage between heat pump and distribution system
2. **Room temperature dynamics**: Building thermal mass and heat transfer

This improves on the transfer function model by capturing the intermediate thermal storage stage.

### 1b.2 State Variables

| Symbol | Variable | Code Name | Units | Description |
|--------|----------|-----------|-------|-------------|
| $T_{buf}$ | Buffer tank temperature | `T_buffer` | °C | Thermal storage state |
| $T_{room}$ | Room temperature | `T_room` | °C | Comfort objective |

### 1b.3 Model Formulation

The discrete-time state-space model with timestep $\Delta t = 0.25$ h (15 minutes):

**Buffer Tank Dynamics:**

$$T_{buf}[k+1] = T_{buf}[k] + \frac{\Delta t}{\tau_{buf}} \left[ (T_{HK2}[k] - T_{buf}[k]) - r_{emit} \cdot (T_{buf}[k] - T_{room}[k]) \right]$$

**Room Temperature Dynamics:**

$$T_{room}[k+1] = T_{room}[k] + \frac{\Delta t}{\tau_{room}} \left[ r_{heat} \cdot (T_{buf}[k] - T_{room}[k]) - (T_{room}[k] - T_{out}[k]) \right] + k_{solar} \cdot PV[k]$$

**Physical Interpretation:**
- The buffer tank receives heat from the heating circuit ($T_{HK2}$) and emits to the room
- The room receives heat from the buffer and loses heat to outdoors
- Solar gains add directly to room temperature

### 1b.4 Input Variables

| Symbol | Variable | Code Name | Units | Source |
|--------|----------|-----------|-------|--------|
| $T_{HK2}$ | Flow temperature | `wp_anlage_hk2_ist` | °C | Heating circuit 2 actual |
| $T_{out}$ | Outdoor temperature | `stiebel_eltron_isg_outdoor_temperature` | °C | Heat pump sensor |
| $PV$ | Solar generation | `pv_generation_kwh` | kWh | 15-min solar production |

### 1b.5 Model Parameters

| Symbol | Parameter | Value | 95% CI | Units | Physical Meaning |
|--------|-----------|-------|--------|-------|------------------|
| $\tau_{buf}$ | `tau_buf` | 2.28 | [1.8, 2.8] | h | Buffer tank time constant |
| $\tau_{room}$ | `tau_room` | 72.0 | [60, 84] | h | Building time constant (3 days) |
| $r_{emit}$ | `r_emit` | 0.10 | [0.08, 0.12] | — | Emitter/HP coupling ratio |
| $r_{heat}$ | `r_heat` | 1.05 | [0.9, 1.2] | — | Heat transfer ratio |
| $k_{solar}$ | `k_solar` | 0.017 | [0.01, 0.03] | K/kWh | Solar gain coefficient |
| $c_{offset}$ | `c_offset` | -0.085 | [-0.2, 0.1] | K/h | Temperature offset (bias correction) |

### 1b.6 Heating Curve Integration

The heating curve model from Phase 2 determines $T_{HK2}$ from controllable parameters:

$$T_{HK2} = T_{setpoint} + curve\_rise \times (T_{ref} - T_{outdoor})$$

Where:
- $T_{setpoint}$ = target room temperature (controllable: 19-22°C comfort, 12-19°C eco)
- $curve\_rise$ = heating curve slope (controllable: 0.80-1.20)
- $T_{ref}$ = reference temperature (21.32°C comfort, 19.18°C eco)

This enables forward simulation with different heating strategies.

### 1b.7 Forward Simulation Algorithm

```
Initialize:
    x[0] = [T_buffer[0], T_room[0]]  # From historical data

For k = 0 to N-1:
    # Generate T_HK2 from heating curve + schedule
    T_HK2[k] = compute_from_schedule(hour[k], params)

    # Buffer tank update
    dT_buf = (dt/tau_buf) * [(T_HK2[k] - T_buf[k]) - r_emit*(T_buf[k] - T_room[k])]
    T_buf[k+1] = T_buf[k] + dT_buf

    # Room temperature update
    dT_room = (dt/tau_room) * [r_heat*(T_buf[k] - T_room[k]) - (T_room[k] - T_out[k])]
    T_room[k+1] = T_room[k] + dT_room + k_solar*PV[k] + c_offset*dt
```

**Implementation**: `src/shared/thermal_simulator.py`

### 1b.8 Parameter Estimation

Parameters are estimated by minimizing prediction error on historical data:

1. **Objective**: Minimize RMSE between predicted and actual $T_{room}$
2. **Method**: Bounded optimization (scipy.optimize.minimize with L-BFGS-B)
3. **Constraints**: Physical bounds on all parameters (see table above)
4. **Data**: 64-day overlap period with all sensors available

### 1b.9 Model Performance

| Metric | Value | Interpretation |
|--------|-------|----------------|
| R² | 0.995 | Explains 99.5% of temperature variance |
| RMSE | 0.064°C | Excellent prediction accuracy |
| MAE | 0.047°C | Mean absolute error |
| Bias | 0.002°C | Negligible systematic error |

### 1b.10 Assumptions

| Assumption | Justification | Impact if Violated |
|------------|---------------|-------------------|
| Two-state lumped model | Buffer + room capture main dynamics | Misses floor/wall thermal mass |
| Linear heat transfer | Small temperature differences | Valid for ΔT < 20K typical |
| Constant parameters | Building characteristics stable | Seasonal variation possible |
| 15-min timestep | Matches sensor data resolution | Adequate for slow dynamics |
| Single room temperature | Weighted average of sensors | Misses room-to-room variation |

### 1b.11 Integration with Phase 4 Optimization

The grey-box model is used for **forward simulation** in Phase 4 optimization:

1. **Parameters loaded from**: `output/phase3/greybox_model_params.json`
2. **Shared module**: `src/shared/thermal_simulator.py`
3. **Used by**:
   - `src/phase4/04_pareto_optimization.py` (NSGA-II multi-objective)
   - `src/phase4/05_strategy_evaluation.py` (comfort violation analysis)
   - `src/phase4/06_strategy_detailed_analysis.py` (detailed time series)

**Warmup Period**: 96 timesteps (24 hours) to allow transient decay before evaluation.

### 1b.12 Advantages over Transfer Function Model

| Aspect | Transfer Function | Grey-Box |
|--------|-------------------|----------|
| Buffer tank | Not modeled | Explicit state |
| Physical parameters | Abstract coefficients | Interpretable (time constants, ratios) |
| R² | 0.10-0.24 | 0.995 |
| RMSE | ~5°C | 0.064°C |
| Forward simulation | Unstable (diverges) | Stable |
| Optimization use | Not suitable | Primary model |

---

## 1c. Model Comparison

### Side-by-Side Performance

| Metric | Transfer Function | Grey-Box | Improvement |
|--------|-------------------|----------|-------------|
| R² | 0.683 (weighted) | 0.995 | +46% |
| RMSE | 0.50°C | 0.064°C | 8× better |
| Forward simulation | Diverges | Stable | ✓ |
| Physical interpretability | Low | High | ✓ |

### When to Use Each Model

- **Grey-Box (Primary)**: All optimization, forward simulation, strategy evaluation
- **Transfer Function (Reference)**: Quick estimates, validation comparison, educational

### Visual Comparison

See `output/phase3/fig18b_greybox_model.png` for side-by-side visualization including:
- State trajectories (T_buffer, T_room)
- Prediction residuals
- Model comparison panel

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

> **Note**: The **Grey-Box Thermal Model** (Section 1b) is used for forward simulation in Phase 4 optimization.
> See `src/shared/thermal_simulator.py` for the shared implementation.

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
      │ Grey-Box  │  │ Heat Pump │  │  Energy   │
      │  Thermal  │  │   Model   │  │  System   │
      │   Model   │  │           │  │   Model   │
      │ T_room(t) │  │  COP(t)   │  │  PV(t)    │
      │ T_buf(t)  │  │           │  │ Battery(t)│
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
| Thermal | Compare simulated vs actual temps | RMSE ~5.7°C (moderate) |
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

### Grey-Box Thermal Model (Primary)

**State Variables:**
- $T_{buf}$ — Buffer tank temperature (°C)
- $T_{room}$ — Room temperature (°C)

**Discrete-Time Equations** ($\Delta t = 0.25$ h):

$$T_{buf}[k+1] = T_{buf}[k] + \frac{\Delta t}{\tau_{buf}} \left[ (T_{HK2}[k] - T_{buf}[k]) - r_{emit}(T_{buf}[k] - T_{room}[k]) \right]$$

$$T_{room}[k+1] = T_{room}[k] + \frac{\Delta t}{\tau_{room}} \left[ r_{heat}(T_{buf}[k] - T_{room}[k]) - (T_{room}[k] - T_{out}[k]) \right] + k_{solar} \cdot PV[k]$$

**Parameters:**
| Symbol | Code | Value | Units |
|--------|------|-------|-------|
| $\tau_{buf}$ | `tau_buf` | 2.28 | h |
| $\tau_{room}$ | `tau_room` | 72.0 | h |
| $r_{emit}$ | `r_emit` | 0.10 | — |
| $r_{heat}$ | `r_heat` | 1.05 | — |
| $k_{solar}$ | `k_solar` | 0.017 | K/kWh |

**Heating Curve:**

$$T_{HK2} = T_{setpoint} + curve\_rise \times (T_{ref} - T_{outdoor})$$

Where $T_{ref} = 21.32$°C (comfort) or $19.18$°C (eco).

### Transfer Function Model (Reference)
```
Time constant: τ = C/UA ≈ 54-60 hours
Temperature change: ΔT = a×(T_flow - T_room) - b×(T_room - T_out) + c×PV
```

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
