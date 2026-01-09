# Research Design Plan: Heating Strategy Optimization

## Objective

Optimize heating strategy for a residential building with solar/battery system to:
1. **Primary**: Minimize energy expenditure
2. **Secondary**: Maintain comfortable temperature in key rooms
3. **Tertiary**: Optimize costs using electricity tariffs (now modeled)

**Constraint**: Prioritize solar power over grid electricity

---

## Key Insights from Analysis (Jan 2026)

### System Performance
| Metric | Value | Interpretation |
|--------|-------|----------------|
| PV Generation | 55.9 kWh/day | Strong solar resource |
| Total Consumption | 24.6 kWh/day | Net exporter |
| Self-sufficiency | 58% (current) | Room for improvement to 85% |
| Heat Pump COP | 4.01 (2.53–5.80) | Good efficiency |
| Forced Grid Heating | 4.7% | Limited optimization headroom |
| Annual Net Cost | CHF -1,048 | Net producer (income) |

### Critical Findings
1. **Battery degradation confirmed**: Feb-Mar 2025 deep-discharge event caused statistically significant 5–11 percentage point drop in round-trip efficiency (p<0.001). Must factor this into optimization.

2. **Heating curve well-characterized**: Linear model (R²=0.98 for normal periods) enables predictive control. Key lever: curve rise (1.08) affects flow temperature by ~1.6–2.1°C per 0.1 change.

3. **Schedule already efficient**: Only 4.7% of heating requires grid. Gains likely come from shifting heating into solar hours, not reducing total consumption.

4. **Control parameters identified**: Setpoint, curve rise, and schedule timing are controllable. Buffer tank requires manual adjustment.

5. **Tariff arbitrage potential**: High/low tariff spread of 8.1 Rp/kWh. Shifting 20-30% of grid consumption to low-tariff periods could increase annual income by 2.4-3.6%.

6. **Four optimization strategies defined**: Baseline, Energy-Optimized (+0.18 COP), Aggressive Solar (+0.25 COP), and Cost-Optimized (+0.22 COP, tariff-aware).

---

## Data Overview

### Energy Balance Data (`data/`)
- **Coverage**: March 2023 – December 2025 (~3 years)
- **Granularities**: Daily (15-min intervals, Watts), Monthly/Yearly (kWh)
- **Metrics**: PV generation, battery charging/discharging, direct consumption, external supply, grid feed-in

### Sensor Data (`data/mainic*.csv`)
- **Coverage**: October 2025 – January 2026 (~3.5 months)
- **Size**: ~50 million records, 2,726 sensors
- **Key heating sensors**:
  - Stiebel Eltron ISG heat pump (96 sensors): temperatures, energy consumed/produced
  - Davis weather station (27 sensors): inside/outside temperature, humidity, wind chill
  - Room sensors (12 sensors): atelier, bric, studio, office, guest, etc.
  - Smart plug consumption (43 sensors)
  - Heat pump circuits (wp_*): buffer tank, heating circuits HK1/HK2, hot water

### Data Overlap
- **Energy + Sensor overlap**: 64 days (2025-10-28 to 2026-01-01)
- **Integrated dataset**: 6,175 rows at 15-min intervals, 185 columns

---

## Phase 1: Data Preprocessing ✓ COMPLETED

### 1.1 Energy Balance Files ✓
- [x] Parse all daily/monthly/yearly CSVs handling European number format (comma decimals)
- [x] Remove duplicate "Direct consumption" columns
- [x] Convert daily Watts to kWh for unit harmonization
- [x] Concatenate into unified time-series dataset with consistent datetime index
- [x] Detect and handle outliers (sensor malfunctions, zero readings during expected generation)
- [x] Validate aggregations: daily sums should approximate monthly values

**Outputs**: `energy_balance_15min.parquet` (99,552 rows), `energy_balance_daily.parquet` (1,037 days), `energy_balance_monthly.parquet` (36 months)

### 1.2 Sensor Data (mainic) ✓
- [x] Parse InfluxDB annotated CSV format (handle #datatype headers)
- [x] Split by domain/entity_id into separate files:
  - `sensors_heating.parquet` - 96 heat pump sensors (1.8M records)
  - `sensors_weather.parquet` - 27 Davis weather station sensors (1M records)
  - `sensors_rooms.parquet` - 12 room temperature sensors (63K records)
  - `sensors_energy.parquet` - 43 smart plug sensors (5.1M records)
- [x] Standardize units (e.g., Davis outdoor temp is Fahrenheit, convert to Celsius)
- [x] Resample to consistent time intervals (15-min to match energy data)
- [x] Handle missing values and sensor dropouts
- [x] Spike removal for cumulative counter sensors (heating/water meters)

**Data quality**: `corrections_log.csv`, `sensor_summary.csv`

### 1.3 Data Integration ✓
- [x] Merge energy balance and sensor data on timestamp
- [x] **Overlap period**: 64 days (2025-10-28 to 2026-01-01)

**Outputs**: `integrated_dataset.parquet` (6,175 rows, 185 columns), `integrated_overlap_only.parquet`

---

## Phase 2: Exploratory Data Analysis ✓ COMPLETED

### 2.1 Energy Patterns ✓
- [x] Time-series plots: PV generation, consumption, grid import/export by day/month/year
- [x] Seasonal decomposition of energy consumption
- [x] Heatmaps: hourly consumption patterns by day-of-week and season
- [x] Self-sufficiency ratio over time (direct consumption / total consumption)

**Key findings**:
- Daily averages: **55.9 kWh PV generation**, **24.6 kWh consumption**
- Grid import: 10.2 kWh/day, Grid export: 41.0 kWh/day
- **Self-sufficiency: 44%** (direct consumption / total)

### 2.2 Heating System Analysis ✓
- [x] Heat pump COP estimation: produced heat / consumed electricity
- [x] Temperature differential analysis: outdoor vs indoor vs target
- [x] Heating demand vs outdoor temperature correlation
- [x] Buffer tank behavior: charging/discharging cycles
- [x] **Heating curve model**: T_target = T_setpoint + curve_rise × (T_ref - T_outdoor)

**Key findings**:
- **Mean COP: 3.55** (range 2.49–5.18)
- Daily heating electricity: mean 23.7 kWh, max 50.0 kWh
- Heating curve slope: 1.08, Model R² = 0.94, RMSE = 1.03°C
- Setpoints: Comfort 20.2°C, Eco 18.0°C
- Schedule detected: Comfort 06:30–20:00 (extended to 21:30 late Dec)

### 2.3 Solar-Heating Correlation ✓
- [x] Overlap analysis: when is solar available vs when is heating needed?
- [x] Battery utilization for heating: evening/night heating from stored solar
- [x] Identify periods of forced grid consumption for heating

**Key findings**:
- Solar hours (8AM–5PM): 51.4% of heating activity
- Non-solar hours: 48.6% of heating activity
- **Forced grid consumption: only 4.7%** of heating time
- Avg grid import: 0.69 kWh/h (solar) vs 1.32 kWh/h (non-solar)

### 2.4 Summary Statistics ✓
- [x] Monthly/seasonal energy consumption breakdown
- [x] Heating degree days analysis
- [x] Peak demand identification

**Outputs**: `phase2_output/fig01-fig12`, `eda_report.html`

### 2.5 Battery Degradation Analysis ✓
Investigation of Feb-Mar 2025 deep-discharge event (faulty inverter).

**Key findings** (statistically significant degradation):
| Model | Effect (pp) | p-value |
|-------|-------------|---------|
| Basic OLS | -10.81 | 0.0001 |
| Seasonal-adjusted | -8.18 | 0.0024 |
| Matched-month paired t-test | -4.62 | 0.0001 |
| Welch's t-test | — | <0.05 |

**Conclusion**: Strong evidence that deep-discharge event reduced battery round-trip efficiency by ~5-11 percentage points. Factor this into future energy optimization calculations.

**Outputs**: `battery_degradation_analysis.png`, `battery_degradation_report.html`

---

## Phase 3: System Modeling ✓ COMPLETED

### 3.1 Thermal Model ✓
- [x] Estimate building thermal characteristics:
  - **Time constant: ~17-31 hours** (varies by room)
  - RC model using T_hk2 (heating circuit temp) as heating effort proxy
- [x] Model room temperature dynamics as function of:
  - Heating effort (T_hk2 - T_room differential)
  - Heat loss (T_room - T_outdoor differential)
  - Solar gain (from PV proxy)

**Key findings**:
- Heating runs continuously (38-77% duty cycle, including nights)
- T_hk2 varies 30°C (night/eco) to 36°C (morning/comfort) - good heating proxy
- Model R² = 0.10-0.24 (improved with correct heating assumption)
- Heating coefficient: ~0.013 K/(15min)/K
- Loss coefficient: ~0.014 K/(15min)/K

**Outputs**: `fig13_thermal_model.png`, `thermal_model_results.csv`

### 3.2 Heat Pump Model ✓ COMPLETED
- [x] COP as function of outdoor temperature: **+0.13 COP per °C**
- [x] COP vs flow temperature: **-0.10 COP per °C**
- [x] Multi-variable COP model (R² = 0.95):
  ```
  COP = 6.52 + 0.1319×T_outdoor - 0.1007×T_flow
  ```
- [x] Capacity analysis: Mean 21.8 kWh/day consumed, 78.5 kWh/day heat produced
- [x] Buffer tank dynamics: Mean 35.8°C, range 22-55°C, charging/discharging ~30°C/h max

**Key findings**:
- Mean COP: **4.01** (range 2.53-5.80)
- Max daily heat production: 136 kWh
- Compressor duty cycle: 8.8%
- Buffer-outdoor correlation: -0.37 (buffer hotter when colder outside)

**Outputs**: `fig14_heat_pump_model.png`, `heat_pump_daily_stats.csv`

### 3.3 Energy System Model ✓
- [x] PV generation patterns: Mean 56.2 kWh/day, peak hours 10:00-16:00
- [x] Battery efficiency: **83.7%** round-trip (degraded from Feb-Mar 2025 event)
- [x] Battery patterns: Charge 7:00-13:00, discharge 16:00-6:00
- [x] Grid interaction: Import 10.2 kWh/day, Export 41.0 kWh/day (net exporter)
- [x] Self-sufficiency scenarios modeled

**Key findings**:
- Current self-sufficiency: **58.1%**
- With 20% load shifting: 63.6% (+5.5pp)
- With 2× battery: 79.9% (+21.8pp)
- Combined optimization: **85.3%** (+27.2pp potential)

**Outputs**: `fig15_energy_system_model.png`

### 3.4 Tariff Cost Model ✓ COMPLETED
- [x] Historical cost analysis with high/low tariff breakdown
- [x] Cost forecasting model using HDD regression (R² = 0.58)
- [x] Load-shifting scenario simulation
- [x] Time-of-use tariff modeling (Primeo Energie rates)

**Key findings**:
- **Annual net income: CHF 1,048** (household is net producer)
- Grid cost: CHF 3.15/day, Feed-in revenue: CHF 6.02/day
- High-tariff share: 48.9% of grid costs
- Tariff spread: 8.1 Rp/kWh (high: 35.9, low: 27.7)
- **Optimization potential**: 2.4-3.6% income increase via load shifting

**Tariff structure** (Primeo Energie 2025):
| Period | Purchase (Rp/kWh) | Feed-in with HKN (Rp/kWh) |
|--------|-------------------|---------------------------|
| High tariff | 35.9 | 13.0-15.5 |
| Low tariff | 27.7 | 13.0-15.5 |

High tariff: Mon-Fri 06:00-21:00, Sat 06:00-12:00
Low tariff: Nights, weekends, holidays

**Outputs**: `fig16_tariff_cost_model.png`, `cost_model_daily_stats.csv`, `cost_forecast_model.json`, `phase3_modeling_report.html`

---

## Phase 4: Optimization Strategy Development ✓ COMPLETED

### 4.1 Control Variables (Updated from Phase 3 Models)
| Variable | Current Value | Controllable? | Impact (from models) |
|----------|---------------|---------------|----------------------|
| Room setpoint (comfort) | 20.2°C | Yes (Home Assistant) | +1°C → +1°C flow temp |
| Room setpoint (eco) | 18.0°C | Yes (Home Assistant) | +1°C → +1°C flow temp |
| Curve rise | 1.08 | Yes (heat pump) | +0.1 → +1.6–2.1°C flow temp |
| Flow temperature | ~34°C mean | Indirect (via setpoint/curve) | **-1°C → +0.10 COP** |
| Comfort schedule start | 06:30 | Yes (heat pump) | Pre-heat timing |
| Comfort schedule end | 20:00–21:30 | Yes (heat pump) | Evening heating mode |
| Buffer tank target | ~36°C mean | Manual only | Thermal storage |

### 4.2 Model Parameters for Optimization
| Parameter | Value | Source |
|-----------|-------|--------|
| Building time constant | 17-31 h (avg ~30h) | Thermal model |
| Heating coefficient | 0.013 K/(15min)/K | Thermal model |
| Loss coefficient | 0.014 K/(15min)/K | Thermal model |
| COP sensitivity (outdoor) | +0.13/°C | Heat pump model |
| COP sensitivity (flow) | -0.10/°C | Heat pump model |
| Battery round-trip efficiency | 83.7% | Energy system model |
| Peak PV hours | 10:00-16:00 | Energy system model |
| Daily PV generation | 56.2 kWh mean | Energy system model |

### 4.3 Four Optimization Strategies ✓ COMPLETED

| Strategy | Schedule | Curve Rise | Setpoints | COP | vs Baseline |
|----------|----------|------------|-----------|-----|-------------|
| **Baseline** | 06:30-20:00 | 1.08 | 20.2°C / 18.0°C | 4.09 | — |
| **Energy-Optimized** | 10:00-18:00 | 0.98 | 20.0°C / 17.5°C | 4.39 | +0.18 |
| **Aggressive Solar** | 10:00-17:00 | 0.95 | 21.0°C / 17.0°C | 4.46 | +0.25 |
| **Cost-Optimized** | 11:00-21:00 | 0.95/0.85* | 20.0°C / 17.0°C | 4.43 | +0.22 |

*Cost-Optimized uses curve_rise 0.85 when grid-dependent

**Strategy Goals**:
- **Energy-Optimized**: Minimize grid electricity consumption, maximize self-sufficiency
- **Aggressive Solar**: Maximum solar utilization with wider comfort tolerance (18-23°C)
- **Cost-Optimized**: Minimize annual electricity bill via tariff arbitrage

### 4.4 Rule-Based Heuristics ✓ COMPLETED
- [x] Pre-heat during solar hours (shift comfort start to 10:00-11:00)
- [x] Lower flow temps (curve rise 0.95-0.98) for improved COP
- [x] Dynamic curve rise reduction when grid-dependent (0.85)
- [x] Tariff-aware scheduling (avoid high-tariff morning hours 06:00-10:00)
- [x] Buffer charging aligned with PV peak (12:00-15:00)

### 4.5 Simulation Results ✓ COMPLETED
Validated strategies on 64 days of historical data:

| Strategy | Self-Sufficiency | Comfort Compliance | Solar Heating % |
|----------|-----------------|-------------------|-----------------|
| Baseline | 40.1% | 76.7% | 29.8% |
| Energy-Optimized | 40.1% | 76.7% | 25.7% |
| Aggressive Solar | 40.1% | 97.6% | 25.7% |
| Cost-Optimized | 40.1% | 91.9% | 21.4% |

**Note**: Self-sufficiency identical in simulation because it uses historical energy data. Real differences will emerge in Phase 5 intervention study.

### 4.6 Constraint Handling
- Comfort bounds: 18–22°C standard, 17-23°C for Aggressive Solar
- Equipment limits: heat pump capacity 136 kWh/day max observed
- Solar priority: penalize grid consumption in objective function
- Battery degradation: account for reduced efficiency (83.7%) post-event
- Comfort compliance minimum: 95% (90% for Cost-Optimized)

### 4.7 Phase 5 Preparation ✓ COMPLETED
- [x] Parameter sets defined: `phase5_parameter_sets.json`
- [x] Testable predictions with confidence intervals: `phase5_predictions.json`
- [x] Implementation protocol: `phase5_implementation_checklist.md`
- [x] Success criteria defined per strategy

**Outputs**: `fig16-18_*.png`, `phase4_optimization_report.html`, `strategy_comparison.csv`

---

## Phase 5: Randomized Intervention Study (Winter 2027-2028)

### 5.1 Study Design
- **Type**: Randomized crossover intervention study
- **Duration**: November 2027 - March 2028 (~20 weeks)
- **Intervention period**: 3-5 days per condition (randomized)
- **Conditions**: 4 parameter sets (baseline + 3 optimized strategies)

### 5.2 Parameter Sets to Compare
Four strategies defined in Phase 4:

| Strategy | Schedule | Curve Rise | Key Focus |
|----------|----------|------------|-----------|
| **Baseline** | 06:30-20:00 | 1.08 | Control (current settings) |
| **Energy-Optimized** | 10:00-18:00 | 0.98 | Minimize grid consumption |
| **Aggressive Solar** | 10:00-17:00 | 0.95 | Maximum solar utilization |
| **Cost-Optimized** | 11:00-21:00 | 0.95/0.85 | Minimize costs via tariff arbitrage |

See `phase5_parameter_sets.json` for exact parameter values.

### 5.3 Randomization Protocol
- [ ] Generate randomized schedule assigning each 3-5 day block to a condition
- [ ] Balance conditions across the winter (each strategy tested in early, mid, and late winter)
- [ ] Record exact switch times for analysis
- [ ] Parameter changes: programmatic where possible, manual intervention logged

### 5.4 Data Collection Per Block
- All sensor readings (temperatures, energy, heat pump status)
- Weather conditions (outdoor temp, solar irradiance, wind)
- Occupancy notes (deviations from normal)
- Comfort complaints/overrides

### 5.5 Success Metrics
| Metric | Definition | Target |
|--------|------------|--------|
| Grid consumption | kWh from external supply per heating degree day | Identify best-performing strategy |
| Electricity cost | Net cost (import - export revenue) per heating degree day | Minimize (esp. Cost-Optimized) |
| Comfort compliance | % time rooms within comfort bounds | ≥95% (≥90% for Cost-Optimized) |
| Solar utilization | % of heating energy from solar/battery | Maximize |
| COP achieved | Heat delivered / electricity consumed | Track by strategy |
| Tariff alignment | % consumption during low-tariff periods | Track for Cost-Optimized |

### 5.6 Statistical Analysis
- Mixed-effects regression: energy ~ strategy + outdoor_temp + solar_availability + (1|block)
- Pairwise comparisons between strategies with weather covariates
- Confidence intervals on effect sizes
- Identify interaction effects (e.g., strategy × weather conditions)

### 5.7 Sample Size Considerations
- ~20 weeks ÷ 4 days/block = ~35 blocks
- With 4 strategies: ~9 blocks per strategy
- Power analysis: sufficient to detect 15% difference in energy consumption
- Consider extending study or reducing block size to maintain statistical power

---

## Implementation Notes

### Technology Stack
- Python 3.14 with pandas, numpy for data processing
- Visualization: matplotlib
- Modeling: scikit-learn, statsmodels (OLS with robust SE)
- Optimization: scipy.optimize, cvxpy, or Pyomo

### Data Gaps to Address
- [x] ~~Sensor data only covers 3.5 months~~ → Now have 64 days overlap, sufficient for initial modeling
- [ ] Sensor data logging needs to continue through 2027 for full winter seasons
- [x] ~~Tariff data~~ → Acquired Primeo Energie rates 2023-2025, integrated into cost model
- [ ] More variation in curve_rise parameter needed to fully characterize COP vs flow temp

### Timeline
- **2025 Q4**: Phase 1-2 complete ✓
- **2026 Q1**: Phase 3-4 complete ✓ (system modeling, optimization strategies, tariff integration)
- **2026-2027**: Continue data collection, refine models
- **Fall 2027**: Prepare randomization schedule, verify parameter sets
- **Winter 2027-2028**: Execute randomized intervention study (Phase 5)

### Risks (Updated)
- ~~Limited overlap between energy balance and sensor data~~ → 64 days is workable
- **Battery degradation**: Round-trip efficiency reduced ~5-11pp after Feb-Mar 2025 event
- Partial heat pump control - some parameters require manual adjustment (buffer tank)
- Occupancy patterns may vary between baseline and treatment
- Schedule detection shows user changed settings Dec 27 (comfort end extended 1.5h) — need to track manual changes

---

## Confirmed Requirements (Updated from Analysis)

- **Heat pump control**: Partial
  - Via Home Assistant: room setpoints (comfort/eco)
  - Via heat pump interface: curve rise, schedule timing
  - Manual only: buffer tank target temperatures
- **Room sensors**: 12 temperature sensors confirmed:
  - atelier, bric, dorme, halle, office1, simlab, studio, guest, cave, plant, bano, davis_inside
- **Comfort bounds**: Flexible range 18–23°C acceptable (17-23°C for Aggressive Solar)
- **Tariff optimization**: ✓ Implemented - Cost-Optimized strategy uses Primeo Energie high/low tariffs
- **Current baseline performance**:
  - Heat pump COP: 4.01 mean (good efficiency)
  - Self-sufficiency: 58%
  - Forced grid for heating: only 4.7% of time
  - Battery efficiency: 83.7% (degraded from ~85%)
  - Annual net income: CHF 1,048 (net producer)
