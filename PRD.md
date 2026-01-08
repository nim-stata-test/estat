# Research Design Plan: Heating Strategy Optimization

## Objective

Optimize heating strategy for a residential building with solar/battery system to:
1. **Primary**: Minimize energy expenditure
2. **Secondary**: Maintain comfortable temperature in key rooms
3. **Tertiary**: Optimize costs using electricity tariffs (incorporated in experiment)

**Constraint**: Prioritize solar power over grid electricity

---

## Key Insights from Analysis (Jan 2026)

### System Performance
| Metric | Value | Interpretation |
|--------|-------|----------------|
| PV Generation | 55.9 kWh/day | Strong solar resource |
| Total Consumption | 24.6 kWh/day | Net exporter |
| Self-sufficiency | 44% | Room for improvement |
| Heat Pump COP | 3.55 (2.49–5.18) | Good efficiency |
| Forced Grid Heating | 4.7% | Limited optimization headroom |

### Critical Findings
1. **Battery degradation confirmed**: Feb-Mar 2025 deep-discharge event caused statistically significant 5–11 percentage point drop in round-trip efficiency (p<0.001). Must factor this into optimization.

2. **Heating curve well-characterized**: Linear model (R²=0.94) enables predictive control. Key lever: curve rise (1.08) affects flow temperature by ~1.6–2.1°C per 0.1 change.

3. **Schedule already efficient**: Only 4.7% of heating requires grid. Gains likely come from shifting heating into solar hours, not reducing total consumption.

4. **Control parameters identified**: Setpoint, curve rise, and schedule timing are controllable. Buffer tank requires manual adjustment.

---

## Data Overview

### Energy Balance Data (`data/`)
- **Coverage**: March 2023 – December 2025 (~3 years)
- **Granularities**: Daily (15-min intervals, Watts), Monthly/Yearly (kWh)
- **Metrics**: PV generation, battery charging/discharging, direct consumption, external supply, grid feed-in

### Sensor Data (`mainic*.csv`)
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

## Phase 3: System Modeling — IN PROGRESS

### 3.1 Thermal Model
- [ ] Estimate building thermal characteristics:
  - Heat loss coefficient (W/K) from temperature decay curves
  - Thermal mass from temperature response to heating inputs
- [ ] Model room temperature dynamics as function of:
  - Outdoor temperature
  - Solar gain
  - Heating input
  - Internal gains (occupancy, appliances)

**Data available**: 12 room temperature sensors, outdoor temperature, heating circuit data (64 days overlap)

### 3.2 Heat Pump Model — PARTIALLY COMPLETE
- [x] COP as function of outdoor temperature: Mean COP 3.55 (2.49–5.18)
- [x] Heating curve model (R² = 0.94): `T_target = T_setpoint + 1.08 × (T_ref - T_outdoor)`
- [ ] COP vs flow temperature relationship (need more data variation)
- [ ] Capacity constraints and modulation behavior
- [ ] Buffer tank dynamics

**Controllable parameters identified**:
- `curve_rise` (currently 1.08, range ~0.97–1.14)
- `T_setpoint` (Comfort: 20.2°C, Eco: 18.0°C)
- Schedule timing (currently 06:30–20:00)

### 3.3 Energy System Model
- [ ] PV generation prediction (from historical patterns + weather)
- [x] Battery efficiency: ~85% pre-event, **~75-80% post-event** (degraded)
- [ ] Battery state-of-charge dynamics
- [ ] Grid interaction constraints

---

## Phase 4: Optimization Strategy Development

### 4.1 Control Variables (Confirmed from EDA)
| Variable | Current Value | Controllable? | Impact |
|----------|---------------|---------------|--------|
| Room setpoint (comfort) | 20.2°C | Yes (Home Assistant) | +1°C → +1°C flow temp |
| Room setpoint (eco) | 18.0°C | Yes (Home Assistant) | +1°C → +1°C flow temp |
| Curve rise | 1.08 | Yes (heat pump) | +0.1 → +1.6–2.1°C flow temp |
| Comfort schedule start | 06:30 | Yes (heat pump) | Pre-heat timing |
| Comfort schedule end | 20:00–21:30 | Yes (heat pump) | Evening heating mode |
| Buffer tank target | TBD | Manual only | Thermal storage |

### 4.2 Optimization Approaches
- [ ] Rule-based heuristics:
  - Pre-heat during solar hours (shift comfort start earlier in winter)
  - Lower curve rise when grid-dependent (reduce flow temps)
  - Buffer charging aligned with PV peak
- [ ] Model Predictive Control (MPC):
  - Rolling horizon optimization
  - Weather forecast integration
  - PV generation prediction

**Insight from EDA**: Only 4.7% of heating time requires forced grid consumption. Optimization potential is limited but still meaningful during winter when solar is scarce.

### 4.3 Constraint Handling
- Comfort bounds: 18–23°C acceptable range (flexible)
- Equipment limits: heat pump capacity ~50 kWh/day max observed
- Solar priority: penalize grid consumption in objective function
- Battery degradation: account for reduced efficiency (~75-80%) post-event

### 4.4 Testable Predictions
- [ ] Quantify expected reduction in grid consumption (kWh/month)
- [ ] Predict comfort impact (temperature deviations from baseline)
- [ ] Estimate solar self-consumption improvement (%)

**Baseline metrics** (from EDA):
- Current self-sufficiency: 44%
- Current grid import: 10.2 kWh/day
- Current forced-grid heating: 4.7% of heating time

---

## Phase 5: Randomized Intervention Study (Winter 2027-2028)

### 5.1 Study Design
- **Type**: Randomized crossover intervention study
- **Duration**: November 2027 - March 2028 (~20 weeks)
- **Intervention period**: 3-5 days per condition (randomized)
- **Conditions**: 3 parameter sets (baseline + 2 optimized strategies)

### 5.2 Parameter Sets to Compare
Three strategies (baseline + 2 optimized):
1. **Baseline**: Current settings (control)
2. **Energy-optimized**: Minimize grid consumption - aggressive pre-heating during solar hours, buffer charging at PV peak, reduced targets when grid-dependent
3. **Cost-optimized**: Minimize electricity costs - shift consumption to low-tariff periods, maximize value of solar export during high-tariff windows

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
| Electricity cost | Net cost (import - export revenue) per heating degree day | Minimize |
| Comfort compliance | % time rooms within 18-23°C bounds | ≥95% for all strategies |
| Solar utilization | % of heating energy from solar/battery | Maximize |
| COP achieved | Heat delivered / electricity consumed | Track by strategy |

### 5.6 Statistical Analysis
- Mixed-effects regression: energy ~ strategy + outdoor_temp + solar_availability + (1|block)
- Pairwise comparisons between strategies with weather covariates
- Confidence intervals on effect sizes
- Identify interaction effects (e.g., strategy × weather conditions)

### 5.7 Sample Size Considerations
- ~20 weeks ÷ 4 days/block = ~35 blocks
- With 3 strategies: ~12 blocks per strategy
- Power analysis: sufficient to detect 15% difference in energy consumption

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
- [ ] Tariff data: acquire current and historical tariff data before experiment (by Fall 2027)
- [ ] More variation in curve_rise parameter needed to fully characterize COP vs flow temp

### Timeline
- **2025 Q4**: Phase 1-2 complete ✓
- **2026**: Continue data collection, Phase 3 modeling
- **Fall 2027**: Finalize parameter sets, prepare randomization schedule
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
- **Comfort bounds**: Flexible range 18–23°C acceptable
- **Tariff optimization**: Deferred to future phase as planned
- **Current baseline performance**:
  - Heat pump COP: 3.55 mean (good efficiency)
  - Self-sufficiency: 44%
  - Forced grid for heating: only 4.7% of time
  - Battery efficiency: ~75-80% (degraded from ~85%)
