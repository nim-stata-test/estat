# Research Design Plan: Heating Strategy Optimization

## Objective

Optimize heating strategy for a residential building with solar/battery system to:
1. **Primary**: Minimize energy expenditure
2. **Secondary**: Maintain comfortable temperature in key rooms
3. **Tertiary**: Optimize costs using electricity tariffs (incorporated in experiment)

**Constraint**: Prioritize solar power over grid electricity

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
  - Stiebel Eltron ISG heat pump (70 sensors): temperatures, energy consumed/produced
  - Davis weather station: inside/outside temperature, humidity, wind chill
  - Room sensors: atelier, bric temperatures
  - Heat pump circuits (wp_*): buffer tank, heating circuits HK1/HK2, hot water

---

## Phase 1: Data Preprocessing

### 1.1 Energy Balance Files
- [ ] Parse all daily/monthly/yearly CSVs handling European number format (comma decimals)
- [ ] Remove duplicate "Direct consumption" columns
- [ ] Convert daily Watts to kWh for unit harmonization
- [ ] Concatenate into unified time-series dataset with consistent datetime index
- [ ] Detect and handle outliers (sensor malfunctions, zero readings during expected generation)
- [ ] Validate aggregations: daily sums should approximate monthly values

### 1.2 Sensor Data (mainic)
- [ ] Parse InfluxDB annotated CSV format (handle #datatype headers)
- [ ] Split by domain/entity_id into separate files:
  - `heating/` - Stiebel Eltron ISG sensors, wp_* sensors
  - `weather/` - Davis weather station
  - `rooms/` - Temperature sensors per room
  - `energy/` - Smart plug power consumption
- [ ] Standardize units (e.g., Davis outdoor temp is Fahrenheit, convert to Celsius)
- [ ] Resample to consistent time intervals (15-min to match energy data)
- [ ] Handle missing values and sensor dropouts

### 1.3 Data Integration
- [ ] Merge energy balance and sensor data on timestamp
- [ ] Note: Only ~3 months overlap (Oct-Dec 2025) - flag this limitation

---

## Phase 2: Exploratory Data Analysis

### 2.1 Energy Patterns
- [ ] Time-series plots: PV generation, consumption, grid import/export by day/month/year
- [ ] Seasonal decomposition of energy consumption
- [ ] Heatmaps: hourly consumption patterns by day-of-week and season
- [ ] Self-sufficiency ratio over time (direct consumption / total consumption)

### 2.2 Heating System Analysis
- [ ] Heat pump COP estimation: produced heat / consumed electricity
- [ ] Temperature differential analysis: outdoor vs indoor vs target
- [ ] Heating demand vs outdoor temperature correlation
- [ ] Buffer tank behavior: charging/discharging cycles

### 2.3 Solar-Heating Correlation
- [ ] Overlap analysis: when is solar available vs when is heating needed?
- [ ] Battery utilization for heating: evening/night heating from stored solar
- [ ] Identify periods of forced grid consumption for heating

### 2.4 Summary Statistics
- [ ] Monthly/seasonal energy consumption breakdown
- [ ] Heating degree days analysis
- [ ] Peak demand identification

---

## Phase 3: System Modeling

### 3.1 Thermal Model
- [ ] Estimate building thermal characteristics:
  - Heat loss coefficient (W/K) from temperature decay curves
  - Thermal mass from temperature response to heating inputs
- [ ] Model room temperature dynamics as function of:
  - Outdoor temperature
  - Solar gain
  - Heating input
  - Internal gains (occupancy, appliances)

### 3.2 Heat Pump Model
- [ ] COP as function of outdoor temperature and flow temperature
- [ ] Capacity constraints and modulation behavior
- [ ] Buffer tank dynamics

### 3.3 Energy System Model
- [ ] PV generation prediction (from historical patterns + weather)
- [ ] Battery state-of-charge dynamics
- [ ] Grid interaction constraints

---

## Phase 4: Optimization Strategy Development

### 4.1 Control Variables
- Heat pump target temperatures (HK1, HK2, buffer, hot water)
- Heating schedule (time-of-day setpoints)
- Buffer tank charging strategy
- Battery allocation (heating vs other loads)

### 4.2 Optimization Approaches
- [ ] Rule-based heuristics:
  - Pre-heat during solar hours
  - Lower setpoints when grid-dependent
  - Buffer charging aligned with PV peak
- [ ] Model Predictive Control (MPC):
  - Rolling horizon optimization
  - Weather forecast integration
  - PV generation prediction

### 4.3 Constraint Handling
- Comfort bounds: minimum temperatures per room/time
- Equipment limits: heat pump capacity, battery limits
- Solar priority: penalize grid consumption in objective function

### 4.4 Testable Predictions
- [ ] Quantify expected reduction in grid consumption (kWh/month)
- [ ] Predict comfort impact (temperature deviations from baseline)
- [ ] Estimate solar self-consumption improvement (%)

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
- Visualization: matplotlib, plotly
- Modeling: scikit-learn, statsmodels, or custom thermal models
- Optimization: scipy.optimize, cvxpy, or Pyomo

### Data Gaps to Address
- Sensor data only covers 3.5 months - need continued logging through 2027
- Tariff data: acquire current and historical tariff data before experiment (by Fall 2027)

### Timeline
- **2025-2027**: Continue data collection, complete Phases 1-4
- **Fall 2027**: Finalize parameter sets, prepare randomization schedule
- **Winter 2027-2028**: Execute randomized intervention study (Phase 5)

### Risks
- Limited overlap between energy balance and sensor data (only Oct-Dec 2025)
- Partial heat pump control - some parameters require manual adjustment
- Occupancy patterns may vary between baseline and treatment

---

## Confirmed Requirements

- **Heat pump control**: Partial - some setpoints controllable via Home Assistant, others manual only. Strategy must work within these constraints.
- **Room sensors**: All heated rooms have temperature sensors (data extraction will reveal full list)
- **Comfort bounds**: Flexible range 18-23°C acceptable, allowing aggressive optimization
- **Tariff optimization**: Deferred to future phase as planned
