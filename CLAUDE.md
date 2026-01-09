# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ESTAT is an energy balance data repository for solar/battery system monitoring. Currently contains historical energy data organized by temporal granularity.

## Data Structure

```
data/
├── daily/       # 15-minute interval readings: Energy_Balance_YYYY_MM_DD.csv
├── monthly/     # Aggregated daily data: Energy_Balance_YYYY_MM.csv
├── yearly/      # Aggregated monthly data: Energy_Balance_YYYY.csv
└── mainic*.csv  # InfluxDB export from Home Assistant (annotated CSV format)
```

## Data Format

- **Delimiter**: Semicolon (`;`)
- **Decimal separator**: Comma (European format, e.g., `1234,56`)
- **Encoding**: UTF-8 with BOM
- **Daily metrics** (15-min intervals, values in Watts):
  - Direct consumption / Mean values [W]
  - Battery discharging / Mean values [W]
  - External energy supply / Mean values [W]
  - Total consumption / Mean values [W]
  - Grid feed-in / Mean values [W]
  - Battery charging / Mean values [W]
  - PV power generation / Mean values [W]
  - Limiting of active power feed-in / Mean values [W]
- **Monthly/Yearly metrics**: Meter change values in kWh

## Development Environment

- **Language**: Python 3.14
- **IDE**: PyCharm configured
- **Virtual env**: `.venv/`

## Commands

```bash
# Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run preprocessing with HTML report (~12 min)
python src/phase1/run_preprocessing.py

# Run EDA with HTML report
python src/phase2/run_eda.py

# Run system modeling with HTML report
python src/phase3/run_phase3.py

# Run optimization with HTML report
python src/phase4/run_optimization.py

# Run all phases
python src/run_all.py              # Run complete pipeline
python src/run_all.py --phase 1    # Run Phase 1 only
python src/run_all.py --phase 2    # Run Phase 2 only
python src/run_all.py --phase 3    # Run Phase 3 only
python src/run_all.py --phase 4    # Run Phase 4 only
python src/run_all.py --step 1.2   # Run specific step
python src/run_all.py --list       # List all steps

# Run individual scripts
python src/phase1/01_preprocess_energy_balance.py  # Phase 1, Step 1
python src/phase1/02_preprocess_sensors.py         # Phase 1, Step 2 (~10 min)
python src/phase1/03_integrate_data.py             # Phase 1, Step 3
python src/phase2/01_eda.py                        # Phase 2, Step 1
python src/phase3/01_thermal_model.py              # Phase 3, Step 1
python src/phase3/02_heat_pump_model.py            # Phase 3, Step 2
python src/phase3/03_energy_system_model.py        # Phase 3, Step 3
python src/phase4/01_rule_based_strategies.py      # Phase 4, Step 1
python src/phase4/02_strategy_simulation.py        # Phase 4, Step 2
python src/phase4/03_parameter_sets.py             # Phase 4, Step 3
```

## Source Code Structure

```
src/
├── run_all.py           # Master script for running all phases
├── phase1/              # Data Preprocessing
│   ├── run_preprocessing.py          # Wrapper: runs all steps + HTML report
│   ├── 01_preprocess_energy_balance.py
│   ├── 02_preprocess_sensors.py
│   └── 03_integrate_data.py
├── phase2/              # Exploratory Data Analysis
│   ├── run_eda.py                    # Wrapper: filters sensors + HTML report
│   ├── 01_eda.py                     # Main EDA (energy, heating, solar)
│   ├── 02_battery_degradation.py     # Standalone battery analysis
│   └── 03_heating_curve_analysis.py  # Heating curve model + schedule detection
├── phase3/              # System Modeling
│   ├── run_phase3.py                 # Wrapper: runs all models + HTML report
│   ├── 01_thermal_model.py           # Building thermal characteristics
│   ├── 02_heat_pump_model.py         # COP relationships, buffer tank
│   └── 03_energy_system_model.py     # PV patterns, battery, self-sufficiency
└── phase4/              # Optimization Strategy Development
    ├── run_optimization.py           # Wrapper: runs all steps + HTML report
    ├── 01_rule_based_strategies.py   # Strategy definitions and rules
    ├── 02_strategy_simulation.py     # Validate strategies on historical data
    └── 03_parameter_sets.py          # Generate Phase 5 parameter sets
```

## Output Directory Structure

```
output/
├── phase1/    # Preprocessing outputs (parquet files, reports)
├── phase2/    # EDA outputs (figures, HTML reports)
├── phase3/    # System modeling outputs (figures, model results)
└── phase4/    # Optimization outputs (strategies, predictions)
```

## Processed Data

After preprocessing, data is saved to `output/phase1/`:

**Parquet datasets:**
- `energy_balance_15min.parquet` - 15-min interval energy data (kWh)
- `energy_balance_daily.parquet` - Daily totals from monthly files
- `energy_balance_monthly.parquet` - Monthly totals from yearly files
- `sensors_heating.parquet` - Heat pump sensors (96 sensors)
- `sensors_weather.parquet` - Davis weather station (27 sensors)
- `sensors_rooms.parquet` - Room temperature sensors (3 sensors)
- `sensors_energy.parquet` - Smart plug consumption (43 sensors)
- `integrated_dataset.parquet` - All data merged (185 columns)
- `integrated_overlap_only.parquet` - Overlap period only (64 days)

**Reports and logs:**
- `preprocessing_report.html` - Comprehensive preprocessing documentation
- `corrections_log.csv` - Data cleaning corrections applied
- `validation_results.csv` - Daily vs monthly validation results
- `sensor_summary.csv` - Per-sensor statistics
- `data_overlap_summary.csv` - Data source overlap info

## EDA Outputs

After running `python src/phase2/run_eda.py`, outputs are saved to `output/phase2/`:

**Figures (fig01-fig12):**
- Energy patterns (time series, monthly, hourly heatmaps, seasonal)
- Heating system (COP analysis, temperature differentials, indoor/outdoor)
- Solar-heating correlation (hourly patterns, battery usage, forced grid)
- Summary statistics (monthly breakdown, HDD analysis, yearly totals)
- Heating curve analysis (schedule detection, model fit, residuals)

**HTML Report:**
- `eda_report.html` - Comprehensive EDA documentation with all figures

## Battery Degradation Analysis

Standalone analysis investigating whether the Feb-Mar 2025 deep-discharge event
affected battery efficiency. Run separately from main EDA:

```bash
python src/phase2/02_battery_degradation.py
```

**Outputs:**
- `output/phase2/battery_degradation_analysis.png` - 4-panel visualization
- `output/phase2/battery_degradation_report.rtf` - Detailed report with methods & results

**Analysis includes:**
- OLS regression with time trend and post-event indicator
- Welch's t-test as robustness check
- Key finding: Statistically significant efficiency drop of ~10.8 percentage points (p<0.001)

## Heating Curve Analysis

Analysis of how target flow temperature depends on controllable parameters.
Run separately from main EDA:

```bash
python src/phase2/03_heating_curve_analysis.py
```

**Outputs:**
- `output/phase2/fig12_heating_curve_schedule.png` - 4-panel visualization
- `output/phase2/heating_curve_schedules.csv` - Detected schedule regimes
- `output/phase2/heating_curve_setpoints.csv` - Detected setpoint regimes
- `output/phase2/heating_curve_report_section.html` - HTML section for report

**Model:**
```
T_target = T_setpoint + curve_rise × (T_ref - T_outdoor)
```

**Key features:**
- Detects comfort/eco schedule from step changes in target temperature
- **Detects setpoint regime changes** (when comfort/eco temperatures change)
- Identifies anomalous periods (eco >= comfort, e.g., eco=30°C disabling schedule)
- Estimates T_ref from normal operation only (excludes anomalies)
- R² = 0.98, RMSE = 0.57°C for normal periods (0.35°C comfort, 0.77°C eco)

**Schedule regimes (timing):**
- 2025-10-30 to 2025-12-26: Comfort 06:30 - 20:00
- 2025-12-27 to 2026-01-03: Comfort 06:30 - 21:30

**Setpoint regimes (notable changes):**
- 2025-10-28 to 2025-12-04: comfort ~20.2-20.5°C, eco ~18.0-18.7°C
- 2025-12-04 to 2025-12-07: eco=30°C (anomalous - effectively 24h comfort)
- 2025-12-14+: comfort rising to 21°C

## Phase 3: System Modeling Outputs

After running `python src/phase3/run_phase3.py`, outputs are saved to `output/phase3/`:

**Figures (fig13-fig15):**
- Thermal model (temperature simulation, decay analysis)
- Heat pump model (COP vs temperature, capacity, buffer tank)
- Energy system (daily profiles, battery patterns, self-sufficiency)

**Reports:**
- `phase3_modeling_report.html` - Combined modeling report
- `thermal_model_results.csv` - Per-room thermal parameters
- `heat_pump_daily_stats.csv` - Daily COP and energy statistics

**Thermal Model Sensors (weighted indoor temperature):**
- davis_inside_temperature: 40%
- office1_temperature: 30%
- atelier_temperature: 10%
- studio_temperature: 10%
- simlab_temperature: 10%

**Key Model Results:**
- Building time constant: ~14-33 hours (varies by sensor)
- davis_inside: tau=14.1h, office1: tau=17.5h, atelier: tau=29.5h
- COP model (R²=0.95): `COP = 6.52 + 0.13×T_outdoor - 0.10×T_flow`
- Current self-sufficiency: 58%, potential with optimization: 85%

## Phase 4: Optimization Strategy Outputs

After running `python src/phase4/run_optimization.py`, outputs are saved to `output/phase4/`:

**Figures (fig16-fig18):**
- Strategy comparison (COP by strategy, schedule alignment, expected improvements)
- Simulation results (time series, self-sufficiency, hourly COP profiles)
- Parameter space (trade-offs, parameter summary table)

**Reports:**
- `phase4_optimization_report.html` - Combined optimization report
- `optimization_strategies.csv` - Strategy definitions and rules
- `strategy_comparison.csv` - Simulated metrics by strategy
- `simulation_daily_metrics.csv` - Daily simulation results

**Phase 5 Preparation:**
- `phase5_parameter_sets.json` - Exact parameter values for intervention study
- `phase5_predictions.json` - Testable predictions with confidence intervals
- `phase5_implementation_checklist.md` - Protocol for randomized study

**Heating Curve Model (from Phase 2):**
```
T_flow = T_setpoint + curve_rise × (T_ref - T_outdoor)
```
Where:
- T_ref = 21.32°C (comfort mode) or 19.18°C (eco mode)
- curve_rise typically 0.85-1.08

**Three Optimization Strategies:**

| Strategy | Schedule | Curve Rise | COP | vs Baseline |
|----------|----------|------------|-----|-------------|
| Baseline | 06:30-20:00 | 1.08 | 4.09 | — |
| Energy-Optimized | 10:00-18:00 | 0.98 | 4.39 | +0.18 |
| Aggressive Solar | 10:00-17:00 | 0.95 | 4.46 | +0.25 |

**Key optimization levers:**
- Shift comfort mode to PV peak hours (10:00-17:00)
- Lower curve_rise for better COP (~0.1 COP improvement per 1°C flow temp reduction)
- Wider comfort band (17-23°C for aggressive strategy)
- Dynamic curve_rise reduction when grid-dependent (0.85-0.90)

## Documentation

- `PRD.md` - Research design plan for heating strategy optimization
- `docs/phase3_models.md` - Detailed documentation of Phase 3 models (assumptions, equations, interpretation)

## Research Design

See `PRD.md` for the heating strategy optimization research plan.
