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

# Run all phases
python src/run_all.py              # Run complete pipeline
python src/run_all.py --phase 1    # Run Phase 1 only
python src/run_all.py --phase 2    # Run Phase 2 only
python src/run_all.py --phase 3    # Run Phase 3 only
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
└── phase3/              # System Modeling
    ├── run_phase3.py                 # Wrapper: runs all models + HTML report
    ├── 01_thermal_model.py           # Building thermal characteristics
    ├── 02_heat_pump_model.py         # COP relationships, buffer tank
    └── 03_energy_system_model.py     # PV patterns, battery, self-sufficiency
```

## Processed Data

After preprocessing, data is saved to `phase1_output/`:

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

After running `python src/phase2/run_eda.py`, outputs are saved to `phase2_output/`:

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
- `phase2_output/battery_degradation_analysis.png` - 4-panel visualization
- `phase2_output/battery_degradation_report.rtf` - Detailed report with methods & results

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
- `phase2_output/fig12_heating_curve_schedule.png` - 4-panel visualization
- `phase2_output/heating_curve_schedules.csv` - Detected schedule regimes
- `phase2_output/heating_curve_report_section.html` - HTML section for report

**Model:**
```
T_target = T_setpoint + curve_rise × (T_ref - T_outdoor)
```

**Key features:**
- Detects comfort/eco schedule from step changes in target temperature
- Estimates time-varying schedule regimes (comfort start/end times)
- Models relationship between setpoint, curve rise, and outdoor temperature
- R² = 0.94, RMSE = 1.03°C overall (0.35°C comfort, 1.54°C eco)

**Detected schedule regimes:**
- 2025-10-30 to 2025-12-26: Comfort 06:30 - 20:00
- 2025-12-27 to 2026-01-03: Comfort 06:30 - 21:30

## Phase 3: System Modeling Outputs

After running `python src/phase3/run_phase3.py`, outputs are saved to `phase3_output/`:

**Figures (fig13-fig15):**
- Thermal model (temperature simulation, decay analysis)
- Heat pump model (COP vs temperature, capacity, buffer tank)
- Energy system (daily profiles, battery patterns, self-sufficiency)

**Reports:**
- `phase3_modeling_report.html` - Combined modeling report
- `thermal_model_results.csv` - Per-room thermal parameters
- `heat_pump_daily_stats.csv` - Daily COP and energy statistics

**Key Model Results:**
- Building time constant: ~54-60 hours
- COP model (R²=0.95): `COP = 6.52 + 0.13×T_outdoor - 0.10×T_flow`
- Current self-sufficiency: 58%, potential with optimization: 85%

## Documentation

- `PRD.md` - Research design plan for heating strategy optimization
- `docs/phase3_models.md` - Detailed documentation of Phase 3 models (assumptions, equations, interpretation)

## Research Design

See `PRD.md` for the heating strategy optimization research plan.
