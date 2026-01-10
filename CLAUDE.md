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
├── mainic*.csv  # InfluxDB export from Home Assistant (annotated CSV format)
└── tariffs/     # Electricity tariff data (Primeo Energie)
    └── primeo_tariffs_source.json  # Manually collected tariff rates
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
python src/phase1/04_preprocess_tariffs.py         # Phase 1, Step 4 (tariffs)
python src/phase2/01_eda.py                        # Phase 2, Step 1
python src/phase2/04_tariff_analysis.py            # Phase 2, Step 4 (tariff EDA)
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
│   ├── 03_integrate_data.py
│   └── 04_preprocess_tariffs.py      # Electricity tariff preprocessing
├── phase2/              # Exploratory Data Analysis
│   ├── run_eda.py                    # Wrapper: filters sensors + HTML report
│   ├── 01_eda.py                     # Main EDA (energy, heating, solar)
│   ├── 02_battery_degradation.py     # Standalone battery analysis
│   ├── 03_heating_curve_analysis.py  # Heating curve model + schedule detection
│   └── 04_tariff_analysis.py         # Electricity tariff analysis (HKN-only)
├── phase3/              # System Modeling
│   ├── run_phase3.py                 # Wrapper: runs all models + HTML report
│   ├── 01_thermal_model.py           # Building thermal characteristics
│   ├── 02_heat_pump_model.py         # COP relationships, buffer tank
│   ├── 03_energy_system_model.py     # PV patterns, battery, self-sufficiency
│   └── 04_tariff_cost_model.py       # Electricity cost analysis + forecasting
├── phase4/              # Optimization Strategy Development
│   ├── run_optimization.py           # Wrapper: runs all steps + HTML report
│   ├── run_pareto.py                 # CLI wrapper for Pareto optimization
│   ├── 01_rule_based_strategies.py   # Strategy definitions and rules
│   ├── 02_strategy_simulation.py     # Validate strategies on historical data
│   ├── 03_parameter_sets.py          # Generate Phase 5 parameter sets
│   └── 04_pareto_optimization.py     # NSGA-II multi-objective optimization
└── phase5/              # Intervention Study
    ├── estimate_study_parameters.py  # Data-driven washout/block estimation
    └── generate_schedule.py          # Randomization schedule generator
```

## Output Directory Structure

```
output/
├── phase1/    # Preprocessing outputs (parquet files, reports)
├── phase2/    # EDA outputs (figures, HTML reports)
├── phase3/    # System modeling outputs (figures, model results)
├── phase4/    # Optimization outputs (strategies, predictions)
└── phase5/    # Intervention study outputs (schedules, logs, analysis)
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

## Electricity Tariffs

Tariff data from Primeo Energie (provider) for cost modeling.

**Important:** Only feed-in tariffs WITH HKN (Herkunftsnachweis) are used in analysis.
Base-only feed-in rates are excluded because this installation participates in the
HKN program, receiving the additional HKN bonus on top of the base rate.

Run:

```bash
python src/phase1/04_preprocess_tariffs.py
```

**Data sources:**
- Primeo Energie official announcements and price sheets
- ElCom LINDAS SPARQL endpoint (official Swiss tariff database)

**Tariff time windows:**
| Tariff | Time Windows |
|--------|--------------|
| High (Hochtarif) | Mon-Fri 06:00-21:00, Sat 06:00-12:00 |
| Low (Niedertarif) | Mon-Fri 21:00-06:00, Sat 12:00 - Mon 06:00, Federal holidays |

**Swiss holidays (low tariff all day):**
- January 1, Good Friday, Easter Saturday/Monday, Ascension, Whit Monday, August 1, December 25

**Purchase tariffs (Rp/kWh):**
| Period | Single Rate | Average Estimate | Notes |
|--------|-------------|------------------|-------|
| 2023 | — | 31.3 | Peak energy crisis year |
| 2024 | 16.50 | 32.8 | +4.8% vs 2023 |
| 2025 | — | 32.6 | -1% vs 2024 |

**Feed-in tariffs (Rückliefervergütung, Rp/kWh):**
| Period | Base Rate | With HKN | Notes |
|--------|-----------|----------|-------|
| Jan-Jun 2023 | 20.0 | 21.5 | Peak rate after energy crisis |
| Jul-Dec 2023 | 16.0 | 17.5 | Market price reduction |
| Jan-Jun 2024 | 16.0 | 17.5 | Stable |
| Jul-Dec 2024 | 13.0 | 14.5 | -20% announced cut |
| Jan-Mar 2025 | 13.0 | 15.5 | HKN increased to 2.5 Rp |
| Apr 2025+ | 10.5 | 13.0 | Further reduction |
| Minimum guarantee | 9.0 | — | Guaranteed through 2028 |

**Outputs saved to `output/phase1/`:**
- `tariff_schedule.csv` - All tariff rates with validity periods
- `tariff_flags_hourly.parquet` - Hourly high/low tariff flags
- `tariff_series_hourly.parquet` - Time-indexed tariff rates
- `tariff_report_section.html` - HTML report section

## EDA Outputs

After running `python src/phase2/run_eda.py`, outputs are saved to `output/phase2/`:

**Figures (fig01-fig16):**
- Energy patterns (time series, monthly, hourly heatmaps, seasonal)
- Heating system (COP analysis, temperature differentials, indoor/outdoor)
- Solar-heating correlation (hourly patterns, battery usage, forced grid)
- Summary statistics (monthly breakdown, HDD analysis, yearly totals)
- Heating curve analysis (schedule detection, model fit, residuals)
- Weighted temperature analysis (parameter response, sensitivity, washout periods)
- Electricity tariffs (timeline, time windows, cost implications)

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

## Weighted Temperature Analysis

Analysis of how the weighted indoor temperature (comfort objective) responds to
controllable heating parameters. Run as part of main EDA or separately:

```bash
python src/phase2/05_weighted_temperature_analysis.py
```

**Outputs:**
- `output/phase2/fig13_weighted_temp_parameters.png` - 4-panel visualization
- `output/phase2/weighted_temp_regimes.csv` - Parameter regime summary
- `output/phase2/weighted_temp_sensitivity.csv` - Parameter effects
- `output/phase2/weighted_temp_report_section.html` - HTML section for report

**Weighted Temperature Formula:**
```
T_weighted = 0.40×davis_inside + 0.30×office1 + 0.10×atelier + 0.10×studio + 0.10×simlab
```

**Key Features:**
- 48-hour (2-day) washout exclusion after each parameter regime change
- Sensitivity analysis: ΔT_weighted per unit parameter change
- Visualizes temperature response to setpoint and curve_rise changes
- Same weights used in Phase 3 thermal modeling and Phase 5 comfort objective

## Phase 3: System Modeling Outputs

After running `python src/phase3/run_phase3.py`, outputs are saved to `output/phase3/`:

**Figures (fig13-fig16):**
- fig13: Thermal model (temperature simulation, decay analysis)
- fig14: Heat pump model (COP vs temperature, capacity, buffer tank)
- fig15: Energy system (daily profiles, battery patterns, self-sufficiency)
- fig16: Tariff cost model (cost breakdown, high/low tariff, forecasting)

**Reports:**
- `phase3_modeling_report.html` - Combined modeling report
- `thermal_model_results.csv` - Per-room thermal parameters
- `heat_pump_daily_stats.csv` - Daily COP and energy statistics
- `cost_model_daily_stats.csv` - Daily cost breakdown (grid, feedin, net)
- `cost_forecast_model.json` - Cost forecasting model coefficients

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
- fig16: Strategy comparison (COP by strategy, schedule alignment, expected improvements)
- fig17: Simulation results (time series, self-sufficiency, hourly COP profiles)
- fig18: Parameter space (trade-offs, parameter summary table)

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

| Strategy | Schedule | Curve Rise | COP | vs Baseline | Goal |
|----------|----------|------------|-----|-------------|------|
| Baseline | 06:30-20:00 | 1.08 | 4.09 | — | Reference |
| Energy-Optimized | 10:00-18:00 | 0.98 | 4.39 | +0.18 | Minimize grid |
| Cost-Optimized | 11:00-21:00 | 0.95/0.85* | 4.43 | +0.22 | Minimize costs |

*Cost-Optimized uses curve_rise 0.85 when grid-dependent

**Comfort Evaluation:**
- Comfort compliance evaluated **only during occupied hours (08:00-22:00)**
- Night temperatures (22:00-08:00) are excluded from comfort objectives
- This allows energy-saving at night without penalty

**Key optimization levers:**
- Shift comfort mode to PV peak hours (10:00-17:00)
- Lower curve_rise for better COP (~0.1 COP improvement per 1°C flow temp reduction)
- Dynamic curve_rise reduction when grid-dependent (0.85-0.90)
- Tariff arbitrage: shift heating to low-tariff periods (21:00-06:00, weekends)

## Pareto Optimization (Phase 4, Step 4)

Multi-objective optimization using NSGA-II to find Pareto-optimal heating strategies.

**Commands:**
```bash
# Run Pareto optimization (default: 10 generations, auto warm-start)
python src/phase4/04_pareto_optimization.py

# More generations for deeper optimization
python src/phase4/04_pareto_optimization.py -g 50

# Start fresh (ignore existing archive)
python src/phase4/04_pareto_optimization.py --fresh -g 200

# Custom settings
python src/phase4/04_pareto_optimization.py -g 100 -p 150 -n 15 --seed 123
```

**Default behavior:**
- Auto-detects `pareto_archive.json` and uses it for warm start
- Runs 10 generations (quick refinement)
- Use `--fresh` to start from scratch

**Decision Variables (5):**
| Variable | Range | Description |
|----------|-------|-------------|
| `setpoint_comfort` | [19.0, 22.0] °C | Comfort mode target temperature |
| `setpoint_eco` | [12.0, 19.0] °C | Eco mode target (12°C = frost protection) |
| `comfort_start` | [06:00, 12:00] | Start of comfort period |
| `comfort_end` | [16:00, 22:00] | End of comfort period |
| `curve_rise` | [0.80, 1.20] | Heating curve slope (Steilheit) |

**Objectives (4, all minimized):**
1. **Mean temp deficit**: Target (20.5°C) - mean T_weighted during 08:00-22:00
2. **Min temp deficit**: Threshold (18.5°C) - min T_weighted during 08:00-22:00
3. **Grid import**: Total kWh purchased from grid
4. **Net cost**: Grid cost - feed-in revenue (CHF)

**T_weighted Adjustment Model:**
Uses Phase 2 regression coefficients to adjust historical T_weighted based on parameter changes:
- Comfort setpoint: +1.22°C per 1°C increase
- Eco setpoint: -0.09°C per 1°C increase (negligible - allows aggressive setback)
- Curve rise: +9.73°C per unit increase

**Key Finding (Jan 2026):** Phase 2 multivariate analysis revealed eco setpoint has minimal
effect on daytime comfort (-0.09°C per 1°C change). This allows aggressive eco setbacks
(down to 12°C) without compromising comfort during occupied hours.

**Outputs:**
```
output/phase4/
├── pareto_archive.json        # Full archive for warm-starting future runs
├── pareto_front.csv           # All Pareto-optimal solutions
├── selected_strategies.csv    # 10 diverse strategies selected
├── selected_strategies.json   # Machine-readable format
├── fig19_pareto_front.png     # 2D projections of Pareto front
├── fig20_strategy_comparison.png # Radar chart comparing strategies
└── pareto_report_section.html # HTML report section
```

**Workflow:**
1. First run: `python src/phase4/04_pareto_optimization.py --fresh -g 200` (full optimization)
2. Refinement: `python src/phase4/04_pareto_optimization.py` (auto warm-start, 10 generations)
3. Review 10 selected strategies in `selected_strategies.csv`
4. Manually select 3 strategies for Phase 5 intervention study

## Phase 5: Intervention Study

Randomized crossover study to test heating strategies in the field.

**Commands:**
```bash
# Estimate optimal study parameters (washout, block length)
python src/phase5/estimate_study_parameters.py

# Generate randomization schedule
python src/phase5/generate_schedule.py --start 2027-11-01 --weeks 20 --seed 42
```

**Study Design (data-driven):**
- Duration: 20 weeks (November 2027 - March 2028)
- Block length: 5 days (3-day washout + 2-day measurement)
- Conditions: 3 strategies (A=Baseline, B=Energy-Optimized, C=Cost-Optimized)
- Total blocks: 28 (~9 per strategy)
- Statistical power: >95% to detect +0.30 COP change

**Controllable Parameters:**

| Parameter | How to Change | Location |
|-----------|---------------|----------|
| Comfort start/end | Heat pump scheduler | Heat pump interface |
| Setpoint comfort/eco | Climate entity | Home Assistant |
| Curve rise (Steilheit) | Heating curve menu | Heat pump interface |

**Strategy Parameter Summary (Pareto-Optimized):**

Three strategies selected from 21 Pareto-optimal solutions for Phase 5 intervention study:

| Parameter | A (Baseline) | B (Grid-Minimal) | C (Balanced) |
|-----------|--------------|------------------|--------------|
| Comfort start | 06:30 | 11:45 | 11:45 |
| Comfort end | 20:00 | 16:00 | 16:00 |
| Setpoint comfort | 20.2°C | 22.0°C | 22.0°C |
| Setpoint eco | 18.5°C | **13.6°C** | **12.5°C** |
| Curve rise | 1.08 | 0.81 | 0.98 |
| Grid (kWh) | ~1200 | **1069** | 1104 |
| Cost (CHF) | ~320 | **279** | 290 |

**Key insight:** Lower eco setpoints (12-14°C) are optimal because:
- Eco setpoint has minimal effect on daytime comfort (-0.09°C per 1°C change)
- Shorter comfort window (4h) aligned with solar peak (11:45-16:00) maximizes PV utilization
- Aggressive eco setback saves energy during non-solar hours

**Comfort Objective (T_weighted):**

Comfort compliance is evaluated using a weighted indoor temperature:
```
T_weighted = 0.40×davis_inside + 0.30×office1 + 0.10×atelier + 0.10×studio + 0.10×simlab
```

- Evaluated during occupied hours only (08:00-22:00)
- Target: ≥95% of readings within comfort bounds (18.5°C - 22°C)
- Same weights used in Phase 3 thermal modeling (see `src/phase3/01_thermal_model.py`)
- See `docs/phase5_experimental_design.md` Section 8.4 for full definition

**Outputs:**
```
output/phase5/
├── block_schedule.csv          # Randomized block schedule
├── block_schedule.json         # Machine-readable schedule
├── experimental_protocol.html  # HTML report for study execution
├── daily_logs/                 # Daily checklist entries
├── block_summaries/            # Block summary entries
└── analysis/                   # Statistical outputs
```

**Documentation:**
- `docs/phase5_experimental_design.md` - Full experimental protocol

## Documentation

- `PRD.md` - Research design plan for heating strategy optimization
- `docs/phase3_models.md` - Detailed documentation of Phase 3 models (assumptions, equations, interpretation)
- `docs/phase5_experimental_design.md` - Phase 5 intervention study protocol

## Research Design

See `PRD.md` for the heating strategy optimization research plan.
