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
python src/run_all.py              # Run complete pipeline (skips Pareto optimization)
python src/run_all.py --phase 1    # Run Phase 1 only
python src/run_all.py --phase 2    # Run Phase 2 only
python src/run_all.py --phase 3    # Run Phase 3 only
python src/run_all.py --phase 4    # Run Phase 4 only (skips Pareto)
python src/run_all.py --phase 4 --rerun_optimization  # Phase 4 with Pareto
python src/run_all.py --step 1.2   # Run specific step
python src/run_all.py --step 4.4   # Run Pareto optimization only
python src/run_all.py --list       # List all steps
python src/run_all.py --report     # Regenerate main report only
python src/generate_main_report.py # Generate output/index.html directly

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
python src/phase4/04_pareto_optimization.py        # Phase 4, Step 4
python src/phase4/05_strategy_evaluation.py        # Phase 4, Step 5
python src/phase4/06_strategy_detailed_analysis.py # Phase 4, Step 6 (Phase 5 strategy details)
python src/phase4/07_pareto_animation.py           # Phase 4, Step 7 (GIF + MP4 animation)
python src/phase4/07_pareto_animation.py --mp4-only  # Convert existing GIFs to MP4 only

# Phase 5 Pilot: T_HK2-Targeted Parameter Exploration (Jan-Mar 2026)
python src/phase5_pilot/run_pilot.py               # Generate T_HK2 design + schedule
python src/phase5_pilot/run_pilot.py --ref-outdoor 3   # Use different reference temp
python src/phase5_pilot/run_pilot.py --design-only # Generate design only
python src/phase5_pilot/run_pilot.py --schedule-only # Generate schedule from existing design
python src/phase5_pilot/run_pilot.py --analyze-rsm # RSM block-averaged analysis
python src/phase5_pilot/03_pilot_analysis.py       # RSM analysis (run after each block)
python src/phase5_pilot/03_pilot_analysis.py --block 5  # Analyze through block 5 only
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
│   ├── 03_heating_curve_analysis.py  # Heating curve model + schedule detection
│   └── 04_tariff_analysis.py         # Electricity tariff analysis (HKN-only)
├── phase3/              # System Modeling
│   ├── run_phase3.py                 # Wrapper: runs all models + HTML report
│   ├── 01_thermal_model.py           # Building thermal characteristics
│   ├── 01b_greybox_thermal_model.py  # Grey-box state-space model (abandoned)
│   ├── 02_heat_pump_model.py         # COP relationships, buffer tank
│   ├── 03_energy_system_model.py     # PV patterns, battery, self-sufficiency
│   ├── 04_tariff_cost_model.py       # Electricity cost analysis + forecasting
│   ├── 05_weekly_decomposition.py    # Weekly thermal model decomposition report
│   └── 06_extended_decomposition.py  # Extended decomposition with energy/COP panels
├── phase4/              # Optimization Strategy Development
│   ├── run_optimization.py           # Wrapper: runs all steps + HTML report
│   ├── run_pareto.py                 # CLI wrapper for Pareto optimization
│   ├── 01_rule_based_strategies.py   # Strategy definitions and rules
│   ├── 02_strategy_simulation.py     # Validate strategies on historical data
│   ├── 03_parameter_sets.py          # Generate Phase 5 parameter sets
│   ├── 04_pareto_optimization.py     # NSGA-II multi-objective optimization
│   ├── 05_strategy_evaluation.py     # Comfort violation analysis + winter predictions
│   ├── 06_strategy_detailed_analysis.py  # Detailed Phase 5 strategy visualizations
│   └── 07_pareto_animation.py        # Pareto evolution GIF + MP4 for PowerPoint
├── phase5/              # Intervention Study
│   ├── estimate_study_parameters.py  # Data-driven washout/block estimation
│   └── generate_schedule.py          # Randomization schedule generator
├── phase5_pilot/        # Pilot Experiment (Jan-Mar 2026)
│   ├── run_pilot.py                  # Main runner: design + schedule + analysis
│   ├── 01_generate_thk2_design.py    # T_HK2-targeted design generation
│   ├── 02_generate_pilot_schedule.py # Dated block schedule
│   ├── 03_pilot_analysis.py          # RSM block-averaged analysis
│   └── 04_dynamical_analysis.py      # Grey-box dynamical analysis (abandoned)
├── shared/              # Shared modules used across phases
│   ├── __init__.py
│   ├── report_style.py               # CSS and colors for HTML reports
│   ├── thermal_simulator.py          # Grey-box thermal model (abandoned)
│   └── energy_system.py              # Battery-aware energy system simulation
└── xtra/                # Standalone analyses (not part of Phase 5 study)
    ├── battery_degradation/          # Battery efficiency degradation study
    │   └── battery_degradation.py
    ├── battery_savings/              # Battery cost savings analysis
    │   └── battery_savings.py
    └── system_diagram.py             # Semi-abstract system diagram for presentations
```

## Output Directory Structure

```
output/
├── index.html    # Main report with TOC linking to all phase reports
├── phase1/       # Preprocessing outputs (parquet files, reports)
├── phase2/       # EDA outputs (figures, HTML reports)
├── phase3/       # System modeling outputs (figures, model results)
│   └── weekly_decomposition/  # Per-week thermal model analysis (incl. extended)
├── phase4/       # Optimization outputs (strategies, predictions)
├── phase5/       # Intervention study outputs (schedules, logs, analysis)
├── phase5_pilot/ # Pilot experiment outputs (Jan-Mar 2026)
└── xtra/         # Standalone analysis outputs
    ├── battery_degradation/  # Battery efficiency study
    ├── battery_savings/      # Battery cost savings analysis
    └── system_diagram.*      # System diagram (PNG, PDF, SVG)
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
- `phase1_report.html` - Comprehensive preprocessing documentation
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
| Period | All-in Average | Notes |
|--------|----------------|-------|
| 2023 | 31.3 | Peak energy crisis year |
| 2024 | 32.8 | +4.8% vs 2023 |
| 2025 | 32.6 | -1% vs 2024 |

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
- `phase2_report.html` - Comprehensive EDA documentation with all figures

## Battery Degradation Analysis (xtra)

Standalone analysis investigating whether the Feb-Mar 2025 deep-discharge event
affected battery efficiency. Not part of the Phase 5 study pipeline.

```bash
python src/xtra/battery_degradation/battery_degradation.py
```

**Outputs** (in `output/xtra/battery_degradation/`):
- `battery_degradation_analysis.png` - 4-panel visualization
- `battery_degradation_report.html` - Detailed report with methods & results

**Analysis includes:**
- OLS regression with time trend and post-event indicator
- Welch's t-test as robustness check
- Key finding: Statistically significant efficiency drop of ~10.8 percentage points (p<0.001)

## Battery Cost Savings Analysis (xtra)

Analyzes how much the battery saves compared to a hypothetical system without a battery.
Considers time-varying tariffs and 30% tax on feed-in income.

```bash
python src/xtra/battery_savings/battery_savings.py
```

**Outputs** (in `output/xtra/battery_savings/`):
- `battery_savings_daily.csv` - Daily cost savings data
- `battery_savings_analysis.png` - Cumulative savings visualization
- `battery_savings_report.html` - Summary report with methodology

**Methodology:**
```
Savings = battery_discharge × purchase_rate - battery_charge × feedin_rate × 0.70
```
- Without battery: charging energy would be fed to grid, discharging energy would be imported
- The 0.70 factor accounts for 30% income tax on feed-in revenue
- Time-varying tariffs applied at 15-minute resolution

**Key finding:** Battery saved CHF 1,143 over ~2.8 years (CHF 1.10/day average)

## System Diagram (xtra)

Semi-abstract diagram of the energy system for PowerPoint presentations.

```bash
python src/xtra/system_diagram.py
```

**Outputs** (in `output/xtra/`):
- `system_diagram.png` - PNG format (300 DPI)
- `system_diagram.pdf` - PDF format (vector)
- `system_diagram.svg` - SVG format (vector)

**Components shown:**
- Sun → PV panels (electricity flow, black)
- PV → Battery, Grid, House (electricity flows, black)
- Grid ↔ House (bidirectional electricity, black)
- Heat pump with outdoor temperature input
- Buffer tank with heat distribution to house (heat flows, red)
- Temperature sensors (T_outdoor, T_room, T_HK2)

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
- `output/phase2/heating_curve_params.json` - Model parameters for Phase 3/4 import
- `output/phase2/heating_curve_report_section.html` - HTML section for report

**Model:**
```
T_HK2 = T_setpoint + curve_rise × (T_ref - T_outdoor)
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

**Heating Curve Parameter Integration:**

The Phase 2 heating curve model exports parameters to JSON for use in Phase 3 and Phase 4:

```json
// output/phase2/heating_curve_params.json
{
  "t_ref_comfort": 21.32,   // Reference temperature for comfort mode
  "t_ref_eco": 19.16,       // Reference temperature for eco mode
  "r_squared": 0.931,       // Model fit (all data)
  "normal_r_squared": 0.963 // Model fit (excluding anomalies)
}
```

**Why this matters:** The optimization only makes sense if controllable parameters
(setpoint, curve_rise) actually affect T_HK2. The parametric model ensures:
- Changing setpoint shifts the heating curve up/down
- Changing curve_rise affects how aggressively target temp responds to cold
- Both propagate through: T_HK2 → COP → energy consumption

**Scripts that load heating curve params:**
- `src/phase3/01_thermal_model.py` - Displays for comparison with simple model
- `src/phase4/01_rule_based_strategies.py` - Strategy definitions
- `src/phase4/02_strategy_simulation.py` - Simulation
- `src/phase4/04_pareto_optimization.py` - NSGA-II optimization
- `src/phase4/05_strategy_evaluation.py` - Violation analysis
- `src/phase4/06_strategy_detailed_analysis.py` - Detailed analysis

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

**Temperature Formula:**
```
T_weighted = Σ(weight_i × T_sensor_i)
```

The system supports weighted averaging of multiple room sensors (office1, atelier,
studio, simlab, davis_inside). However, in practice only `davis_inside_temperature`
is used (100% weight) because other room sensors have excessive measurement noise
that degrades model performance. Single-sensor model achieves R²=0.683 vs R²=0.569
with multiple sensors.

**Key Features:**
- 48-hour (2-day) washout exclusion after each parameter regime change
- Sensitivity analysis: ΔT_weighted per unit parameter change
- Visualizes temperature response to setpoint and curve_rise changes
- Same sensor used in Phase 3 thermal modeling and Phase 5 comfort objective

## Phase 3: System Modeling Outputs

After running `python src/phase3/run_phase3.py`, outputs are saved to `output/phase3/`:

**Figures (fig18-fig22):**
- fig18: Thermal model (temperature simulation, decay analysis)
- fig19: Heat pump model (COP vs temperature, capacity, buffer tank)
- fig20: Energy system (daily profiles, battery patterns, self-sufficiency)
- fig21: Tariff cost model (cost breakdown, high/low tariff, forecasting)
- fig22: Extended decomposition (10 panels: thermal model + energy + COP)

**Reports:**
- `phase3_report.html` - Combined modeling report
- `thermal_model_results.csv` - Per-room thermal parameters (transfer function)
- `heat_pump_daily_stats.csv` - Daily COP and energy statistics
- `cost_model_daily_stats.csv` - Daily cost breakdown (grid, feedin, net)
- `cost_forecast_model.json` - Cost forecasting model coefficients

**Thermal Model Sensor:**
- davis_inside_temperature: 100% (single sensor, least noise)

**Outdoor Temperature Sensor:**
- Column: `stiebel_eltron_isg_outdoor_temperature`
- Source: Heat pump's built-in outdoor sensor (mounted near the house)
- Note: This is NOT true ambient outdoor temperature, but rather what the heat pump
  uses internally for heating curve calculations. This is intentional - the model
  should use the same temperature input as the heat pump to accurately predict
  flow temperatures and COP.
- Range (Nov-Dec 2025): -2.2 to 15.0°C, mean 7.3°C
- **Sensor comparison (COP prediction):** Stiebel R²=0.793 vs Davis weather station R²=0.718.
  The heat pump sensor is 10% better at predicting COP, likely because it measures
  actual air temperature at the evaporator location. See fig05 for visual comparison.

**Key Model Results:**
- Building time constant: ~14-33 hours (varies by sensor)
- davis_inside: tau=24h (primary comfort sensor)
- COP model (R²=0.94): `COP = 5.93 + 0.13×T_outdoor - 0.08×T_HK2`
- Current self-sufficiency: 58%, potential with optimization: 85%

## Energy System Simulation Module

Shared module (`src/shared/energy_system.py`) providing intra-day energy system simulation
with battery constraints. Used by Phase 3 extended decomposition and Phase 4 optimization.

**Battery Model:**
```python
BATTERY_PARAMS = {
    'capacity_kwh': 13.8,        # Total capacity (usable: 13.8 × 0.8 = 11.04 kWh)
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

**Battery Model Improvements (Jan 2026):**
Analysis comparing model vs observed battery behavior revealed:
- **Total capacity**: 13.8 kWh with 20% min SoC → usable capacity 11.04 kWh
- **Post-degradation efficiency**: 77% round-trip (down from 84% after Feb-Mar 2025 event)
- **Time-of-use strategy**: Discharge concentrated 15:00-22:00, minimal overnight
- **Result**: Model grid import now within +0.8% of observed (was -2.1%)

**COP Model:**
```
COP = 5.93 + 0.13×T_outdoor - 0.08×T_HK2
```
Clipped to [1.5, 8.0] for physical limits.

**Heating Curve Model:**
```
T_HK2 = setpoint + curve_rise × (T_ref - T_outdoor)
```
Where T_ref = 21.32°C (comfort) or 19.18°C (eco).

**Key Functions:**
| Function | Description |
|----------|-------------|
| `simulate_battery_soc()` | SoC tracking with capacity constraints |
| `predict_cop()` | Intra-day COP from T_outdoor and T_HK2 |
| `predict_t_hk2_variable_setpoint()` | Heating curve with comfort/eco modes |
| `is_high_tariff()` | Tariff period detection |
| `calculate_electricity_cost()` | Tariff-aware cost calculation |
| `simulate_energy_system()` | Full system simulation |

**Energy Flow Logic:**
1. Net = PV - consumption
2. If Net > 0: charge battery (up to capacity), excess to grid
3. If Net < 0: discharge battery (if available), deficit from grid

**Used by:**
- `src/phase3/06_extended_decomposition.py` - Panels 5-8, 10
- `src/phase4/04_pareto_optimization.py` - Strategy evaluation
- `src/phase4/02_strategy_simulation.py` - Strategy comparison

## Grey-Box Thermal Model (Abandoned)

Physics-based two-state discrete-time model for room temperature prediction.
**Status:** Tried but did not work well - poor predictive accuracy on validation data.

**Model Formulation (Δt = 15 min):**
```
T_buffer[k+1] = T_buffer[k] + (dt/tau_buf) × [(T_HK2[k] - T_buffer[k]) - r_emit × (T_buffer[k] - T_room[k])]
T_room[k+1] = T_room[k] + (dt/tau_room) × [r_heat × (T_buffer[k] - T_room[k]) - (T_room[k] - T_out[k])] + k_solar × PV[k]
```

Parameters: `tau_buf` (buffer time constant), `tau_room` (building time constant),
`r_emit`/`r_heat` (coupling ratios), `k_solar` (solar gain), `c_offset` (bias).

Script: `src/phase3/01b_greybox_thermal_model.py`

## Transfer Function Thermal Model

Linear transfer function model using low-pass filtered inputs:
```
T_room = offset + g_outdoor×LPF(T_outdoor, 24h) + g_effort×LPF(Effort, 2h) + g_pv×LPF(PV, 12h)
```

Where `Effort = T_HK2 - baseline_curve` (deviation from heating curve).

**Key Parameters:**
- `g_effort = 0.208` (STABLE: coefficient of variation = 9%)
- `g_outdoor = 0.442` (UNSTABLE: CV = 95%)
- Model R² = 0.68 (captures 68% of variance)

**Causal Coefficients (for Phase 4 optimization):**

The transfer function provides causal estimates of how heating parameters affect room temperature:

| Parameter | Phase 2 Regression | Phase 3 Causal | Ratio |
|-----------|-------------------|----------------|-------|
| comfort_setpoint | +1.22°C/°C | **+0.21°C/°C** | 5.9x |
| curve_rise | +9.73°C/unit | **+2.92°C/unit** | 3.3x |

**Important:** Phase 2 regression coefficients overestimate effects by 3-6x because they
capture associations, not causal effects. Phase 4 now uses the causal coefficients from
`output/phase3/causal_coefficients.json`.

**Causal Chain:**
```
setpoint +1°C → T_HK2 +1°C → Effort +1°C → LPF(Effort) +1°C → T_room +0.21°C
```

Scripts:
- `src/phase3/01_thermal_model.py` - Main thermal model
- `src/phase3/01e_adaptive_thermal_model.py` - Time-varying parameters (RLS)
- `src/phase3/01f_transfer_function_integration.py` - Causal coefficient derivation

## Phase 4: Optimization Strategy Outputs

After running `python src/phase4/run_optimization.py`, outputs are saved to `output/phase4/`:

**Figures (fig22-fig31):**
- fig22: Strategy comparison (COP by strategy, schedule alignment, expected improvements)
- fig23: Simulation results (time series, self-sufficiency, hourly COP profiles)
- fig24: Parameter space (trade-offs, parameter summary table)
- fig25: Pareto front (2D projections of Pareto-optimal solutions)
- fig26: Pareto strategy comparison (radar chart comparing strategies)
- fig27: Pareto evolution (optimization history animation frame)
- fig28: Strategy temperature predictions (winter 2026/2027, violation analysis)
- fig29: Detailed time series (T_weighted, outdoor, solar, grid by strategy)
- fig30: Hourly patterns (temperature heatmaps, PV/grid profiles, comfort windows)
- fig31: Energy patterns (daily balance, self-sufficiency, temperature distributions)

**Reports:**
- `phase4_report.html` - Combined optimization report
- `optimization_strategies.csv` - Strategy definitions and rules
- `strategy_comparison.csv` - Simulated metrics by strategy
- `simulation_daily_metrics.csv` - Daily simulation results

**Phase 5 Preparation:**
- `phase5_parameter_sets.json` - Exact parameter values for intervention study
- `phase5_predictions.json` - Testable predictions with confidence intervals
- `phase5_implementation_checklist.md` - Protocol for randomized study

**Strategy Evaluation (Step 5):**
- `strategy_violation_analysis.csv` - Comfort violation stats per strategy
- `strategy_evaluation_report.html` - HTML report with violation analysis
- `fig28_strategy_temperature_predictions.png` - Winter 2026/2027 predictions

**Detailed Strategy Analysis (Step 6):**
- `strategy_detailed_stats.csv` - Comprehensive statistics for Phase 5 strategies
- `strategy_detailed_report.html` - HTML report with time series and energy analysis
- `fig29_strategy_detailed_timeseries.png` - Full-period temperature and energy time series
- `fig30_strategy_hourly_patterns.png` - Hourly patterns, heatmaps, and comfort windows
- `fig31_strategy_energy_patterns.png` - Energy balance, self-sufficiency, temperature distributions

**Heating Curve Model (from Phase 2):**
```
T_HK2 = T_setpoint + curve_rise × (T_ref - T_outdoor)
```
Where:
- T_HK2 = target flow temperature (heating curve setpoint)
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
- Comfort constraint evaluated **only during occupied hours (08:00-22:00)**
- Constraint: T_weighted < 18.5°C for ≤20% of daytime hours (soft penalty)
- No upper temperature limit - higher is always better
- Night temperatures (22:00-08:00) are excluded from comfort objectives

**Key optimization levers:**
- Shift comfort mode to PV peak hours (10:00-17:00)
- Lower curve_rise for better COP (~0.1 COP improvement per 1°C flow temp reduction)
- Dynamic curve_rise reduction when grid-dependent (0.85-0.90)
- Tariff arbitrage: shift heating to low-tariff periods (21:00-06:00, weekends)

## Pareto Optimization (Phase 4, Step 4)

Multi-objective optimization using NSGA-II to find Pareto-optimal heating strategies.

**Commands:**
```bash
# Run Pareto optimization (default: 200 generations, auto warm-start)
python src/phase4/04_pareto_optimization.py

# Quick refinement (fewer generations)
python src/phase4/04_pareto_optimization.py -g 50

# Start fresh (ignore existing archive)
python src/phase4/04_pareto_optimization.py --fresh

# Custom settings
python src/phase4/04_pareto_optimization.py -g 100 -p 150 -n 15 --seed 123
```

**Default behavior:**
- Auto-detects `pareto_archive.json` and uses it for warm start
- Runs 200 generations (full optimization)
- Use `--fresh` to start from scratch
- Uses ε-dominance to filter meaningfully different solutions (default: enabled)

**ε-Dominance Filtering:**

Standard Pareto optimization produces many solutions that differ by negligible amounts
(e.g., 0.01°C temperature difference). ε-dominance keeps only solutions that are
meaningfully different by snapping objectives to an epsilon grid before dominance comparison.

| Objective | Epsilon | Description |
|-----------|---------|-------------|
| Temperature | **0.1°C** | Comfort differences below this are imperceptible |
| Grid import | **100 kWh** | ~5% of typical total range |
| Cost | **10 CHF** | Fine-grained cost differences |

Effect: Reduces hundreds of solutions → ~3-5 meaningfully different solutions.

```bash
# Disable ε-dominance (keep all Pareto solutions)
python src/phase4/04_pareto_optimization.py --no-epsilon

# Custom epsilon values (more conservative filtering)
python src/phase4/04_pareto_optimization.py --eps-temp 0.3 --eps-grid 30 --eps-cost 10
```

**Decision Variables (5):**
| Variable | Range | Description |
|----------|-------|-------------|
| `setpoint_comfort` | [19.0, 22.0] °C | Comfort mode target temperature |
| `setpoint_eco` | [12.0, 19.0] °C | Eco mode target (12°C = frost protection) |
| `comfort_start` | [06:00, 12:00] | Start of comfort period |
| `comfort_end` | [16:00, 22:00] | End of comfort period |
| `curve_rise` | [0.80, 1.20] | Heating curve slope (Steilheit) |

**Objectives (3, all minimized in NSGA-II):**
1. **Negative mean temperature**: `-mean(T_weighted)` during 08:00-22:00 (minimizing = maximize avg temp)
2. **Grid import**: Total kWh purchased from grid
3. **Net cost**: Grid cost - feed-in revenue (CHF)

**Comfort Constraint Parameters:**

| Parameter | Value | Code Constant | Description |
|-----------|-------|---------------|-------------|
| `COMFORT_THRESHOLD` | 18.5°C | `04_pareto_optimization.py:260` | Minimum acceptable T_weighted |
| `VIOLATION_LIMIT` | **5%** | `04_pareto_optimization.py:348` | Max allowed daytime hours below threshold |
| `OCCUPIED_START` | 08:00 | `04_pareto_optimization.py:98` | Start of comfort evaluation window |
| `OCCUPIED_END` | 22:00 | `04_pareto_optimization.py:99` | End of comfort evaluation window |

**Constraint Mechanism (soft penalty):**
The optimizer uses NSGA-II's constraint-handling approach:
- Constraint function: `g = violation_pct - 0.05`
- If `g ≤ 0`: Solution is **feasible** (≤5% of daytime below 18.5°C)
- If `g > 0`: Solution is **infeasible** but NOT excluded
- Feasible solutions always dominate infeasible ones in ranking
- Among infeasible solutions, smaller violation is preferred
- No explicit penalty coefficient; constraint satisfaction is binary for dominance

**Note:** The 5% limit was tightened from 20% in Jan 2026 after evaluation showed
energy-optimized strategies had 15-19% violation and minimum temps of 16.7°C.

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
├── pareto_archive.json        # Full archive with optimization history
├── pareto_front.csv           # All Pareto-optimal solutions
├── selected_strategies.csv    # 10 diverse strategies selected
├── selected_strategies.json   # Machine-readable format
├── fig25_pareto_front.png     # 2D projections of Pareto front
├── fig26_pareto_strategy_comparison.png # Radar chart comparing strategies
├── fig27_pareto_evolution.png # Pareto front evolution frame
├── pareto_evolution.gif       # 2D animated Pareto evolution
├── pareto_evolution.mp4       # 2D animation for PowerPoint
├── pareto_evolution_3d.gif    # 3D animated Pareto evolution
├── pareto_evolution_3d.mp4    # 3D animation for PowerPoint
└── pareto_report_section.html # HTML report section
```

**Optimization History (in pareto_archive.json):**
The archive includes full optimization history for visualization:
- `optimization_history.generations`: Per-generation snapshots with population composition
- `optimization_history.all_solutions`: All unique parameter sets evaluated
- Each solution tracks: `first_gen`, `pareto_generations` (list of generations where it was on the Pareto front)
- Enables animated visualization of Pareto front evolution

**Workflow:**
1. First run: `python src/phase4/04_pareto_optimization.py --fresh` (full optimization, 200 generations)
2. Refinement: `python src/phase4/04_pareto_optimization.py -g 50` (auto warm-start, quick refinement)
3. **Evaluate strategies**: `python src/phase4/05_strategy_evaluation.py`
4. Review violation analysis in `strategy_violation_analysis.csv`
5. Manually select 3 strategies for Phase 5 intervention study

## Strategy Evaluation (Phase 4, Step 5)

Evaluates selected strategies for comfort violations and generates winter predictions.

```bash
python src/phase4/05_strategy_evaluation.py
```

**Evaluation Results (Jan 2026, 500 pop × 100 gen):**

| Strategy | Violation % | Cold Hours | Min Temp | Mean Temp | Status |
|----------|-------------|------------|----------|-----------|--------|
| Grid-Minimal | 2.9% | 31h | 17.7°C | 19.6°C | ✓ Pass |
| Balanced | 2.9% | 31h | 17.7°C | 19.6°C | ✓ Pass |
| Cost-Minimal | 2.9% | 31h | 17.7°C | 19.6°C | ✓ Pass |
| Comfort-First | 0.0% | 0h | 21.5°C | 23.4°C | ✓ Comfortable |

**Key improvements from large-scale optimization:**
- Violation reduced: 4.5% → 2.9% (well under 5% limit)
- Cold hours reduced: 40h → 31h
- Min temps improved: 17.3°C → 17.7°C
- Optimizer found lower curve_rise: 0.89-0.90 → 0.82 with higher eco setpoint (14°C)

**Outputs:**
```
output/phase4/
├── fig28_strategy_temperature_predictions.png  # Winter 2026/2027 predictions
├── strategy_violation_analysis.csv             # Detailed violation stats
└── strategy_evaluation_report.html             # HTML report section
```

## Phase 5 Pilot: T_HK2-Targeted Parameter Exploration (Jan-Mar 2026)

T_HK2-targeted design experiment to learn the thermal response function:
```
T_indoor = f(T_HK2 history, T_outdoor history, thermal_mass)
```

**Key Insight:** The heating curve model is deterministic and well-understood:
```
T_HK2 = T_setpoint + curve_rise × (T_ref - T_outdoor)
```
What we DON'T understand is how indoor temperature depends on T_HK2 history.
Therefore, the pilot maximizes **T_HK2 spread** rather than raw parameter spread.

**Commands:**
```bash
# Generate design + schedule (10 blocks, starting Jan 13, 2026)
python src/phase5_pilot/run_pilot.py

# Use different reference outdoor temperature for T_HK2 calculation
python src/phase5_pilot/run_pilot.py --ref-outdoor 3

# Analyze data (RSM block-averaged analysis)
python src/phase5_pilot/run_pilot.py --analyze-rsm
python src/phase5_pilot/03_pilot_analysis.py --block 5  # RSM through block 5
```

**Design:**
- Type: T_HK2-targeted (optimizes for flow temperature spread)
- Blocks: 10 (70 days = 10 weeks)
- Block length: 7 days
- Period: Jan 13 - Mar 23, 2026

**T_HK2 Spread (at reference T_outdoor = 5°C):**

| Mode | Min T_HK2 | Max T_HK2 | Spread |
|------|-----------|-----------|--------|
| Comfort | 32.1°C | 41.6°C | 9.5°C |
| Eco | 25.3°C | 36.0°C | 10.7°C |

**Parameter Bounds:**

| Parameter | Min | Max | Goal |
|-----------|-----|-----|------|
| comfort_setpoint | 19.0°C | 22.0°C | Varies T_HK2 comfort |
| eco_setpoint | 14.0°C | 19.0°C | Varies T_HK2 eco |
| curve_rise | 0.80 | 1.20 | Varies T_HK2 slope |
| comfort_hours | 8h | 16h | Schedule (orthogonal to T_HK2) |

**Safety Constraints:**
- Minimum T_weighted: 17.0°C (abort block if breached)
- Minimum COP: 2.0 (check heat pump if below)
- Maximum violation %: 50% (pilot allows more than Phase 5)

**Outputs:**
```
output/phase5_pilot/
├── thk2_design.csv             # T_HK2-targeted design matrix
├── thk2_design.json            # Machine-readable design with T_HK2 values
├── pilot_schedule.csv          # Dated block schedule
├── pilot_schedule.json         # Machine-readable schedule
├── pilot_protocol.html         # Human-readable protocol with T_HK2 values
├── pilot_analysis_results.csv  # Block-level metrics (RSM)
├── pilot_model_coefficients.json # T_HK2-based RSM model results
└── pilot_analysis_report.html  # RSM analysis report
```

**Analysis (RSM Block-Averaged):**
```
T_indoor = b0 + b1×T_HK2_comfort + b2×T_HK2_eco + b3×comfort_hours + b4×T_outdoor
```

**Note:** A grey-box dynamical analysis approach was also tried (`04_dynamical_analysis.py`)
but did not produce reliable predictions. The RSM approach uses block averages with washout.

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
- Block length: 7 days (3-day washout + 4-day measurement) — weekly parameter changes
- Conditions: 3 strategies (A=Baseline, B=Grid-Minimal, C=Balanced)
- Total blocks: 20 (~6-7 per strategy)
- Statistical power: 97% to detect +0.30 COP change
- Washout based on τ_effort (heating response time) = 12.4h weighted avg

**Controllable Parameters:**

| Parameter | How to Change | Location |
|-----------|---------------|----------|
| Comfort start/end | Heat pump scheduler | Heat pump interface |
| Setpoint comfort/eco | Climate entity | Home Assistant |
| Curve rise (Steilheit) | Heating curve menu | Heat pump interface |

**Strategy Parameter Summary (Pareto-Optimized, 500 pop × 100 gen):**

Three strategies selected from Pareto-optimal solutions for Phase 5 intervention study:

| Parameter | A (Baseline) | B (Grid-Minimal) | C (Balanced) |
|-----------|--------------|------------------|--------------|
| Comfort start | 06:30 | 11:30 | 11:45 |
| Comfort end | 20:00 | 16:00 | 16:00 |
| Setpoint comfort | 20.2°C | 22.0°C | 22.0°C |
| Setpoint eco | 18.5°C | **14.1°C** | **14.2°C** |
| Curve rise | 1.08 | **0.82** | **0.82** |
| Grid (kWh)* | — | **2007** | 2007 |
| Cost (CHF)* | — | 598 | **597** |
| Violation % | — | 2.9% | 2.9% |
| Min temp | — | 17.7°C | 17.7°C |

*52-day simulation period

**Key insight:** Large-scale optimization found a different trade-off than initial runs:
- Eco setpoint at 14°C (vs 12°C) provides buffer without affecting daytime comfort
- Lower curve_rise (0.82) achieves better COP with narrower comfort window
- Later comfort start (11:30-11:45) aligns better with solar production
- Result: 10% lower grid import (2007 vs 2235 kWh) with better comfort (2.9% violation)

**Comfort Objective (T_weighted):**

The comfort objective uses weighted indoor temperature:
```
T_weighted = Σ(weight_i × T_sensor_i)
```

In principle, multiple room sensors can be weighted. In practice, only
`davis_inside_temperature` (100% weight) is used because other sensors
have too much noise in the data.

**Optimization framework:**
- **Three objectives**: Maximize avg temp, minimize grid import, minimize net cost
- **Soft constraint**: T_weighted < 18.5°C for ≤20% of daytime hours (08:00-22:00)
- **No upper temperature limit** - higher temperatures are always acceptable
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
