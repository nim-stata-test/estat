# Phase 2: Exploratory Data Analysis

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

## Heating Curve Analysis

Analysis of how target flow temperature depends on controllable parameters.

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
T_HK2 = T_setpoint + curve_rise * (T_ref - T_outdoor)
```

**Key features:**
- Detects comfort/eco schedule from step changes in target temperature
- **Detects setpoint regime changes** (when comfort/eco temperatures change)
- Identifies anomalous periods (eco >= comfort, e.g., eco=30C disabling schedule)
- Estimates T_ref from normal operation only (excludes anomalies)
- R2 = 0.98, RMSE = 0.57C for normal periods (0.35C comfort, 0.77C eco)

**Schedule regimes (timing):**
- 2025-10-30 to 2025-12-26: Comfort 06:30 - 20:00
- 2025-12-27 to 2026-01-03: Comfort 06:30 - 21:30

**Setpoint regimes (notable changes):**
- 2025-10-28 to 2025-12-04: comfort ~20.2-20.5C, eco ~18.0-18.7C
- 2025-12-04 to 2025-12-07: eco=30C (anomalous - effectively 24h comfort)
- 2025-12-14+: comfort rising to 21C

### Heating Curve Parameter Integration

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
- Both propagate through: T_HK2 -> COP -> energy consumption

**Scripts that load heating curve params:**
- `src/phase3/01_thermal_model.py` - Displays for comparison with simple model
- `src/phase4/01_rule_based_strategies.py` - Strategy definitions
- `src/phase4/02_strategy_simulation.py` - Simulation
- `src/phase4/04_pareto_optimization.py` - NSGA-II optimization
- `src/phase4/05_strategy_evaluation.py` - Violation analysis
- `src/phase4/06_strategy_detailed_analysis.py` - Detailed analysis

## Weighted Temperature Analysis

Analysis of how the weighted indoor temperature (comfort objective) responds to
controllable heating parameters.

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
T_weighted = sum(weight_i * T_sensor_i)
```

The system supports weighted averaging of multiple room sensors (office1, atelier,
studio, simlab, davis_inside). However, in practice only `davis_inside_temperature`
is used (100% weight) because other room sensors have excessive measurement noise
that degrades model performance. Single-sensor model achieves R2=0.683 vs R2=0.569
with multiple sensors.

**Key Features:**
- 48-hour (2-day) washout exclusion after each parameter regime change
- Sensitivity analysis: delta_T_weighted per unit parameter change
- Visualizes temperature response to setpoint and curve_rise changes
- Same sensor used in Phase 3 thermal modeling and Phase 5 comfort objective
