# Phase 5: Intervention Study

## Phase 5 Pilot: T_HK2-Targeted Parameter Exploration (Jan-Mar 2026)

T_HK2-targeted design experiment to learn the thermal response function:
```
T_indoor = f(T_HK2 history, T_outdoor history, thermal_mass)
```

**Key Insight:** The heating curve model is deterministic and well-understood:
```
T_HK2 = T_setpoint + curve_rise * (T_ref - T_outdoor)
```
What we DON'T understand is how indoor temperature depends on T_HK2 history.
Therefore, the pilot maximizes **T_HK2 spread** rather than raw parameter spread.

### Commands

```bash
# Generate design + schedule (10 blocks, starting Jan 13, 2026)
python src/phase5_pilot/run_pilot.py

# Use different reference outdoor temperature for T_HK2 calculation
python src/phase5_pilot/run_pilot.py --ref-outdoor 3

# Analyze data (RSM block-averaged analysis)
python src/phase5_pilot/run_pilot.py --analyze-rsm
python src/phase5_pilot/03_pilot_analysis.py --block 5  # RSM through block 5
```

### Design

- Type: T_HK2-targeted (optimizes for flow temperature spread)
- Blocks: 10 (70 days = 10 weeks)
- Block length: 7 days
- Period: Jan 13 - Mar 23, 2026

### T_HK2 Spread (at reference T_outdoor = 5C)

| Mode | Min T_HK2 | Max T_HK2 | Spread |
|------|-----------|-----------|--------|
| Comfort | 32.1C | 41.6C | 9.5C |
| Eco | 25.3C | 36.0C | 10.7C |

### Parameter Bounds

| Parameter | Min | Max | Goal |
|-----------|-----|-----|------|
| comfort_setpoint | 19.0C | 22.0C | Varies T_HK2 comfort |
| eco_setpoint | 14.0C | 19.0C | Varies T_HK2 eco |
| curve_rise | 0.80 | 1.20 | Varies T_HK2 slope |
| comfort_hours | 8h | 16h | Schedule (orthogonal to T_HK2) |

### Safety Constraints

- Minimum T_weighted: 17.0C (abort block if breached)
- Minimum COP: 2.0 (check heat pump if below)
- Maximum violation %: 50% (pilot allows more than Phase 5)

### Outputs

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

### Analysis (RSM Block-Averaged)

```
T_indoor = b0 + b1*T_HK2_comfort + b2*T_HK2_eco + b3*comfort_hours + b4*T_outdoor
```

**Note:** A grey-box dynamical analysis approach was also tried (`04_dynamical_analysis.py`)
but did not produce reliable predictions. The RSM approach uses block averages with washout.

---

## Phase 5: Main Intervention Study

Randomized crossover study to test heating strategies in the field.

### Commands

```bash
# Estimate optimal study parameters (washout, block length)
python src/phase5/estimate_study_parameters.py

# Generate randomization schedule
python src/phase5/generate_schedule.py --start 2027-11-01 --weeks 20 --seed 42
```

### Study Design (data-driven)

- Duration: 20 weeks (November 2027 - March 2028)
- Block length: 7 days (3-day washout + 4-day measurement) - weekly parameter changes
- Conditions: 3 strategies (A=Baseline, B=Grid-Minimal, C=Balanced)
- Total blocks: 20 (~6-7 per strategy)
- Statistical power: 97% to detect +0.30 COP change
- Washout based on tau_effort (heating response time) = 12.4h weighted avg

### Controllable Parameters

| Parameter | How to Change | Location |
|-----------|---------------|----------|
| Comfort start/end | Heat pump scheduler | Heat pump interface |
| Setpoint comfort/eco | Climate entity | Home Assistant |
| Curve rise (Steilheit) | Heating curve menu | Heat pump interface |

### Strategy Parameter Summary (Pareto-Optimized, 500 pop x 100 gen)

Three strategies selected from Pareto-optimal solutions for Phase 5 intervention study:

| Parameter | A (Baseline) | B (Grid-Minimal) | C (Balanced) |
|-----------|--------------|------------------|--------------|
| Comfort start | 06:30 | 11:30 | 11:45 |
| Comfort end | 20:00 | 16:00 | 16:00 |
| Setpoint comfort | 20.2C | 22.0C | 22.0C |
| Setpoint eco | 18.5C | **14.1C** | **14.2C** |
| Curve rise | 1.08 | **0.82** | **0.82** |
| Grid (kWh)* | - | **2007** | 2007 |
| Cost (CHF)* | - | 598 | **597** |
| Violation % | - | 2.9% | 2.9% |
| Min temp | - | 17.7C | 17.7C |

*52-day simulation period

### Key Insight

Large-scale optimization found a different trade-off than initial runs:
- Eco setpoint at 14C (vs 12C) provides buffer without affecting daytime comfort
- Lower curve_rise (0.82) achieves better COP with narrower comfort window
- Later comfort start (11:30-11:45) aligns better with solar production
- Result: 10% lower grid import (2007 vs 2235 kWh) with better comfort (2.9% violation)

### Comfort Objective (T_weighted)

The comfort objective uses weighted indoor temperature:
```
T_weighted = sum(weight_i * T_sensor_i)
```

In principle, multiple room sensors can be weighted. In practice, only
`davis_inside_temperature` (100% weight) is used because other sensors
have too much noise in the data.

### Optimization Framework

- **Three objectives**: Maximize avg temp, minimize grid import, minimize net cost
- **Soft constraint**: T_weighted < 18.5C for <=20% of daytime hours (08:00-22:00)
- **No upper temperature limit** - higher temperatures are always acceptable
- Same weights used in Phase 3 thermal modeling (see `src/phase3/01_thermal_model.py`)
- See `docs/phase5_experimental_design.md` Section 8.4 for full definition

### Outputs

```
output/phase5/
├── block_schedule.csv          # Randomized block schedule
├── block_schedule.json         # Machine-readable schedule
├── experimental_protocol.html  # HTML report for study execution
├── daily_logs/                 # Daily checklist entries
├── block_summaries/            # Block summary entries
└── analysis/                   # Statistical outputs
```

### Documentation

- `docs/phase5_experimental_design.md` - Full experimental protocol
