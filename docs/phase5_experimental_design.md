# Phase 5: Experimental Design

## Randomized Crossover Intervention Study for Heating Strategy Optimization

---

## 1. Study Overview

### 1.1 Objective
Evaluate three heating control strategies in a real-world setting to determine which optimizes the trade-off between comfort, energy efficiency, and cost.

### 1.2 Study Design
| Aspect | Value |
|--------|-------|
| **Type** | Randomized crossover intervention study |
| **Duration** | 20 weeks (November 2027 - March 2028) |
| **Block length** | 5 days per condition |
| **Washout period** | 3 days (excluded from analysis) |
| **Measurement period** | 2 days (used for analysis) |
| **Conditions** | 3 strategies |
| **Total blocks** | 28 blocks (~9 per strategy) |
| **Statistical power** | >95% to detect +0.30 COP change |
| **Randomization** | Latin square with weather stratification |

### 1.3 Data-Driven Parameter Estimates

These parameters were estimated from 69 days of historical data using `src/phase5/estimate_study_parameters.py`:

| Parameter | Value | Basis |
|-----------|-------|-------|
| **Thermal time constant** | 19.3 hours | Weighted average across sensors |
| **Washout period** | 58 hours (3 days) | 3 tau = 95% equilibrium |
| **COP residual std** | 0.21 | After HDD adjustment (R²=0.93) |
| **Minimum detectable effect** | +0.25 COP | 80% power threshold |

### 1.4 Primary Research Questions
1. Which strategy achieves the best COP while maintaining comfort?
2. How much grid consumption can be reduced by shifting heating to solar hours?
3. What is the cost savings potential of tariff-aware scheduling?
4. Are there strategy × weather interactions that inform adaptive control?

---

## 2. Controllable Parameters

### 2.1 Parameter Summary

| Parameter | Symbol | Range | How to Change |
|-----------|--------|-------|---------------|
| Comfort start time | `t_comfort_start` | 06:00 - 12:00 | Heat pump scheduler |
| Comfort end time | `t_comfort_end` | 16:00 - 22:00 | Heat pump scheduler |
| Setpoint (comfort) | `T_setpoint_comfort` | 19 - 22°C | Home Assistant climate entity |
| Setpoint (eco) | `T_setpoint_eco` | 16 - 19°C | Home Assistant climate entity |
| Curve rise (Steilheit) | `curve_rise` | 0.80 - 1.20 | Heat pump interface menu |

### 2.2 Parameter Change Locations

#### Heat Pump Interface (Stiebel Eltron ISG)
- **Schedule timing**: Menu → Time Programs → Heating → Set comfort period start/end
- **Curve rise**: Menu → Heating → Heating Curve → Steilheit (slope)

#### Home Assistant
- **Setpoints**: Climate entity → Set target temperature
  - Comfort mode: Set during comfort hours
  - Eco mode: Set during eco hours
- **Automation**: Optional scripts for scheduled setpoint changes

### 2.3 Derived Parameters

The heating curve model determines flow temperature:
```
T_flow = T_setpoint + curve_rise × (T_ref - T_outdoor)
```

Where:
- `T_ref` = 21.32°C (comfort mode) or 19.18°C (eco mode)
- Lower `curve_rise` → lower flow temp → higher COP (~+0.10 COP per 1°C reduction)

---

## 3. Strategy Definitions

### 3.1 Strategy A: Baseline (Control)
Current system settings serving as reference.

| Parameter | Value | Notes |
|-----------|-------|-------|
| Comfort start | **06:30** | Early morning pre-heat |
| Comfort end | **20:00** | Evening cooldown |
| Setpoint comfort | **20.2°C** | Current winter setting |
| Setpoint eco | **18.5°C** | Night/away setting |
| Curve rise | **1.08** | Factory default |

**Expected outcomes**: COP 4.09, Self-sufficiency 40%, Comfort 85%

---

### 3.2 Strategy B: Energy-Optimized
Minimize grid electricity while maintaining comfort.

| Parameter | Value | Notes |
|-----------|-------|-------|
| Comfort start | **10:00** | Aligned with solar production |
| Comfort end | **18:00** | Before evening peak demand |
| Setpoint comfort | **20.0°C** | Slightly reduced |
| Setpoint eco | **18.0°C** | Morning pre-heat lower |
| Curve rise | **0.98** | 10% reduction for better COP |

**Expected outcomes**: COP 4.39 (+0.30), Grid -15%, Comfort 85%

---

### 3.3 Strategy C: Cost-Optimized
Minimize electricity costs via tariff arbitrage.

| Parameter | Value | Notes |
|-----------|-------|-------|
| Comfort start | **11:00** | After morning high-tariff peak |
| Comfort end | **21:00** | Into low-tariff evening |
| Setpoint comfort | **20.0°C** | Standard comfort |
| Setpoint eco | **17.5°C** | Aggressive night setback |
| Curve rise | **0.95** | Low flow temps |

**Dynamic rule**: When grid-dependent, reduce curve_rise to 0.85

**Expected outcomes**: COP 4.43 (+0.34), Cost -15%, Comfort 92%

---

## 4. Randomization Schedule

### 4.1 Design Principles
- **Latin square**: Each strategy appears once in each season tercile
- **Weather stratification**: Balance cold/mild periods across strategies
- **Carryover balance**: No strategy follows itself

### 4.2 Schedule Generation
Run the schedule generator before study start:
```bash
python src/phase5/generate_schedule.py --start 2027-11-01 --weeks 20 --seed 42
```

Output: `output/phase5/block_schedule.csv`

### 4.3 Example Schedule Structure

| Block | Start Date | End Date | Strategy | Season | Washout | Measurement |
|-------|------------|----------|----------|--------|---------|-------------|
| 1 | 2027-11-01 | 2027-11-05 | C | Early | Nov 1-3 | Nov 4-5 |
| 2 | 2027-11-06 | 2027-11-10 | B | Early | Nov 6-8 | Nov 9-10 |
| 3 | 2027-11-11 | 2027-11-15 | C | Early | Nov 11-13 | Nov 14-15 |
| 4 | 2027-11-16 | 2027-11-20 | A | Early | Nov 16-18 | Nov 19-20 |
| ... | ... | ... | ... | ... | ... | ... |

### 4.4 Season Definitions
- **Early winter**: Nov 1 - Dec 15 (blocks 1-9)
- **Mid winter**: Dec 16 - Feb 15 (blocks 10-21)
- **Late winter**: Feb 16 - Mar 21 (blocks 22-28)

---

## 5. Block Transition Protocol

### 5.1 Transition Timing
- **Change time**: 00:00 (midnight) on block day 1
- **Washout period**: Days 1-3 (72 hours, excluded from analysis)
- **Measurement period**: Days 4-5 (48 hours, used for analysis)
- **Block ends**: 23:59 on day 5

The 3-day washout is based on the building's thermal time constant (19.3 hours). After 3 tau (~58 hours), the system reaches 95% of the new equilibrium.

### 5.2 Step-by-Step Parameter Change Procedure

#### Before Transition (Evening Before)
1. [ ] Verify current block data is complete
2. [ ] Review next block's strategy parameters
3. [ ] Prepare any manual changes needed

#### At Transition Time (00:00)
1. [ ] **Heat pump scheduler**: Set new comfort start/end times
2. [ ] **Curve rise**: Navigate to Heating Curve menu, adjust Steilheit
3. [ ] **Home Assistant**: Update climate entity setpoints
4. [ ] **Log**: Record exact change time in block log

#### Verification (Morning After)
1. [ ] Confirm schedule is active (check heat pump display)
2. [ ] Verify setpoints responded correctly
3. [ ] Check sensor data logging is operational

### 5.3 Parameter Change Quick Reference

```
STRATEGY A (Baseline):
  Schedule: 06:30 - 20:00
  Setpoints: Comfort 20.2°C, Eco 18.5°C
  Curve rise: 1.08

STRATEGY B (Energy-Optimized):
  Schedule: 10:00 - 18:00
  Setpoints: Comfort 20.0°C, Eco 18.0°C
  Curve rise: 0.98

STRATEGY C (Cost-Optimized):
  Schedule: 11:00 - 21:00
  Setpoints: Comfort 20.0°C, Eco 17.5°C
  Curve rise: 0.95
```

---

## 6. Daily Monitoring

### 6.1 Daily Checklist

Complete each day during the study:

- [ ] **Comfort check**: Verify T_room within bounds during 08:00-22:00
- [ ] **Sensor quality**: Confirm all sensors reporting (no gaps)
- [ ] **Manual overrides**: Log any user interventions with reason
- [ ] **Occupancy notes**: Record deviations from normal patterns
- [ ] **Weather log**: Note significant weather events

### 6.2 Alert Thresholds

| Condition | Threshold | Action |
|-----------|-----------|--------|
| Room temp too low | T_room < 16°C | Abort strategy, return to Baseline |
| Flow temp too high | T_flow > 45°C | Check curve settings |
| Comfort violation | < 85% for 24h | Review and document |
| Sensor dropout | > 2h gap | Log and verify data |

### 6.3 Daily Data Log Template

```
Date: ________
Block: ___ of 28
Strategy: [ ] A  [ ] B  [ ] C

Weather:
  - Outdoor temp (min/max): ___°C / ___°C
  - HDD (base 18°C): ___
  - Cloud cover: [ ] Clear  [ ] Partial  [ ] Overcast

Comfort compliance (08:00-22:00):
  - Hours in bounds: ___ / 14
  - Min temp: ___°C at ___:___
  - Max temp: ___°C at ___:___

Energy:
  - Grid import: ___ kWh
  - PV generation: ___ kWh
  - Heat produced: ___ kWh
  - COP (if calculable): ___

Overrides/Notes:
_________________________________
_________________________________
```

---

## 7. Block Summary Log

Complete at end of each 4-day block:

### 7.1 Block Summary Template

```
BLOCK SUMMARY

Block number: ___
Strategy: ___
Dates: ___________ to ___________

Weather summary:
  - Mean outdoor temp: ___°C
  - Total HDD: ___
  - PV generation days: ___ good / ___ partial / ___ poor

Performance metrics:
  - Comfort compliance: ___% (target: ≥95%)
  - Grid consumption: ___ kWh
  - Grid per HDD: ___ kWh/HDD
  - Mean COP: ___
  - Net cost: CHF ___

Issues/Observations:
_________________________________
_________________________________

Manual overrides: [ ] None  [ ] See notes

Block quality: [ ] Good  [ ] Usable  [ ] Exclude (reason: ___)
```

---

## 8. Outcome Measures

### 8.1 Primary Outcomes

| Outcome | Definition | Unit | Target |
|---------|------------|------|--------|
| **Comfort compliance** | % time T_room in bounds (08:00-22:00) | % | ≥95% |
| **Grid per HDD** | External supply ÷ Heating degree days | kWh/HDD | Minimize |
| **COP** | Heat produced ÷ Electricity consumed | - | Maximize |
| **Net cost per HDD** | (Import×rate - Export×feedin) ÷ HDD | CHF/HDD | Minimize |

### 8.2 Secondary Outcomes

| Outcome | Definition | Unit |
|---------|------------|------|
| Solar utilization | % heating during solar hours | % |
| Battery contribution | Battery discharge for heating | kWh |
| Peak flow temperature | Max T_flow observed | °C |
| Compressor cycles | Number of on/off cycles | count |

### 8.3 Comfort Bounds by Strategy

| Strategy | Min Temp | Max Temp | Evaluation Hours |
|----------|----------|----------|------------------|
| A (Baseline) | 18.5°C | 22°C | 08:00-22:00 |
| B (Energy-Opt) | 18.5°C | 22°C | 08:00-22:00 |
| C (Cost-Opt) | 18.5°C | 22.5°C | 08:00-22:00 |

---

## 9. Safety Protocol

### 9.1 Hard Limits (Automatic Abort)

| Parameter | Limit | Action |
|-----------|-------|--------|
| Min room temperature | < 16°C | Revert to Baseline immediately |
| Max flow temperature | > 48°C | Check heat pump, reduce curve_rise |
| Heat pump error | Any fault code | Pause study, resolve issue |

### 9.2 Soft Limits (Review Required)

| Parameter | Limit | Action |
|-----------|-------|--------|
| Comfort < 85% | For 24 hours | Document, consider adjustment |
| COP < 3.0 | Single day | Check conditions, verify data |
| Manual override | Any | Document reason, include in analysis |

### 9.3 Study Pause Criteria
- Equipment malfunction requiring repair
- Extended absence (> 7 days)
- Unusual weather event (e.g., power outage)
- Request from household member

---

## 10. Statistical Analysis Plan

### 10.1 Primary Analysis

Mixed-effects linear regression:
```
Y ~ Strategy + HDD + Solar_hours + Season + (1|Block)
```

Where:
- `Y` = outcome (grid_per_HDD, COP, cost_per_HDD)
- `Strategy` = factor (A, B, C, D)
- `HDD` = heating degree days (covariate)
- `Solar_hours` = daily solar availability (covariate)
- `Season` = early/mid/late winter (factor)
- `(1|Block)` = random block effect

### 10.2 Pairwise Comparisons

- Compare each optimized strategy vs Baseline
- Tukey HSD adjustment for multiple comparisons
- Report effect size with 95% CI

### 10.3 Subgroup Analyses

- Strategy × Weather interaction (cold vs mild days)
- Strategy × Season interaction
- Weekday vs weekend effects

### 10.4 Sample Size Justification

Based on power analysis using historical data (`src/phase5/estimate_study_parameters.py`):

| Metric | Value |
|--------|-------|
| COP residual std (after HDD adjustment) | 0.21 |
| Minimum detectable effect | +0.25 COP (80% power) |
| Expected effect (Energy-Optimized) | +0.30 COP |
| Blocks per strategy | 7 |
| Statistical power | 93% |
| α | 0.05 (two-sided) |

**Key insight**: HDD explains 92.5% of COP variance. Using HDD as a covariate in the analysis reduces residual variance by 73%, enabling detection of smaller effects with fewer blocks.

---

## 11. Data Management

### 11.1 Automated Data Collection

Continuous logging via Home Assistant / InfluxDB:
- All temperature sensors (15-min intervals)
- Energy meters (15-min intervals)
- Heat pump status (1-min intervals)
- Weather station (5-min intervals)

### 11.2 Manual Data Collection

- Daily checklist (see Section 6)
- Block summary (see Section 7)
- Override log with timestamps

### 11.3 Data Quality Checks

Run weekly:
```bash
python src/phase5/data_quality_check.py --week N
```

Checks:
- Sensor gaps > 1 hour
- Outlier values (|z| > 4)
- Missing block logs

### 11.4 Data Storage

```
output/phase5/
├── block_schedule.csv          # Pre-generated schedule
├── daily_logs/                 # Daily checklist entries
│   ├── 2027-11-01.json
│   └── ...
├── block_summaries/            # Block summary entries
│   ├── block_01.json
│   └── ...
├── sensor_data/                # Extracted sensor data per block
│   ├── block_01_sensors.parquet
│   └── ...
└── analysis/                   # Statistical outputs
    ├── primary_results.csv
    └── figures/
```

---

## 12. Timeline

### 12.1 Pre-Study (October 2027)

- [ ] Week -4: Verify all sensors operational
- [ ] Week -3: Test parameter change procedures
- [ ] Week -2: Generate randomization schedule
- [ ] Week -1: Backup current settings, prepare logs

### 12.2 Study Period (November 2027 - March 2028)

| Phase | Dates | Blocks | Focus |
|-------|-------|--------|-------|
| Early winter | Nov 1 - Dec 15 | 1-11 | Initial data, refine washout |
| Mid winter | Dec 16 - Feb 15 | 12-26 | Core data collection |
| Late winter | Feb 16 - Mar 31 | 27-35 | Final blocks, wrap-up |

### 12.3 Post-Study (April 2028)

- [ ] Week +1: Final data extraction
- [ ] Week +2-3: Statistical analysis
- [ ] Week +4: Report and recommendations

---

## 13. Appendices

### Appendix A: Heating Curve Reference

The heat pump uses a heating curve to determine flow temperature based on outdoor temperature:

```
T_flow = T_setpoint + curve_rise × (T_ref - T_outdoor)
```

Example calculations (T_outdoor = 0°C):

| Strategy | Setpoint | Curve Rise | T_ref | T_flow |
|----------|----------|------------|-------|--------|
| Baseline | 20.2°C | 1.08 | 21.3°C | 43.2°C |
| Energy-Opt | 20.0°C | 0.98 | 21.3°C | 40.9°C |
| Cost-Opt | 20.0°C | 0.95 | 21.3°C | 40.2°C |

### Appendix B: Tariff Schedule Reference

| Period | Hours | Rate Type | Purchase (Rp/kWh) |
|--------|-------|-----------|-------------------|
| Mon-Fri | 06:00-21:00 | High | 35.9 |
| Mon-Fri | 21:00-06:00 | Low | 27.7 |
| Saturday | 06:00-12:00 | High | 35.9 |
| Saturday | 12:00-24:00 | Low | 27.7 |
| Sunday | All day | Low | 27.7 |
| Holidays | All day | Low | 27.7 |

### Appendix C: Sensor List

Primary sensors for analysis:
- `davis_inside_temperature` - Main indoor reference (40% weight)
- `office1_temperature` - Secondary reference (30% weight)
- `atelier_temperature` - Zone sensor (10% weight)
- `studio_temperature` - Zone sensor (10% weight)
- `simlab_temperature` - Zone sensor (10% weight)
- `davis_outside_temperature` - Outdoor reference
- `wp_hk2_ist` - Flow temperature (HK2)
- `energy_produced_heating` - Heat output
- `energy_consumed_heating` - Electricity input

### Appendix D: Emergency Procedures

**If room temperature drops below 16°C:**
1. Immediately switch to Strategy A (Baseline)
2. If still declining after 2 hours, increase setpoint to 22°C
3. Document incident in block log
4. Review cause before resuming test strategy

**If heat pump shows error:**
1. Note error code and timestamp
2. Attempt reset per manufacturer instructions
3. If unresolved, pause study and call service
4. Resume from Baseline when repaired

---

*Document version: 1.0*
*Created: January 2026*
*Author: ESTAT Project*
