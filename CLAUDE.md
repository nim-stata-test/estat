# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ESTAT is an energy balance data repository for solar/battery system monitoring with heating optimization. The project analyzes historical energy data to optimize heat pump heating strategies for comfort, grid independence, and cost.

## Development Environment

- **Language**: Python 3.13 (with numba JIT for 10x faster optimization)
- **IDE**: PyCharm configured
- **Virtual env**: `.venv/`

## Quick Start

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

# Run all phases
python src/run_all.py              # Complete pipeline (skips Pareto)
python src/run_all.py --phase 1    # Phase 1 only
python src/run_all.py --phase 2    # Phase 2 only
python src/run_all.py --phase 3    # Phase 3 only
python src/run_all.py --phase 4    # Phase 4 only
python src/run_all.py --list       # List all steps

# Phase wrappers (generate HTML reports)
python src/phase1/run_preprocessing.py   # ~12 min
python src/phase2/run_eda.py
python src/phase3/run_phase3.py
python src/phase4/run_optimization.py

# Main report
python src/generate_main_report.py       # output/index.html
```

## Phase 4 Pipeline (Default)

`python src/phase4/run_optimization.py` runs all steps:

| Step | Script | Description |
|------|--------|-------------|
| 4.1 | `01_rule_based_strategies.py` | Define baseline + optimized strategies |
| 4.2 | `02_strategy_simulation.py` | Simulate on historical data |
| 4.3 | `03_parameter_sets.py` | Generate Phase 5 parameters |
| 4.4a | `04b_grid_search_optimization.py --coarse` | Grid search (~3 min) |
| 4.4b | `04_pareto_optimization.py` | NSGA-II multi-objective |
| 4.5 | `05_strategy_evaluation.py` | Comfort violation analysis |
| 4.6 | `06_strategy_detailed_analysis.py` | Detailed visualizations |
| 4.7 | `07_pareto_animation.py` | Pareto evolution GIF/MP4 |

**Outputs:** 12 figures (fig4.01-fig4.12), animations, combined HTML report.

## Key Commands

| Command | Description |
|---------|-------------|
| `python src/phase4/04_pareto_optimization.py --fresh` | Fresh NSGA-II (ignore archive) |
| `python src/phase4/04b_grid_search_optimization.py` | Full grid search (~38 min) |
| `python src/phase4/04b_grid_search_optimization.py --coarse` | Coarse grid (~3 min) |
| `python src/phase5_pilot/run_pilot.py` | Generate pilot design + schedule |
| `python src/phase5_pilot/run_pilot.py --analyze-rsm` | RSM block-averaged analysis |

## Source Code Structure

```
src/
├── run_all.py           # Master script
├── phase1/              # Data Preprocessing
├── phase2/              # Exploratory Data Analysis
├── phase3/              # System Modeling (thermal, COP, energy)
├── phase4/              # Optimization Strategy Development
├── phase5/              # Intervention Study (Nov 2027 - Mar 2028)
├── phase5_pilot/        # Pilot Experiment (Jan-Mar 2026)
├── shared/              # Shared modules (energy_system.py, report_style.py)
└── xtra/                # Standalone analyses (battery degradation, savings)
```

## Shared Constants

The `src/shared/__init__.py` module exports constants used across all phases:

| Constant | Value | Description |
|----------|-------|-------------|
| `ANALYSIS_START_DATE` | 2025-10-29 | Good sensor data begins here. All model estimation and weekly analyses use this as start. |

Usage: `from shared import ANALYSIS_START_DATE`

## Output Structure

```
output/
├── index.html           # Main report with TOC
├── phase1/              # Parquet files, preprocessing report
├── phase2/              # EDA figures (fig2.01-fig2.19), HTML reports
├── phase3/              # Model figures (fig3.01-fig3.08), weekly decomposition
├── phase4/              # Optimization figures (fig4.01-fig4.12), Pareto archive, animations
├── phase5/              # Intervention study outputs
├── phase5_pilot/        # Pilot experiment outputs
└── xtra/                # Standalone analysis outputs
```

## Key Models

**Heating Curve:**
```
T_HK2 = T_setpoint + curve_rise * (T_ref - T_outdoor)
```
Where T_ref = 21.32C (comfort) or 19.18C (eco). See `output/phase2/heating_curve_params.json`.

**COP Model:**
```
COP = 5.93 + 0.13*T_outdoor - 0.08*T_HK2   (R2=0.94, clipped to [1.5, 8.0])
```

**Transfer Function Thermal Model:**
```
T_room = offset + g_outdoor*LPF(T_outdoor,24h) + g_effort*LPF(Effort,2h) + g_pv*LPF(PV,12h)
```
Where Effort = T_HK2 - baseline_curve. Key: g_effort = 0.208 (stable, CV=9%).

## Optimization Summary

**Three objectives:** Maximize comfort (avg temp), minimize grid import, minimize net cost

**Comfort constraint:** T_weighted < 18.5C for <=5% of daytime hours (08:00-22:00)

**Decision variables:** setpoint_comfort [19-22C], setpoint_eco [12-19C], comfort_start [06-12h], comfort_end [16-22h], curve_rise [0.80-1.20]

**Key finding:** Optimal strategies use narrow afternoon comfort windows (12:00-16:00) aligned with PV production, minimal eco setback (19C), achieving 2.9% violation with 17.7C minimum.

## Key Sensors

- **Indoor temperature:** `davis_inside_temperature` (100% weight, least noise)
- **Outdoor temperature:** `stiebel_eltron_isg_outdoor_temperature` (heat pump sensor, better COP prediction than Davis weather station)

## Detailed Documentation

| Topic | File |
|-------|------|
| Data format, tariffs | [`docs/data_format.md`](docs/data_format.md) |
| Phase 2 EDA, heating curve | [`docs/phase2_eda.md`](docs/phase2_eda.md) |
| Phase 3 models | [`docs/phase3_models.md`](docs/phase3_models.md) |
| Phase 4 optimization | [`docs/phase4_optimization.md`](docs/phase4_optimization.md) |
| Phase 5 study | [`docs/phase5_study.md`](docs/phase5_study.md) |
| Energy system module | [`docs/energy_system.md`](docs/energy_system.md) |
| Standalone analyses | [`docs/xtra_analyses.md`](docs/xtra_analyses.md) |
| Experimental protocol | [`docs/phase5_experimental_design.md`](docs/phase5_experimental_design.md) |
| Research design | [`PRD.md`](PRD.md) |
