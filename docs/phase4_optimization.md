# Phase 4: Optimization Strategy Development

## Outputs

After running `python src/phase4/run_optimization.py`, outputs are saved to `output/phase4/`:

**Figures (fig4.01-fig4.10):**
- fig4.01: Strategy comparison (COP by strategy, schedule alignment, expected improvements)
- fig4.02: Simulation results (time series, self-sufficiency, hourly COP profiles)
- fig4.03: Parameter space (trade-offs, parameter summary table)
- fig4.04: Pareto front (2D projections of Pareto-optimal solutions)
- fig4.05: Pareto strategy comparison (radar chart comparing strategies)
- fig4.06: Pareto evolution (optimization history animation frame)
- fig4.07: Strategy temperature predictions (winter 2026/2027, violation analysis)
- fig4.08: Detailed time series (T_weighted, outdoor, solar, grid by strategy)
- fig4.09: Hourly patterns (temperature heatmaps, PV/grid profiles, comfort windows)
- fig4.10: Energy patterns (daily balance, self-sufficiency, temperature distributions)

**Reports:**
- `phase4_report.html` - Combined optimization report
- `optimization_strategies.csv` - Strategy definitions and rules
- `strategy_comparison.csv` - Simulated metrics by strategy
- `simulation_daily_metrics.csv` - Daily simulation results

**Phase 5 Preparation:**
- `phase5_parameter_sets.json` - Exact parameter values for intervention study
- `phase5_predictions.json` - Testable predictions with confidence intervals
- `phase5_implementation_checklist.md` - Protocol for randomized study

## Heating Curve Model (from Phase 2)

```
T_HK2 = T_setpoint + curve_rise * (T_ref - T_outdoor)
```
Where:
- T_HK2 = target flow temperature (heating curve setpoint)
- T_ref = 21.32C (comfort mode) or 19.18C (eco mode)
- curve_rise typically 0.85-1.08

## Three Optimization Strategies

| Strategy | Schedule | Curve Rise | COP | vs Baseline | Goal |
|----------|----------|------------|-----|-------------|------|
| Baseline | 06:30-20:00 | 1.08 | 4.09 | - | Reference |
| Energy-Optimized | 10:00-18:00 | 0.98 | 4.39 | +0.18 | Minimize grid |
| Cost-Optimized | 11:00-21:00 | 0.95/0.85* | 4.43 | +0.22 | Minimize costs |

*Cost-Optimized uses curve_rise 0.85 when grid-dependent

## Comfort Evaluation

- Comfort constraint evaluated **only during occupied hours (08:00-22:00)**
- Constraint: T_weighted < 18.5C for <=5% of daytime hours (soft penalty)
- No upper temperature limit - higher is always better
- Night temperatures (22:00-08:00) are excluded from comfort objectives

## Key Optimization Levers

- Shift comfort mode to PV peak hours (10:00-17:00)
- Lower curve_rise for better COP (~0.1 COP improvement per 1C flow temp reduction)
- Dynamic curve_rise reduction when grid-dependent (0.85-0.90)
- Tariff arbitrage: shift heating to low-tariff periods (21:00-06:00, weekends)

---

## Pareto Optimization (Step 4)

Multi-objective optimization using NSGA-II to find Pareto-optimal heating strategies.

### Commands

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

### Default Behavior

- Auto-detects `pareto_archive.json` and uses it for warm start
- Runs 200 generations (full optimization)
- Use `--fresh` to start from scratch
- Uses epsilon-dominance to filter meaningfully different solutions (default: enabled)

### Epsilon-Dominance Filtering

Standard Pareto optimization produces many solutions that differ by negligible amounts
(e.g., 0.01C temperature difference). Epsilon-dominance keeps only solutions that are
meaningfully different by snapping objectives to an epsilon grid before dominance comparison.

| Objective | Epsilon | Description |
|-----------|---------|-------------|
| Temperature | **0.1C** | Comfort differences below this are imperceptible |
| Grid import | **100 kWh** | ~5% of typical total range |
| Cost | **10 CHF** | Fine-grained cost differences |

Effect: Reduces hundreds of solutions -> ~3-5 meaningfully different solutions.

```bash
# Disable epsilon-dominance (keep all Pareto solutions)
python src/phase4/04_pareto_optimization.py --no-epsilon

# Custom epsilon values (more conservative filtering)
python src/phase4/04_pareto_optimization.py --eps-temp 0.3 --eps-grid 30 --eps-cost 10
```

### Decision Variables (5)

| Variable | Range | Description |
|----------|-------|-------------|
| `setpoint_comfort` | [19.0, 22.0] C | Comfort mode target temperature |
| `setpoint_eco` | [12.0, 19.0] C | Eco mode target (12C = frost protection) |
| `comfort_start` | [06:00, 12:00] | Start of comfort period |
| `comfort_end` | [16:00, 22:00] | End of comfort period |
| `curve_rise` | [0.80, 1.20] | Heating curve slope (Steilheit) |

### Objectives (3, all minimized in NSGA-II)

1. **Negative mean temperature**: `-mean(T_weighted)` during 08:00-22:00 (minimizing = maximize avg temp)
2. **Grid import**: Total kWh purchased from grid
3. **Net cost**: Grid cost - feed-in revenue (CHF)

### Comfort Constraint Parameters

| Parameter | Value | Code Constant | Description |
|-----------|-------|---------------|-------------|
| `COMFORT_THRESHOLD` | 18.5C | `04_pareto_optimization.py:260` | Minimum acceptable T_weighted |
| `VIOLATION_LIMIT` | **5%** | `04_pareto_optimization.py:348` | Max allowed daytime hours below threshold |
| `OCCUPIED_START` | 08:00 | `04_pareto_optimization.py:98` | Start of comfort evaluation window |
| `OCCUPIED_END` | 22:00 | `04_pareto_optimization.py:99` | End of comfort evaluation window |

### Constraint Mechanism (soft penalty)

The optimizer uses NSGA-II's constraint-handling approach:
- Constraint function: `g = violation_pct - 0.05`
- If `g <= 0`: Solution is **feasible** (<=5% of daytime below 18.5C)
- If `g > 0`: Solution is **infeasible** but NOT excluded
- Feasible solutions always dominate infeasible ones in ranking
- Among infeasible solutions, smaller violation is preferred
- No explicit penalty coefficient; constraint satisfaction is binary for dominance

**Note:** The 5% limit was tightened from 20% in Jan 2026 after evaluation showed
energy-optimized strategies had 15-19% violation and minimum temps of 16.7C.

### T_weighted Adjustment Model

Uses Phase 2 regression coefficients to adjust historical T_weighted based on parameter changes:
- Comfort setpoint: +1.22C per 1C increase
- Eco setpoint: -0.09C per 1C increase (negligible - allows aggressive setback)
- Curve rise: +9.73C per unit increase

**Key Finding (Jan 2026):** Phase 2 multivariate analysis revealed eco setpoint has minimal
effect on daytime comfort (-0.09C per 1C change). This allows aggressive eco setbacks
(down to 12C) without compromising comfort during occupied hours.

### Outputs

```
output/phase4/
├── pareto_archive.json        # Full archive with optimization history
├── pareto_front.csv           # All Pareto-optimal solutions
├── selected_strategies.csv    # 10 diverse strategies selected
├── selected_strategies.json   # Machine-readable format
├── fig4.04_pareto_front.png     # 2D projections of Pareto front
├── fig4.05_pareto_strategy_comparison.png # Radar chart comparing strategies
├── fig4.06_pareto_evolution.png # Pareto front evolution frame
├── pareto_evolution.gif       # 2D animated Pareto evolution
├── pareto_evolution.mp4       # 2D animation for PowerPoint
├── pareto_evolution_3d.gif    # 3D animated Pareto evolution
├── pareto_evolution_3d.mp4    # 3D animation for PowerPoint
└── pareto_report_section.html # HTML report section
```

### Optimization History (in pareto_archive.json)

The archive includes full optimization history for visualization:
- `optimization_history.generations`: Per-generation snapshots with population composition
- `optimization_history.all_solutions`: All unique parameter sets evaluated
- Each solution tracks: `first_gen`, `pareto_generations` (list of generations where it was on the Pareto front)
- Enables animated visualization of Pareto front evolution

### Workflow

1. First run: `python src/phase4/04_pareto_optimization.py --fresh` (full optimization, 200 generations)
2. Refinement: `python src/phase4/04_pareto_optimization.py -g 50` (auto warm-start, quick refinement)
3. **Evaluate strategies**: `python src/phase4/05_strategy_evaluation.py`
4. Review violation analysis in `strategy_violation_analysis.csv`
5. Manually select 3 strategies for Phase 5 intervention study

---

## Grid Search Optimization (Step 4b)

Exhaustive grid search alternative to NSGA-II for finding Pareto-optimal heating strategies.
Evaluates all valid parameter combinations on a discrete grid.

### Commands

```bash
# Run grid search (default resolution, ~38 minutes)
python src/phase4/04b_grid_search_optimization.py

# Coarser grid for faster results (~8 minutes)
python src/phase4/04b_grid_search_optimization.py --coarse
```

### Grid Configuration (default)

| Variable | Min | Max | Step | Values |
|----------|-----|-----|------|--------|
| `setpoint_comfort` | 19.0C | 22.0C | 0.5C | 7 |
| `setpoint_eco` | 12.0C | 19.0C | 1.0C | 8 |
| `comfort_start` | 06:00 | 12:00 | 0.5h | 13 |
| `comfort_end` | 16:00 | 22:00 | 0.5h | 13 |
| `curve_rise` | 0.80 | 1.20 | 0.05 | 9 |

**Coarse grid** (`--coarse`): Steps doubled for ~4x faster runtime.

### Search Space

- Total combinations: 85,176 (before filtering)
- Valid combinations: ~85,000 (after min_hours >= 4h constraint)
- Feasible solutions: ~5.6% (pass comfort constraint)
- Evaluation time: ~38 minutes (85k evals)

### Comparison with NSGA-II

| Aspect | NSGA-II | Grid Search |
|--------|---------|-------------|
| Evaluations | ~2,100 (50 pop x 42 gen) | 85,176 |
| Runtime | ~3 minutes | ~38 minutes |
| Coverage | Stochastic, may miss regions | Exhaustive, complete |
| Pareto solutions | ~3-5 after epsilon-dominance | 9 (all convergent) |

### Key Finding (Jan 2026)

All 9 Pareto solutions from grid search converge to:
- **Schedule**: 12:00-16:00 (4-hour comfort window)
- **Eco setpoint**: 19C (maximum allowed)
- **Variation only in**: comfort setpoint (21.5-22.0C) and curve_rise (1.00-1.20)

This confirms the NSGA-II result: optimal strategies use narrow afternoon comfort windows
aligned with PV production, with minimal temperature setback during eco periods.

### Outputs

```
output/phase4/
├── grid_search_all_results.csv    # All 85,176 evaluations
├── grid_search_pareto.csv         # 9 Pareto-optimal solutions
├── grid_search_pareto.json        # Machine-readable Pareto front
├── fig4.11_grid_search_results.png  # Feasible region visualization
└── fig4.12_objective_landscape.png  # Objective space heatmaps
```

### Pareto Solutions (Jan 2026, at T_outdoor ref = 5C)

| Label | Comf C | Eco C | Schedule | Rise | Temp C | Grid kWh | Cost CHF |
|-------|--------|-------|----------|------|--------|----------|----------|
| Comfort-First | 22.0 | 19 | 12-16h | 1.20 | 20.2 | 2287 | 660 |
| Balanced-1-6 | 21.5-22.0 | 19 | 12-16h | 1.05-1.20 | 19.6-20.1 | 2145-2281 | 618-658 |
| Grid-Minimal | 22.0 | 19 | 12-16h | 1.00 | 19.6 | 2109 | 608 |

### When to Use

- Use NSGA-II (04_pareto_optimization.py) for rapid exploration and iterative refinement
- Use grid search (04b_grid_search_optimization.py) for exhaustive validation and complete coverage

---

## Strategy Evaluation (Step 5)

Evaluates selected strategies for comfort violations and generates winter predictions.

```bash
python src/phase4/05_strategy_evaluation.py
```

### Evaluation Results (Jan 2026, 500 pop x 100 gen)

| Strategy | Violation % | Cold Hours | Min Temp | Mean Temp | Status |
|----------|-------------|------------|----------|-----------|--------|
| Grid-Minimal | 2.9% | 31h | 17.7C | 19.6C | Pass |
| Balanced | 2.9% | 31h | 17.7C | 19.6C | Pass |
| Cost-Minimal | 2.9% | 31h | 17.7C | 19.6C | Pass |
| Comfort-First | 0.0% | 0h | 21.5C | 23.4C | Comfortable |

### Key Improvements from Large-Scale Optimization

- Violation reduced: 4.5% -> 2.9% (well under 5% limit)
- Cold hours reduced: 40h -> 31h
- Min temps improved: 17.3C -> 17.7C
- Optimizer found lower curve_rise: 0.89-0.90 -> 0.82 with higher eco setpoint (14C)

### Outputs

```
output/phase4/
├── fig4.07_strategy_temperature_predictions.png  # Winter 2026/2027 predictions
├── strategy_violation_analysis.csv             # Detailed violation stats
└── strategy_evaluation_report.html             # HTML report section
```

---

## Detailed Strategy Analysis (Step 6)

```bash
python src/phase4/06_strategy_detailed_analysis.py
```

### Outputs

- `strategy_detailed_stats.csv` - Comprehensive statistics for Phase 5 strategies
- `strategy_detailed_report.html` - HTML report with time series and energy analysis
- `fig4.08_strategy_detailed_timeseries.png` - Full-period temperature and energy time series
- `fig4.09_strategy_hourly_patterns.png` - Hourly patterns, heatmaps, and comfort windows
- `fig4.10_strategy_energy_patterns.png` - Energy balance, self-sufficiency, temperature distributions
