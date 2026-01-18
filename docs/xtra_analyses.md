# Extra Standalone Analyses

Standalone analyses not part of the Phase 5 study pipeline, located in `src/xtra/`.

## Battery Degradation Analysis

Investigates whether the Feb-Mar 2025 deep-discharge event affected battery efficiency.

```bash
python src/xtra/battery_degradation/battery_degradation.py
```

### Outputs (in `output/xtra/battery_degradation/`)

- `battery_degradation_analysis.png` - 4-panel visualization
- `battery_degradation_report.html` - Detailed report with methods & results

### Analysis Includes

- OLS regression with time trend and post-event indicator
- Welch's t-test as robustness check
- Key finding: Statistically significant efficiency drop of ~10.8 percentage points (p<0.001)

---

## Battery Cost Savings Analysis

Analyzes how much the battery saves compared to a hypothetical system without a battery.
Considers time-varying tariffs and 30% tax on feed-in income.

```bash
python src/xtra/battery_savings/battery_savings.py
```

### Outputs (in `output/xtra/battery_savings/`)

- `battery_savings_daily.csv` - Daily cost savings data
- `battery_savings_analysis.png` - Cumulative savings visualization
- `battery_savings_report.html` - Summary report with methodology

### Methodology

```
Savings = battery_discharge * purchase_rate - battery_charge * feedin_rate * 0.70
```
- Without battery: charging energy would be fed to grid, discharging energy would be imported
- The 0.70 factor accounts for 30% income tax on feed-in revenue
- Time-varying tariffs applied at 15-minute resolution

### Key Finding

Battery saved CHF 1,143 over ~2.8 years (CHF 1.10/day average)

---

## System Diagram

Semi-abstract diagram of the energy system for PowerPoint presentations.

```bash
python src/xtra/system_diagram.py
```

### Outputs (in `output/xtra/`)

- `system_diagram.png` - PNG format (300 DPI)
- `system_diagram.pdf` - PDF format (vector)
- `system_diagram.svg` - SVG format (vector)

### Components Shown

- Sun -> PV panels (electricity flow, black)
- PV -> Battery, Grid, House (electricity flows, black)
- Grid <-> House (bidirectional electricity, black)
- Heat pump with outdoor temperature input
- Buffer tank with heat distribution to house (heat flows, red)
- Temperature sensors (T_outdoor, T_room, T_HK2)
