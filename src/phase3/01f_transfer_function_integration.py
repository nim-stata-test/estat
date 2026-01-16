#!/usr/bin/env python3
"""
Phase 3: Transfer Function Integration Analysis

Reconciles the discrepancy between:
1. Phase 2 regression coefficients (observational associations)
2. Phase 3 transfer function (causal physical model)

Key Question: What coefficients should Phase 4 use for optimization?

Answer: The transfer function provides causal estimates via the chain:
    Parameters → T_HK2 (heating curve) → Effort → T_room (via g_eff)
"""

from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Paths
ROOT_DIR = Path(__file__).parent.parent.parent
OUTPUT_DIR = ROOT_DIR / 'output' / 'phase3'
PHASE1_DIR = ROOT_DIR / 'output' / 'phase1'
PHASE2_DIR = ROOT_DIR / 'output' / 'phase2'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Transfer function parameters (from Phase 3 analysis)
TRANSFER_FUNCTION = {
    'g_outdoor': 0.442,   # Outdoor temperature coefficient
    'g_effort': 0.208,    # Heating effort coefficient (STABLE: CV=9%)
    'g_pv': 0.0039,       # PV gain coefficient
    'tau_outdoor_h': 24,  # Outdoor LPF time constant
    'tau_effort_h': 2,    # Effort LPF time constant
    'tau_pv_h': 12,       # PV LPF time constant
}

# Phase 2 regression coefficients (observational)
PHASE2_REGRESSION = {
    'comfort_setpoint': 1.218,   # +1.22°C per 1°C setpoint
    'eco_setpoint': -0.090,
    'curve_rise': 9.73,
    'comfort_hours': -0.020,
}

# Heating curve parameters
HEATING_CURVE = {
    't_ref_comfort': 21.32,
    't_ref_eco': 19.18,
    'baseline_curve_rise': 1.08,
}

# Mean outdoor temperature during heating season
T_OUTDOOR_MEAN = 7.3  # °C (Nov-Dec 2025)


def exponential_smooth(x: np.ndarray, tau_steps: float) -> np.ndarray:
    """First-order exponential smoothing (low-pass filter)."""
    if tau_steps < 1:
        return x.copy()
    alpha = 1 - np.exp(-1/tau_steps)
    result = np.zeros_like(x, dtype=float)
    first_valid = x[~np.isnan(x)][0] if any(~np.isnan(x)) else 0
    result[0] = x[0] if not np.isnan(x[0]) else first_valid
    for i in range(1, len(x)):
        if np.isnan(x[i]):
            result[i] = result[i-1]
        else:
            result[i] = alpha * x[i] + (1 - alpha) * result[i-1]
    return result


def compute_causal_coefficients():
    """
    Derive causal coefficients from transfer function.

    The chain is:
    1. setpoint +1°C → T_HK2 +1°C (direct)
    2. T_HK2 +1°C → Effort +1°C (assuming baseline curve unchanged)
    3. Effort +1°C → LPF(Effort) +1°C (after settling)
    4. LPF(Effort) +1°C → T_room +g_eff = +0.208°C

    For curve_rise:
    1. curve_rise +0.1 → T_HK2 +0.1×(T_ref - T_outdoor) at each timestep
    2. At T_outdoor = 7°C: T_HK2 +0.1×(21.32 - 7) = +1.43°C
    3. → T_room +0.208×1.43 = +0.30°C per 0.1 curve_rise
    4. → +2.97°C per unit curve_rise
    """
    g_eff = TRANSFER_FUNCTION['g_effort']
    t_ref = HEATING_CURVE['t_ref_comfort']
    t_out_mean = T_OUTDOOR_MEAN

    # Setpoint effect: 1:1 to T_HK2, then g_eff to T_room
    setpoint_effect = g_eff * 1.0  # +1°C setpoint → +g_eff T_room

    # Curve rise effect: depends on (T_ref - T_outdoor)
    curve_rise_effect = g_eff * (t_ref - t_out_mean)

    return {
        'comfort_setpoint': setpoint_effect,
        'eco_setpoint': setpoint_effect,  # Same physical mechanism
        'curve_rise': curve_rise_effect,
        'comfort_hours': 0.0,  # Schedule doesn't directly affect temperature
    }


def load_data():
    """Load integrated dataset."""
    print("Loading data...")
    df = pd.read_parquet(PHASE1_DIR / 'integrated_dataset.parquet')

    # Filter to overlap period
    overlap_start = pd.Timestamp('2025-10-28', tz='UTC')
    df = df[df.index >= overlap_start]

    print(f"  {len(df)} rows from {df.index.min()} to {df.index.max()}")
    return df


def analyze_effort_response(df: pd.DataFrame) -> dict:
    """
    Analyze how room temperature responds to heating effort.

    This validates the g_eff coefficient by looking at actual effort changes.
    """
    print("\nAnalyzing effort response...")

    # Key columns
    outdoor_col = 'stiebel_eltron_isg_outdoor_temperature'
    room_col = 'davis_inside_temperature'
    hk2_col = 'stiebel_eltron_isg_target_temperature_hk_2'

    outdoor = df[outdoor_col].values
    hk2 = df[hk2_col].values
    room = df[room_col].values

    # Fit heating curve baseline
    model = LinearRegression()
    valid = ~(np.isnan(outdoor) | np.isnan(hk2))
    model.fit(outdoor[valid].reshape(-1, 1), hk2[valid])
    # Predict only for valid outdoor values, fill NaN for others
    hk2_baseline = np.full_like(outdoor, np.nan)
    hk2_baseline[~np.isnan(outdoor)] = model.predict(outdoor[~np.isnan(outdoor)].reshape(-1, 1))

    # Compute effort
    effort = hk2 - hk2_baseline

    # Apply LPF to effort
    effort_smooth = exponential_smooth(effort, TRANSFER_FUNCTION['tau_effort_h'] * 4)

    # Look at daily changes
    daily = pd.DataFrame({
        'effort_smooth': effort_smooth,
        'room': room,
    }, index=df.index).dropna()

    daily_avg = daily.resample('D').mean()

    # Compute lagged correlation (effort today → room tomorrow)
    effort_lag1 = daily_avg['effort_smooth'].shift(1)
    room_change = daily_avg['room'] - daily_avg['room'].shift(1)

    valid = ~(effort_lag1.isna() | room_change.isna())
    if valid.sum() > 10:
        corr = np.corrcoef(effort_lag1[valid], room_change[valid])[0, 1]

        # Linear regression to get actual coefficient
        model = LinearRegression()
        model.fit(effort_lag1[valid].values.reshape(-1, 1), room_change[valid].values)
        actual_g_eff = model.coef_[0]
    else:
        corr = np.nan
        actual_g_eff = np.nan

    print(f"  Effort-room correlation (lag 1 day): {corr:.3f}")
    print(f"  Estimated g_eff from daily changes: {actual_g_eff:.3f}")
    print(f"  Transfer function g_eff: {TRANSFER_FUNCTION['g_effort']:.3f}")

    return {
        'correlation': corr,
        'estimated_g_eff': actual_g_eff,
        'transfer_function_g_eff': TRANSFER_FUNCTION['g_effort'],
    }


def create_comparison_figure(df: pd.DataFrame):
    """Create visualization comparing coefficient estimates."""
    print("\nCreating comparison figure...")

    causal = compute_causal_coefficients()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Phase 2 Regression vs Phase 3 Transfer Function Coefficients',
                 fontsize=14, fontweight='bold')

    # Panel 1: Coefficient comparison
    ax1 = axes[0, 0]
    params = ['comfort_setpoint', 'curve_rise']
    x = np.arange(len(params))
    width = 0.35

    phase2_vals = [PHASE2_REGRESSION[p] for p in params]
    causal_vals = [causal[p] for p in params]

    bars1 = ax1.bar(x - width/2, phase2_vals, width, label='Phase 2 Regression', color='steelblue')
    bars2 = ax1.bar(x + width/2, causal_vals, width, label='Phase 3 Transfer Function', color='coral')

    ax1.set_ylabel('Coefficient (°C effect)')
    ax1.set_title('Coefficient Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Setpoint\n(°C/°C)', 'Curve Rise\n(°C/unit)'])
    ax1.legend()
    ax1.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)

    # Add value labels
    for bar, val in zip(bars1, phase2_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.2f}', ha='center', fontsize=9)
    for bar, val in zip(bars2, causal_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.2f}', ha='center', fontsize=9)

    # Panel 2: The causal chain
    ax2 = axes[0, 1]
    ax2.axis('off')

    chain_text = """
    CAUSAL CHAIN (Transfer Function)
    ─────────────────────────────────

    1. Setpoint +1°C
           ↓
    2. T_HK2 +1°C (via heating curve)
           ↓
    3. Effort +1°C (deviation from baseline)
           ↓
    4. LPF(Effort) +1°C (after τ=2h settling)
           ↓
    5. T_room +0.21°C (via g_eff=0.208)

    ─────────────────────────────────
    WHY REGRESSION OVERESTIMATES:

    • Confounding: Setpoint changes often
      coincide with other factors (weather,
      occupancy, schedule changes)

    • Regression captures ASSOCIATIONS
    • Transfer function captures CAUSATION
    """
    ax2.text(0.1, 0.9, chain_text, transform=ax2.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax2.set_title('Causal Interpretation')

    # Panel 3: Curve rise effect varies with outdoor temp
    ax3 = axes[1, 0]
    t_outdoor = np.linspace(-5, 15, 100)
    t_ref = HEATING_CURVE['t_ref_comfort']
    g_eff = TRANSFER_FUNCTION['g_effort']

    # Effect of +0.1 curve rise on T_room at different outdoor temps
    curve_rise_effect = g_eff * (t_ref - t_outdoor) * 0.1

    ax3.plot(t_outdoor, curve_rise_effect, 'b-', linewidth=2)
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax3.axvline(x=T_OUTDOOR_MEAN, color='red', linestyle='--', alpha=0.5,
                label=f'Mean outdoor ({T_OUTDOOR_MEAN}°C)')

    # Mark the mean effect
    mean_effect = g_eff * (t_ref - T_OUTDOOR_MEAN) * 0.1
    ax3.plot(T_OUTDOOR_MEAN, mean_effect, 'ro', markersize=10)
    ax3.annotate(f'{mean_effect:.2f}°C', (T_OUTDOOR_MEAN, mean_effect),
                 xytext=(T_OUTDOOR_MEAN + 2, mean_effect + 0.05),
                 fontsize=10)

    ax3.set_xlabel('Outdoor Temperature (°C)')
    ax3.set_ylabel('Room Temp Change (°C)')
    ax3.set_title('Effect of +0.1 Curve Rise on Room Temperature')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Panel 4: Summary table
    ax4 = axes[1, 1]
    ax4.axis('off')

    table_data = [
        ['Parameter', 'Phase 2\n(Regression)', 'Phase 3\n(Causal)', 'Ratio'],
        ['Setpoint (°C/°C)', f'{PHASE2_REGRESSION["comfort_setpoint"]:.2f}',
         f'{causal["comfort_setpoint"]:.2f}',
         f'{PHASE2_REGRESSION["comfort_setpoint"]/causal["comfort_setpoint"]:.1f}x'],
        ['Curve Rise (°C/unit)', f'{PHASE2_REGRESSION["curve_rise"]:.2f}',
         f'{causal["curve_rise"]:.2f}',
         f'{PHASE2_REGRESSION["curve_rise"]/causal["curve_rise"]:.1f}x'],
        ['', '', '', ''],
        ['RECOMMENDATION:', '', '', ''],
        ['Use Phase 3', 'causal coefficients', 'for optimization', ''],
    ]

    table = ax4.table(cellText=table_data, loc='center', cellLoc='center',
                      colWidths=[0.3, 0.25, 0.25, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('lightblue')
        table[(0, i)].set_text_props(weight='bold')

    # Style recommendation
    for i in range(4):
        table[(4, i)].set_facecolor('lightyellow')
        table[(5, i)].set_facecolor('lightyellow')

    ax4.set_title('Summary')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_transfer_function_integration.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: fig_transfer_function_integration.png")


def generate_report():
    """Generate integration report."""
    print("\nGenerating report...")

    causal = compute_causal_coefficients()

    html = f"""<!DOCTYPE html>
<html>
<head>
<style>
body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; }}
h1 {{ color: #2c3e50; }}
h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 5px; }}
table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
th {{ background-color: #3498db; color: white; }}
tr:nth-child(even) {{ background-color: #f9f9f9; }}
.warning {{ background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 10px; margin: 10px 0; }}
.recommendation {{ background-color: #d4edda; border-left: 4px solid #28a745; padding: 10px; margin: 10px 0; }}
.code {{ background-color: #f4f4f4; padding: 10px; font-family: monospace; overflow-x: auto; }}
</style>
</head>
<body>

<h1>Transfer Function Integration Analysis</h1>

<h2>The Problem</h2>
<p>Phase 4 optimization uses regression coefficients from Phase 2 that predict how room temperature
responds to heating parameter changes:</p>

<table>
<tr><th>Parameter</th><th>Phase 2 Coefficient</th><th>Interpretation</th></tr>
<tr><td>comfort_setpoint</td><td>+1.22°C per °C</td><td>+1°C setpoint → +1.22°C room</td></tr>
<tr><td>curve_rise</td><td>+9.73°C per unit</td><td>+0.1 rise → +0.97°C room</td></tr>
</table>

<div class="warning">
<strong>Issue:</strong> These coefficients are from observational data - they capture <em>associations</em>,
not <em>causal effects</em>. If setpoint changes historically coincided with other factors
(weather, occupancy, schedule), the regression attributes all effects to the setpoint.
</div>

<h2>The Transfer Function (Phase 3)</h2>
<p>The transfer function models the physical causal chain:</p>

<div class="code">
T_room = offset + g_outdoor×LPF(T_outdoor, 24h) + g_effort×LPF(Effort, 2h) + g_pv×LPF(PV, 12h)

Where:
- g_effort = 0.208 (STABLE: coefficient of variation = 9%)
- Effort = T_HK2 - baseline_curve
- T_HK2 = setpoint + curve_rise × (T_ref - T_outdoor)
</div>

<h2>Deriving Causal Coefficients</h2>
<p>Following the causal chain:</p>

<ol>
<li><strong>Setpoint +1°C</strong> → T_HK2 +1°C (direct via heating curve)</li>
<li><strong>T_HK2 +1°C</strong> → Effort +1°C (deviation from baseline)</li>
<li><strong>Effort +1°C</strong> → LPF(Effort) +1°C (after τ=2h settling)</li>
<li><strong>LPF(Effort) +1°C</strong> → T_room +{TRANSFER_FUNCTION['g_effort']:.3f}°C (via g_effort)</li>
</ol>

<table>
<tr><th>Parameter</th><th>Phase 2 (Regression)</th><th>Phase 3 (Causal)</th><th>Ratio</th></tr>
<tr>
<td>comfort_setpoint</td>
<td>{PHASE2_REGRESSION['comfort_setpoint']:.2f}°C/°C</td>
<td>{causal['comfort_setpoint']:.2f}°C/°C</td>
<td>{PHASE2_REGRESSION['comfort_setpoint']/causal['comfort_setpoint']:.1f}x</td>
</tr>
<tr>
<td>curve_rise</td>
<td>{PHASE2_REGRESSION['curve_rise']:.2f}°C/unit</td>
<td>{causal['curve_rise']:.2f}°C/unit</td>
<td>{PHASE2_REGRESSION['curve_rise']/causal['curve_rise']:.1f}x</td>
</tr>
</table>

<p>The regression overestimates effects by <strong>3-6x</strong>!</p>

<h2>Why This Matters for Optimization</h2>

<p>If Phase 4 uses the inflated regression coefficients:</p>
<ul>
<li>It overestimates temperature gains from setpoint increases</li>
<li>It overestimates temperature losses from curve rise reductions</li>
<li>Optimization may select strategies that don't actually achieve predicted comfort</li>
</ul>

<div class="recommendation">
<strong>Recommendation:</strong> Update Phase 4 to use causal coefficients from the transfer function.
<br><br>
<strong>Updated coefficients:</strong>
<ul>
<li>comfort_setpoint: {causal['comfort_setpoint']:.3f} (was {PHASE2_REGRESSION['comfort_setpoint']:.3f})</li>
<li>eco_setpoint: {causal['eco_setpoint']:.3f} (was {PHASE2_REGRESSION['eco_setpoint']:.3f})</li>
<li>curve_rise: {causal['curve_rise']:.2f} (was {PHASE2_REGRESSION['curve_rise']:.2f})</li>
</ul>
</div>

<h2>Important Caveats</h2>

<ol>
<li><strong>Transfer function R² = 0.68</strong> - captures only 68% of variance.
    The adaptive model (RLS) achieves 0.86 but parameters vary over time.</li>
<li><strong>g_effort is stable</strong> (CV=9%) but g_outdoor varies significantly (CV=95%)</li>
<li><strong>Schedule effects are indirect</strong> - comfort_hours doesn't directly affect
    temperature; it changes the proportion of time in comfort vs eco mode</li>
<li><strong>Nonlinear effects</strong> - curve_rise effect depends on outdoor temperature:
    larger effect at colder outdoor temps</li>
</ol>

<h2>Integration Proposal</h2>

<p>Replace the simple linear adjustment in Phase 4 with a physics-based simulation:</p>

<div class="code">
# Current (problematic):
delta_T = coef_setpoint × (setpoint - baseline)
T_room_adjusted = T_room_historical + delta_T

# Proposed (physics-based):
for each timestep:
    T_HK2 = compute_heating_curve(setpoint, curve_rise, T_outdoor, schedule)
    Effort = T_HK2 - baseline_curve(T_outdoor)
    T_room_pred = offset + g_eff × LPF(Effort) + g_out × LPF(T_outdoor) + g_pv × LPF(PV)
</div>

<p>This ensures the optimization respects the physical constraints of the system.</p>

<figure>
<img src="fig_transfer_function_integration.png" alt="Coefficient comparison" style="max-width:100%">
<figcaption>Figure: Comparison of Phase 2 regression vs Phase 3 transfer function coefficients</figcaption>
</figure>

</body>
</html>"""

    with open(OUTPUT_DIR / 'transfer_function_integration_report.html', 'w') as f:
        f.write(html)
    print("  Saved: transfer_function_integration_report.html")


def save_causal_coefficients():
    """Save causal coefficients for Phase 4 to use."""
    causal = compute_causal_coefficients()

    output = {
        'source': 'Phase 3 transfer function analysis',
        'method': 'Causal coefficients derived from g_eff × physical pathways',
        'g_eff': TRANSFER_FUNCTION['g_effort'],
        'g_eff_stability': 'CV = 9% (stable)',
        'coefficients': causal,
        'heating_curve': HEATING_CURVE,
        't_outdoor_mean': T_OUTDOOR_MEAN,
        'note': 'These replace Phase 2 regression coefficients for optimization',
    }

    with open(OUTPUT_DIR / 'causal_coefficients.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: causal_coefficients.json")


def main():
    """Run transfer function integration analysis."""
    print("=" * 60)
    print("Phase 3: Transfer Function Integration Analysis")
    print("=" * 60)

    # Compute and display causal coefficients
    causal = compute_causal_coefficients()

    print("\n" + "-" * 60)
    print("COEFFICIENT COMPARISON")
    print("-" * 60)
    print(f"\n{'Parameter':<20} {'Phase 2':<12} {'Phase 3':<12} {'Ratio':<10}")
    print(f"{'(Regression)':<20} {'(Causal)':<12} {'':<12}")
    print("-" * 60)

    for param in ['comfort_setpoint', 'curve_rise']:
        p2 = PHASE2_REGRESSION[param]
        p3 = causal[param]
        ratio = p2 / p3 if p3 != 0 else float('inf')
        print(f"{param:<20} {p2:<12.2f} {p3:<12.2f} {ratio:.1f}x")

    print("\n" + "-" * 60)
    print("KEY INSIGHT")
    print("-" * 60)
    print("""
Phase 2 regression coefficients overestimate effects by 3-6x
because they capture associations, not causal effects.

The transfer function provides causal estimates via g_eff = 0.208:
  - Setpoint +1°C → T_room +0.21°C (not +1.22°C)
  - Curve rise +1 → T_room +2.9°C (not +9.73°C)

RECOMMENDATION: Phase 4 should use the causal coefficients.
""")

    # Load data and create visualizations
    df = load_data()
    analyze_effort_response(df)
    create_comparison_figure(df)
    generate_report()
    save_causal_coefficients()

    print("\n" + "=" * 60)
    print("Analysis complete. See output/phase3/ for results.")
    print("=" * 60)


if __name__ == '__main__':
    main()
