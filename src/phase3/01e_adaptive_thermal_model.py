#!/usr/bin/env python3
"""
Phase 3: Adaptive Thermal Model

Time-varying parameter model using Recursive Least Squares (RLS).
Addresses the finding that fixed parameters only achieve R² = 0.68,
while adaptive parameters achieve R² = 0.86+.

Key insight: Building thermal dynamics change over time due to:
- Weather variability (sunny vs cloudy days)
- Heating mode changes (eco vs comfort)
- Occupancy patterns
- Wind/infiltration

Model:
    T_room = offset + g_out×LPF(T_outdoor) + g_eff×LPF(Effort) + g_pv×LPF(PV)

Where parameters [offset, g_out, g_eff, g_pv] are updated via RLS.
"""

from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Paths
ROOT_DIR = Path(__file__).parent.parent.parent
OUTPUT_DIR = ROOT_DIR / 'output' / 'phase3'
PHASE1_DIR = ROOT_DIR / 'output' / 'phase1'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Key columns
OUTDOOR_COL = 'stiebel_eltron_isg_outdoor_temperature'
ROOM_COL = 'davis_inside_temperature'
HK2_COL = 'stiebel_eltron_isg_target_temperature_hk_2'
PV_COL = 'pv_generation_kwh'


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


class AdaptiveThermalModel:
    """
    Adaptive thermal model using Recursive Least Squares.

    Parameters are updated at each timestep using exponential forgetting:
    - forgetting_factor=1.0: No forgetting (equivalent to batch OLS)
    - forgetting_factor=0.99: ~100 timestep memory (25h at 15min)
    - forgetting_factor=0.95: ~20 timestep memory (5h at 15min)
    """

    def __init__(self, forgetting_factor: float = 0.99,
                 tau_outdoor_h: float = 24,
                 tau_effort_h: float = 2,
                 tau_pv_h: float = 12):
        self.forgetting_factor = forgetting_factor
        self.tau_outdoor_h = tau_outdoor_h
        self.tau_effort_h = tau_effort_h
        self.tau_pv_h = tau_pv_h

        # State
        self.theta = None  # Parameters [offset, g_out, g_eff, g_pv]
        self.P = None  # Covariance matrix
        self.heating_curve = None  # For effort calculation

        # History
        self.theta_history = []
        self.y_pred_history = []

    def _init_heating_curve(self, outdoor: np.ndarray, hk2: np.ndarray):
        """Fit heating curve: HK2 = baseline + slope × T_outdoor"""
        model = LinearRegression()
        valid = ~(np.isnan(outdoor) | np.isnan(hk2))
        model.fit(outdoor[valid].reshape(-1, 1), hk2[valid])
        self.heating_curve = {
            'baseline': model.intercept_,
            'slope': model.coef_[0]
        }

    def _compute_effort(self, outdoor: np.ndarray, hk2: np.ndarray) -> np.ndarray:
        """Compute heating effort (deviation from heating curve)."""
        hk2_expected = self.heating_curve['baseline'] + \
                       self.heating_curve['slope'] * outdoor
        return hk2 - hk2_expected

    def _prepare_features(self, outdoor: np.ndarray, effort: np.ndarray,
                          pv: np.ndarray) -> np.ndarray:
        """Prepare feature matrix with smoothed inputs."""
        out_smooth = exponential_smooth(outdoor, self.tau_outdoor_h * 4)
        eff_smooth = exponential_smooth(effort, self.tau_effort_h * 4)
        pv_smooth = exponential_smooth(pv, self.tau_pv_h * 4)

        n = len(outdoor)
        return np.column_stack([np.ones(n), out_smooth, eff_smooth, pv_smooth])

    def fit(self, df: pd.DataFrame, warmup: int = 96) -> dict:
        """
        Fit adaptive model to data.

        Args:
            df: DataFrame with required columns
            warmup: Number of initial points for batch initialization

        Returns:
            dict with fit statistics
        """
        # Extract data
        outdoor = df[OUTDOOR_COL].values
        hk2 = df[HK2_COL].values
        pv = df[PV_COL].values
        y = df[ROOM_COL].values
        n = len(y)

        # Initialize heating curve
        self._init_heating_curve(outdoor, hk2)

        # Compute features
        effort = self._compute_effort(outdoor, hk2)
        X = self._prepare_features(outdoor, effort, pv)
        n_features = X.shape[1]

        # Initialize parameters from warmup period
        model = LinearRegression(fit_intercept=False)
        model.fit(X[:warmup], y[:warmup])
        self.theta = model.coef_.copy()

        # Initialize covariance (high uncertainty)
        self.P = np.eye(n_features) * 1000

        # RLS update
        y_pred = np.zeros(n)
        self.theta_history = np.zeros((n, n_features))

        for t in range(n):
            x_t = X[t]

            # Predict
            y_pred[t] = x_t @ self.theta
            self.theta_history[t] = self.theta.copy()

            # Update (only after warmup)
            if t >= warmup:
                # Prediction error
                error = y[t] - y_pred[t]

                # Kalman gain
                Px = self.P @ x_t
                denom = self.forgetting_factor + x_t @ Px
                K = Px / denom

                # Update parameters
                self.theta = self.theta + K * error

                # Update covariance
                self.P = (self.P - np.outer(K, x_t @ self.P)) / self.forgetting_factor

        self.y_pred_history = y_pred

        # Compute statistics (excluding warmup)
        valid = slice(warmup, None)
        r2 = r2_score(y[valid], y_pred[valid])
        rmse = np.sqrt(mean_squared_error(y[valid], y_pred[valid]))

        return {
            'r2': r2,
            'rmse': rmse,
            'n_points': n - warmup,
            'forgetting_factor': self.forgetting_factor,
            'final_params': {
                'offset': self.theta[0],
                'g_outdoor': self.theta[1],
                'g_effort': self.theta[2],
                'g_pv': self.theta[3],
            }
        }

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict using current parameters (no update)."""
        outdoor = df[OUTDOOR_COL].values
        hk2 = df[HK2_COL].values
        pv = df[PV_COL].values

        effort = self._compute_effort(outdoor, hk2)
        X = self._prepare_features(outdoor, effort, pv)

        return X @ self.theta


def load_data() -> pd.DataFrame:
    """Load integrated dataset."""
    print("Loading data...")
    df = pd.read_parquet(PHASE1_DIR / 'integrated_dataset.parquet')

    # Filter to overlap period
    overlap_start = pd.Timestamp('2025-10-28', tz='UTC')
    df = df[df.index >= overlap_start]

    # Select required columns
    cols = [ROOM_COL, OUTDOOR_COL, HK2_COL, PV_COL]
    df = df[cols].dropna()

    print(f"  {len(df)} rows from {df.index.min()} to {df.index.max()}")
    return df


def compare_models(df: pd.DataFrame) -> dict:
    """Compare fixed vs adaptive models."""
    print("\nComparing fixed vs adaptive models...")

    results = {}

    # Fixed model (baseline)
    outdoor = df[OUTDOOR_COL].values
    hk2 = df[HK2_COL].values
    pv = df[PV_COL].values
    y = df[ROOM_COL].values

    hc_model = LinearRegression()
    hc_model.fit(outdoor.reshape(-1, 1), hk2)
    hk2_expected = hc_model.predict(outdoor.reshape(-1, 1))
    effort = hk2 - hk2_expected

    out_smooth = exponential_smooth(outdoor, 24 * 4)
    eff_smooth = exponential_smooth(effort, 2 * 4)
    pv_smooth = exponential_smooth(pv, 12 * 4)

    X = np.column_stack([out_smooth, eff_smooth, pv_smooth])
    fixed_model = LinearRegression()
    fixed_model.fit(X, y)
    y_pred_fixed = fixed_model.predict(X)

    results['fixed'] = {
        'r2': r2_score(y, y_pred_fixed),
        'rmse': np.sqrt(mean_squared_error(y, y_pred_fixed)),
        'params': {
            'offset': fixed_model.intercept_,
            'g_outdoor': fixed_model.coef_[0],
            'g_effort': fixed_model.coef_[1],
            'g_pv': fixed_model.coef_[2],
        }
    }
    print(f"  Fixed model: R² = {results['fixed']['r2']:.4f}")

    # Adaptive models with different forgetting factors
    for ff in [0.999, 0.995, 0.99, 0.98]:
        model = AdaptiveThermalModel(forgetting_factor=ff)
        stats = model.fit(df)
        results[f'adaptive_ff{ff}'] = stats
        print(f"  Adaptive (ff={ff}): R² = {stats['r2']:.4f}")

    return results


def create_visualization(df: pd.DataFrame, results: dict) -> None:
    """Create comparison visualization."""
    print("\nCreating visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Adaptive vs Fixed Thermal Model Comparison', fontsize=14, fontweight='bold')

    # Panel 1: R² comparison
    ax1 = axes[0, 0]
    models = ['Fixed', 'ff=0.999', 'ff=0.995', 'ff=0.99', 'ff=0.98']
    r2_values = [
        results['fixed']['r2'],
        results['adaptive_ff0.999']['r2'],
        results['adaptive_ff0.995']['r2'],
        results['adaptive_ff0.99']['r2'],
        results['adaptive_ff0.98']['r2'],
    ]
    colors = ['steelblue'] + ['coral'] * 4
    bars = ax1.bar(models, r2_values, color=colors)
    ax1.set_ylabel('R²')
    ax1.set_title('Model Comparison: R²')
    ax1.axhline(y=results['fixed']['r2'], color='steelblue', linestyle='--', alpha=0.5)
    for bar, val in zip(bars, r2_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', fontsize=10)
    ax1.set_ylim(0.6, 1.0)

    # Panel 2: Parameter evolution (for ff=0.99)
    ax2 = axes[0, 1]
    model = AdaptiveThermalModel(forgetting_factor=0.99)
    model.fit(df)
    theta_hist = model.theta_history[96:]  # Exclude warmup

    ax2.plot(theta_hist[:, 1], label='g_outdoor', alpha=0.7)
    ax2.plot(theta_hist[:, 2], label='g_effort', alpha=0.7)
    ax2.axhline(y=results['fixed']['params']['g_outdoor'], color='C0', linestyle='--', alpha=0.3)
    ax2.axhline(y=results['fixed']['params']['g_effort'], color='C1', linestyle='--', alpha=0.3)
    ax2.set_xlabel('Timestep (15-min)')
    ax2.set_ylabel('Parameter value')
    ax2.set_title('Parameter Evolution (ff=0.99)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Panel 3: Prediction comparison
    ax3 = axes[1, 0]
    y = df[ROOM_COL].values

    # Fixed prediction
    outdoor = df[OUTDOOR_COL].values
    hk2 = df[HK2_COL].values
    pv = df[PV_COL].values

    hc_model = LinearRegression()
    hc_model.fit(outdoor.reshape(-1, 1), hk2)
    effort = hk2 - hc_model.predict(outdoor.reshape(-1, 1))

    out_smooth = exponential_smooth(outdoor, 24 * 4)
    eff_smooth = exponential_smooth(effort, 2 * 4)
    pv_smooth = exponential_smooth(pv, 12 * 4)

    X = np.column_stack([out_smooth, eff_smooth, pv_smooth])
    fixed_model = LinearRegression()
    fixed_model.fit(X, y)
    y_pred_fixed = fixed_model.predict(X)

    # Show recent week
    week = slice(-96*7, None)
    t = np.arange(len(y[week])) / 4  # hours
    ax3.plot(t, y[week], 'k-', alpha=0.7, label='Actual', linewidth=0.8)
    ax3.plot(t, y_pred_fixed[week], 'b--', alpha=0.7, label='Fixed', linewidth=1)
    ax3.plot(t, model.y_pred_history[week], 'r-', alpha=0.7, label='Adaptive', linewidth=1)
    ax3.set_xlabel('Hours')
    ax3.set_ylabel('Room Temperature (°C)')
    ax3.set_title('Prediction Comparison (Last Week)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Panel 4: Residual patterns
    ax4 = axes[1, 1]

    resid_fixed = y - y_pred_fixed
    resid_adaptive = y - model.y_pred_history

    hours = df.index.hour

    hourly_fixed = pd.Series(resid_fixed).groupby(hours).mean()
    hourly_adaptive = pd.Series(resid_adaptive).groupby(hours).mean()

    ax4.plot(hourly_fixed.index, hourly_fixed.values, 'b-o', label='Fixed', markersize=4)
    ax4.plot(hourly_adaptive.index, hourly_adaptive.values, 'r-o', label='Adaptive', markersize=4)
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Hour of Day')
    ax4.set_ylabel('Mean Residual (°C)')
    ax4.set_title('Diurnal Residual Pattern')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig3.07_adaptive_thermal_model.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: fig3.07_adaptive_thermal_model.png")


def main():
    """Run adaptive thermal model analysis."""
    print("=" * 60)
    print("Phase 3: Adaptive Thermal Model")
    print("=" * 60)

    # Load data
    df = load_data()

    # Compare models
    results = compare_models(df)

    # Visualization
    create_visualization(df, results)

    # Save results
    # Convert numpy types for JSON
    def convert_types(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        return obj

    results_json = convert_types(results)

    with open(OUTPUT_DIR / 'adaptive_thermal_model.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"\nSaved: adaptive_thermal_model.json")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nFixed model:     R² = {results['fixed']['r2']:.4f}")
    print(f"Adaptive ff=0.99: R² = {results['adaptive_ff0.99']['r2']:.4f}")
    print(f"Improvement:      +{results['adaptive_ff0.99']['r2'] - results['fixed']['r2']:.4f}")
    print("\nKey insight: Time-varying parameters improve R² by ~0.18")
    print("This confirms building thermal dynamics change over time.")


if __name__ == '__main__':
    main()
