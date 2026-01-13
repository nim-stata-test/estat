#!/usr/bin/env python3
"""
Phase 3, Step 1b: Two-State Grey-Box Thermal Model

A physics-based discrete-time state-space model with:
- State 1: Buffer tank temperature (intermediate thermal storage)
- State 2: Room temperature (comfort objective)

Energy balance equations:
  T_buffer[k+1] = T_buffer[k] + (dt/tau_buf) * [(T_HK2[k] - T_buffer[k]) - r_emit*(T_buffer[k] - T_room[k])]
  T_room[k+1] = T_room[k] + (dt/tau_room) * [r_heat*(T_buffer[k] - T_room[k]) - (T_room[k] - T_out[k])] + k_solar*PV[k]

Parameters:
  tau_buf: Buffer tank time constant (hours)
  tau_room: Building time constant (hours)
  r_emit: Emitter/HP coupling ratio
  r_heat: Heat transfer ratio
  k_solar: Solar gain coefficient (K/kWh)
  c_offset: Temperature offset (K)

Key differences from transfer function model (01_thermal_model.py):
- Explicitly models buffer tank as intermediate thermal mass
- Uses state-space formulation with physical parameter interpretation
- Constrained optimization ensures physical plausibility
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import least_squares
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import json
import warnings
warnings.filterwarnings('ignore')

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / 'output' / 'phase1'
OUTPUT_DIR = PROJECT_ROOT / 'output' / 'phase3'
OUTPUT_DIR.mkdir(exist_ok=True)

# Sensor columns
BUFFER_COL = 'stiebel_eltron_isg_actual_temperature_buffer'
HK2_COL = 'wp_anlage_hk2_ist'
ROOM_COL = 'davis_inside_temperature'
OUTDOOR_COL = 'stiebel_eltron_isg_outdoor_temperature'
PV_COL = 'pv_generation_kwh'

# Time step (15 minutes = 0.25 hours)
DT_HOURS = 0.25

# Parameter bounds [lower, upper]
PARAM_BOUNDS = {
    'tau_buf': (0.5, 4.0),      # Buffer time constant (hours)
    'tau_room': (12.0, 72.0),   # Building time constant (hours)
    'r_emit': (0.1, 3.0),       # Emitter coupling ratio
    'r_heat': (0.1, 3.0),       # Heat transfer ratio
    'k_solar': (0.0, 2.0),      # Solar gain (K/kWh)
    'c_offset': (-3.0, 3.0),    # Temperature offset (K)
}

PARAM_NAMES = list(PARAM_BOUNDS.keys())


def load_data() -> pd.DataFrame:
    """Load integrated dataset for thermal modeling."""
    print("Loading data for grey-box thermal model...")

    df = pd.read_parquet(PROCESSED_DIR / 'integrated_dataset.parquet')
    df.index = pd.to_datetime(df.index)

    print(f"  Dataset: {len(df):,} rows ({df.index.min().date()} to {df.index.max().date()})")

    return df


def prepare_data(df: pd.DataFrame) -> tuple:
    """
    Prepare aligned data for state-space model.

    Returns:
        x_obs: Observed states [T_buffer, T_room] shape (n, 2)
        u_inputs: Inputs [T_HK2, T_outdoor, PV] shape (n, 3)
        timestamps: DatetimeIndex
    """
    print("\nPreparing data...")

    required = [BUFFER_COL, HK2_COL, ROOM_COL, OUTDOOR_COL, PV_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Extract and align data
    data = df[required].copy()

    # Forward fill small gaps (up to 1 hour = 4 steps)
    data = data.ffill(limit=4)

    # Drop remaining NaN rows
    data = data.dropna()

    print(f"  Valid observations: {len(data):,}")
    print(f"  Period: {data.index.min().date()} to {data.index.max().date()}")

    x_obs = data[[BUFFER_COL, ROOM_COL]].values
    u_inputs = data[[HK2_COL, OUTDOOR_COL, PV_COL]].values
    timestamps = data.index

    return x_obs, u_inputs, timestamps


def simulate_forward(params: np.ndarray, x0: np.ndarray, u_inputs: np.ndarray,
                     dt: float = DT_HOURS) -> np.ndarray:
    """
    Forward simulate the two-state thermal model (recursive).

    Note: This compounds errors over time. For fitting, use one_step_predict().

    Args:
        params: [tau_buf, tau_room, r_emit, r_heat, k_solar, c_offset]
        x0: Initial state [T_buffer_0, T_room_0]
        u_inputs: Inputs [T_HK2, T_outdoor, PV] shape (n, 3)
        dt: Time step in hours

    Returns:
        x_pred: Predicted states shape (n, 2)
    """
    tau_buf, tau_room, r_emit, r_heat, k_solar, c_offset = params

    n = len(u_inputs)
    x_pred = np.zeros((n, 2))
    x_pred[0] = x0

    for k in range(n - 1):
        T_buf = x_pred[k, 0]
        T_room = x_pred[k, 1]
        T_hk2, T_out, pv = u_inputs[k]

        # Buffer dynamics: heat from HP, heat to room
        dT_buf = (dt / tau_buf) * ((T_hk2 - T_buf) - r_emit * (T_buf - T_room))

        # Room dynamics: heat from buffer, heat loss to outdoor, solar gain
        dT_room = (dt / tau_room) * (r_heat * (T_buf - T_room) - (T_room - T_out)) + k_solar * pv

        x_pred[k + 1, 0] = T_buf + dT_buf
        x_pred[k + 1, 1] = T_room + dT_room + c_offset * dt  # Small drift correction

    return x_pred


def one_step_predict(params: np.ndarray, x_obs: np.ndarray, u_inputs: np.ndarray,
                     dt: float = DT_HOURS) -> np.ndarray:
    """
    One-step-ahead prediction using observed states.

    This is more robust than forward simulation as it doesn't compound errors.
    Each prediction uses the actual observed state from the previous step.

    Args:
        params: [tau_buf, tau_room, r_emit, r_heat, k_solar, c_offset]
        x_obs: Observed states [T_buffer, T_room] shape (n, 2)
        u_inputs: Inputs [T_HK2, T_outdoor, PV] shape (n, 3)
        dt: Time step in hours

    Returns:
        x_pred: One-step-ahead predictions shape (n, 2)
    """
    tau_buf, tau_room, r_emit, r_heat, k_solar, c_offset = params

    n = len(u_inputs)
    x_pred = np.zeros((n, 2))
    x_pred[0] = x_obs[0]  # First prediction = first observation

    for k in range(n - 1):
        # Use OBSERVED states for prediction (key difference from forward sim)
        T_buf = x_obs[k, 0]
        T_room = x_obs[k, 1]
        T_hk2, T_out, pv = u_inputs[k]

        # Buffer dynamics: heat from HP, heat to room
        dT_buf = (dt / tau_buf) * ((T_hk2 - T_buf) - r_emit * (T_buf - T_room))

        # Room dynamics: heat from buffer, heat loss to outdoor, solar gain
        dT_room = (dt / tau_room) * (r_heat * (T_buf - T_room) - (T_room - T_out)) + k_solar * pv

        x_pred[k + 1, 0] = T_buf + dT_buf
        x_pred[k + 1, 1] = T_room + dT_room + c_offset * dt

    return x_pred


def residual_function(params: np.ndarray, x_obs: np.ndarray, u_inputs: np.ndarray,
                      dt: float = DT_HOURS, weights: tuple = (0.3, 1.0)) -> np.ndarray:
    """
    Compute weighted residuals between observed and predicted states.

    Uses one-step-ahead prediction (not forward simulation) to avoid error accumulation.

    Args:
        params: Model parameters
        x_obs: Observed states shape (n, 2)
        u_inputs: Inputs shape (n, 3)
        dt: Time step
        weights: (w_buffer, w_room) - weight room more heavily

    Returns:
        Flat array of residuals
    """
    # Use one-step prediction for fitting (more robust than forward simulation)
    x_pred = one_step_predict(params, x_obs, u_inputs, dt)

    # Weighted residuals (exclude first point which is identity)
    resid_buf = weights[0] * (x_obs[1:, 0] - x_pred[1:, 0])
    resid_room = weights[1] * (x_obs[1:, 1] - x_pred[1:, 1])

    return np.concatenate([resid_buf, resid_room])


def estimate_parameters(x_obs: np.ndarray, u_inputs: np.ndarray,
                        n_restarts: int = 5, seed: int = 42) -> dict:
    """
    Estimate model parameters using constrained least squares with multi-start.

    Args:
        x_obs: Observed states
        u_inputs: Input signals
        n_restarts: Number of random restarts
        seed: Random seed for reproducibility

    Returns:
        dict with best parameters, covariance, and fit statistics
    """
    print("\nEstimating parameters (constrained least squares)...")

    rng = np.random.RandomState(seed)

    # Parameter bounds
    lower = np.array([PARAM_BOUNDS[p][0] for p in PARAM_NAMES])
    upper = np.array([PARAM_BOUNDS[p][1] for p in PARAM_NAMES])

    best_result = None
    best_cost = np.inf

    # Initial guess from physics
    p0_default = np.array([
        1.0,   # tau_buf: 1 hour
        24.0,  # tau_room: 24 hours
        1.0,   # r_emit: unity
        1.0,   # r_heat: unity
        0.5,   # k_solar: moderate
        0.0,   # c_offset: no offset
    ])

    for i in range(n_restarts):
        if i == 0:
            p0 = p0_default
        else:
            # Latin hypercube sampling
            p0 = lower + (upper - lower) * rng.rand(len(lower))

        try:
            result = least_squares(
                residual_function,
                p0,
                args=(x_obs, u_inputs, DT_HOURS),
                bounds=(lower, upper),
                method='trf',
                ftol=1e-8,
                xtol=1e-8,
                max_nfev=1000,
                verbose=0
            )

            if result.cost < best_cost:
                best_cost = result.cost
                best_result = result
                print(f"  Restart {i + 1}: cost = {result.cost:.2f} (new best)")
            else:
                print(f"  Restart {i + 1}: cost = {result.cost:.2f}")

        except Exception as e:
            print(f"  Restart {i + 1}: failed ({e})")

    if best_result is None:
        raise RuntimeError("All optimization restarts failed")

    # Extract parameters
    params = best_result.x
    param_dict = {name: params[i] for i, name in enumerate(PARAM_NAMES)}

    # Compute Jacobian-based covariance estimate
    J = best_result.jac
    try:
        cov = np.linalg.inv(J.T @ J) * (best_result.fun ** 2).mean()
        param_std = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        param_std = np.full(len(params), np.nan)

    # Compute fit statistics using ONE-STEP prediction (what we optimized)
    x_pred_1step = one_step_predict(params, x_obs, u_inputs)

    # Also compute forward simulation for visualization (shows accumulated error)
    x_pred_forward = simulate_forward(params, x_obs[0], u_inputs)

    stats = {
        'params': param_dict,
        'param_std': {name: param_std[i] for i, name in enumerate(PARAM_NAMES)},
        # One-step metrics (what we optimized for)
        'r2_buffer': r2_score(x_obs[1:, 0], x_pred_1step[1:, 0]),
        'r2_room': r2_score(x_obs[1:, 1], x_pred_1step[1:, 1]),
        'rmse_buffer': np.sqrt(mean_squared_error(x_obs[1:, 0], x_pred_1step[1:, 0])),
        'rmse_room': np.sqrt(mean_squared_error(x_obs[1:, 1], x_pred_1step[1:, 1])),
        'mae_buffer': mean_absolute_error(x_obs[1:, 0], x_pred_1step[1:, 0]),
        'mae_room': mean_absolute_error(x_obs[1:, 1], x_pred_1step[1:, 1]),
        'bias_buffer': (x_pred_1step[1:, 0] - x_obs[1:, 0]).mean(),
        'bias_room': (x_pred_1step[1:, 1] - x_obs[1:, 1]).mean(),
        # Forward simulation metrics (for comparison)
        'r2_room_forward': r2_score(x_obs[:, 1], x_pred_forward[:, 1]),
        'rmse_room_forward': np.sqrt(mean_squared_error(x_obs[:, 1], x_pred_forward[:, 1])),
        'n_points': len(x_obs),
        'optimization_cost': best_result.cost,
        'x_pred': x_pred_1step,  # Use 1-step for visualization
        'x_pred_forward': x_pred_forward,  # Also save forward sim
    }

    print(f"\nBest fit:")
    for name in PARAM_NAMES:
        std = stats['param_std'][name]
        print(f"  {name}: {param_dict[name]:.3f} +/- {std:.3f}")
    print(f"\n  R² (buffer): {stats['r2_buffer']:.3f}")
    print(f"  R² (room):   {stats['r2_room']:.3f}")
    print(f"  RMSE (room): {stats['rmse_room']:.3f}°C")

    return stats


def train_test_split(x_obs: np.ndarray, u_inputs: np.ndarray,
                     timestamps: pd.DatetimeIndex, test_fraction: float = 0.33) -> dict:
    """
    Split data temporally for validation.

    Returns:
        dict with train/test arrays and indices
    """
    n = len(x_obs)
    split_idx = int(n * (1 - test_fraction))

    return {
        'x_train': x_obs[:split_idx],
        'u_train': u_inputs[:split_idx],
        't_train': timestamps[:split_idx],
        'x_test': x_obs[split_idx:],
        'u_test': u_inputs[split_idx:],
        't_test': timestamps[split_idx:],
        'split_idx': split_idx,
    }


def validate_model(params: np.ndarray, split: dict) -> dict:
    """
    Validate model on held-out test data.

    Uses one-step prediction to measure predictive accuracy.

    Returns:
        dict with test set metrics
    """
    # One-step prediction on test data (uses observed states)
    x_pred_1step = one_step_predict(params, split['x_test'], split['u_test'])

    # Forward simulation (uses only initial condition - harder test)
    x0_test = split['x_train'][-1]
    x_pred_forward = simulate_forward(params, x0_test, split['u_test'])

    return {
        # One-step metrics (realistic for control applications)
        'r2_buffer_test': r2_score(split['x_test'][1:, 0], x_pred_1step[1:, 0]),
        'r2_room_test': r2_score(split['x_test'][1:, 1], x_pred_1step[1:, 1]),
        'rmse_buffer_test': np.sqrt(mean_squared_error(split['x_test'][1:, 0], x_pred_1step[1:, 0])),
        'rmse_room_test': np.sqrt(mean_squared_error(split['x_test'][1:, 1], x_pred_1step[1:, 1])),
        'mae_room_test': mean_absolute_error(split['x_test'][1:, 1], x_pred_1step[1:, 1]),
        'bias_room_test': (x_pred_1step[1:, 1] - split['x_test'][1:, 1]).mean(),
        # Forward simulation metrics (harder test)
        'r2_room_test_forward': r2_score(split['x_test'][:, 1], x_pred_forward[:, 1]),
        'rmse_room_test_forward': np.sqrt(mean_squared_error(split['x_test'][:, 1], x_pred_forward[:, 1])),
        'x_pred_test': x_pred_1step,
        'x_pred_test_forward': x_pred_forward,
    }


def compute_residual_diagnostics(residuals: np.ndarray, max_lag: int = 96) -> dict:
    """
    Compute residual diagnostics (autocorrelation, normality).

    Args:
        residuals: Prediction residuals
        max_lag: Maximum lag for autocorrelation (96 = 24 hours at 15-min)

    Returns:
        dict with diagnostic statistics
    """
    n = len(residuals)

    # Autocorrelation
    acf = np.zeros(max_lag)
    for lag in range(max_lag):
        if lag < n:
            acf[lag] = np.corrcoef(residuals[:-lag - 1], residuals[lag + 1:])[0, 1] if lag > 0 else 1.0

    # Find lag at which autocorrelation drops below 0.2
    decorr_lag = np.argmax(np.abs(acf) < 0.2) if any(np.abs(acf) < 0.2) else max_lag

    # Durbin-Watson statistic
    dw = np.sum(np.diff(residuals) ** 2) / np.sum(residuals ** 2)

    return {
        'acf': acf,
        'decorr_lag_steps': decorr_lag,
        'decorr_lag_hours': decorr_lag * DT_HOURS,
        'durbin_watson': dw,
        'residual_std': residuals.std(),
        'residual_mean': residuals.mean(),
    }


def plot_greybox_results(stats: dict, validation: dict, split: dict,
                         timestamps: pd.DatetimeIndex, diagnostics: dict) -> None:
    """Create 4-panel visualization of grey-box model results."""
    print("\nCreating grey-box model visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    params = np.array([stats['params'][p] for p in PARAM_NAMES])
    x_pred = stats['x_pred']
    x_obs = np.vstack([split['x_train'], split['x_test']])

    # Panel 1: State trajectories (last 2 weeks)
    ax1 = axes[0, 0]
    recent_days = 14
    recent_mask = timestamps >= timestamps.max() - pd.Timedelta(days=recent_days)
    t_recent = timestamps[recent_mask]

    ax1.plot(t_recent, x_obs[recent_mask, 1], 'b-', linewidth=1, alpha=0.8, label='Actual (room)')
    ax1.plot(t_recent, x_pred[recent_mask, 1], 'r--', linewidth=1, alpha=0.8, label='Predicted (room)')
    ax1.fill_between(t_recent, x_obs[recent_mask, 0], alpha=0.3, color='orange', label='Buffer tank')

    ax1.set_xlabel('Date')
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_title(f'State Trajectories (Last {recent_days} Days)')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Add train/test boundary
    split_time = timestamps[split['split_idx']]
    if split_time >= t_recent.min():
        ax1.axvline(split_time, color='green', linestyle=':', linewidth=2, label='Train/Test split')

    # Panel 2: Actual vs Predicted scatter
    ax2 = axes[0, 1]
    step = max(1, len(x_obs) // 1000)
    ax2.scatter(x_obs[::step, 1], x_pred[::step, 1], alpha=0.4, s=10, c='blue')
    temp_range = [x_obs[:, 1].min() - 0.5, x_obs[:, 1].max() + 0.5]
    ax2.plot(temp_range, temp_range, 'r--', linewidth=1.5, label='Perfect fit')

    ax2.set_xlabel('Actual Room Temperature (°C)')
    ax2.set_ylabel('Predicted Room Temperature (°C)')
    ax2.set_title(f'Room: R²={stats["r2_room"]:.3f}, RMSE={stats["rmse_room"]:.2f}°C')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')

    # Panel 3: Residual histogram and autocorrelation
    ax3 = axes[1, 0]
    residuals = x_obs[:, 1] - x_pred[:, 1]

    # Histogram
    ax3.hist(residuals, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='white')
    ax3.axvline(0, color='red', linestyle='--', linewidth=1.5)

    # Add normal curve
    x_norm = np.linspace(residuals.min(), residuals.max(), 100)
    from scipy.stats import norm
    ax3.plot(x_norm, norm.pdf(x_norm, residuals.mean(), residuals.std()),
             'r-', linewidth=2, label=f'Normal (μ={residuals.mean():.2f}, σ={residuals.std():.2f})')

    ax3.set_xlabel('Residual (°C)')
    ax3.set_ylabel('Density')
    ax3.set_title(f'Residual Distribution (DW={diagnostics["durbin_watson"]:.2f})')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Model comparison bar chart
    ax4 = axes[1, 1]

    # Load transfer function results for comparison
    tf_results_file = OUTPUT_DIR / 'thermal_model_results.csv'
    if tf_results_file.exists():
        tf_df = pd.read_csv(tf_results_file)
        tf_r2 = tf_df[tf_df['room'] == 'davis_inside']['r2'].values[0]
        tf_rmse = tf_df[tf_df['room'] == 'davis_inside']['rmse'].values[0]
    else:
        tf_r2 = 0.68  # Default from CLAUDE.md
        tf_rmse = 0.50

    models = ['Transfer Function\n(current)', 'Grey-Box\n(new)']
    r2_values = [tf_r2, stats['r2_room']]
    rmse_values = [tf_rmse, stats['rmse_room']]

    x_pos = np.arange(len(models))
    width = 0.35

    bars1 = ax4.bar(x_pos - width / 2, r2_values, width, label='R²', color='steelblue')
    ax4.bar_label(bars1, fmt='%.3f', padding=3, fontsize=9)

    ax4_twin = ax4.twinx()
    bars2 = ax4_twin.bar(x_pos + width / 2, rmse_values, width, label='RMSE (°C)', color='coral')
    ax4_twin.bar_label(bars2, fmt='%.2f', padding=3, fontsize=9)

    ax4.set_ylabel('R²', color='steelblue')
    ax4_twin.set_ylabel('RMSE (°C)', color='coral')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(models)
    ax4.set_title('Model Comparison')
    ax4.set_ylim(0, 1.0)
    ax4_twin.set_ylim(0, max(rmse_values) * 1.5)
    ax4.grid(True, alpha=0.3, axis='y')

    # Add improvement annotation
    r2_improvement = (stats['r2_room'] - tf_r2) / tf_r2 * 100
    ax4.annotate(f'{r2_improvement:+.1f}% R²', xy=(1, stats['r2_room']),
                 xytext=(1.3, stats['r2_room']), fontsize=10,
                 arrowprops=dict(arrowstyle='->', color='green'),
                 color='green', fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig17b_greybox_model.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: fig17b_greybox_model.png")


def generate_report(stats: dict, validation: dict, diagnostics: dict) -> str:
    """Generate HTML report section for grey-box thermal model."""

    params = stats['params']
    param_std = stats['param_std']

    # Parameter table
    param_rows = ""
    for name in PARAM_NAMES:
        bounds = PARAM_BOUNDS[name]
        param_rows += f"""
        <tr>
            <td><code>{name}</code></td>
            <td>{params[name]:.3f}</td>
            <td>+/- {param_std[name]:.3f}</td>
            <td>[{bounds[0]}, {bounds[1]}]</td>
        </tr>
        """

    html = f"""
    <section id="greybox-thermal-model">
    <h2>3.1b Grey-Box Thermal Model</h2>

    <h3>Model Formulation</h3>
    <p>A physics-based discrete-time state-space model with two states:</p>

    <h4>State Variables</h4>
    <ul>
        <li><strong>T_buffer</strong>: Buffer tank temperature (intermediate thermal storage)</li>
        <li><strong>T_room</strong>: Room/indoor temperature (comfort objective)</li>
    </ul>

    <h4>Discrete-Time Equations (dt = 15 min)</h4>
    <pre>
T_buffer[k+1] = T_buffer[k] + (dt/tau_buf) × [(T_HK2[k] - T_buffer[k]) - r_emit × (T_buffer[k] - T_room[k])]

T_room[k+1] = T_room[k] + (dt/tau_room) × [r_heat × (T_buffer[k] - T_room[k]) - (T_room[k] - T_outdoor[k])] + k_solar × PV[k]
    </pre>

    <h3>Estimated Parameters</h3>
    <table>
        <tr>
            <th>Parameter</th>
            <th>Value</th>
            <th>Std Error</th>
            <th>Bounds</th>
        </tr>
        {param_rows}
    </table>

    <h3>Physical Interpretation</h3>
    <ul>
        <li><strong>tau_buf = {params['tau_buf']:.2f}h</strong>: Buffer tank responds to heat pump input
            with time constant of ~{params['tau_buf'] * 60:.0f} minutes</li>
        <li><strong>tau_room = {params['tau_room']:.1f}h</strong>: Building thermal mass gives
            ~{params['tau_room']:.0f}-hour time constant for room temperature</li>
        <li><strong>r_emit = {params['r_emit']:.2f}</strong>: Ratio of heat transfer from buffer to room
            vs heat input from heat pump</li>
        <li><strong>r_heat = {params['r_heat']:.2f}</strong>: Ratio of heat transfer from buffer to room
            vs heat loss to outdoors</li>
        <li><strong>k_solar = {params['k_solar']:.3f} K/kWh</strong>: Room gains {params['k_solar']:.2f}°C
            per kWh of PV generation (proxy for solar irradiance)</li>
    </ul>

    <h3>Fit Statistics (Training Set)</h3>
    <table>
        <tr>
            <th>Metric</th>
            <th>Buffer Tank</th>
            <th>Room</th>
        </tr>
        <tr>
            <td>R²</td>
            <td>{stats['r2_buffer']:.3f}</td>
            <td><strong>{stats['r2_room']:.3f}</strong></td>
        </tr>
        <tr>
            <td>RMSE</td>
            <td>{stats['rmse_buffer']:.2f}°C</td>
            <td><strong>{stats['rmse_room']:.2f}°C</strong></td>
        </tr>
        <tr>
            <td>MAE</td>
            <td>{stats['mae_buffer']:.2f}°C</td>
            <td>{stats['mae_room']:.2f}°C</td>
        </tr>
        <tr>
            <td>Bias</td>
            <td>{stats['bias_buffer']:+.3f}°C</td>
            <td>{stats['bias_room']:+.3f}°C</td>
        </tr>
    </table>

    <h3>Validation (Test Set)</h3>
    <table>
        <tr>
            <th>Metric</th>
            <th>Buffer Tank</th>
            <th>Room</th>
        </tr>
        <tr>
            <td>R²</td>
            <td>{validation['r2_buffer_test']:.3f}</td>
            <td><strong>{validation['r2_room_test']:.3f}</strong></td>
        </tr>
        <tr>
            <td>RMSE</td>
            <td>{validation['rmse_buffer_test']:.2f}°C</td>
            <td><strong>{validation['rmse_room_test']:.2f}°C</strong></td>
        </tr>
        <tr>
            <td>Bias</td>
            <td>—</td>
            <td>{validation['bias_room_test']:+.3f}°C</td>
        </tr>
    </table>

    <h3>Residual Diagnostics</h3>
    <ul>
        <li><strong>Durbin-Watson</strong>: {diagnostics['durbin_watson']:.2f}
            (ideal = 2.0; &lt;1.5 indicates positive autocorrelation)</li>
        <li><strong>Decorrelation lag</strong>: {diagnostics['decorr_lag_hours']:.1f} hours
            (residuals become uncorrelated after this time)</li>
        <li><strong>Residual std</strong>: {diagnostics['residual_std']:.3f}°C</li>
    </ul>

    <h3>Comparison with Transfer Function Model</h3>
    <p>The grey-box model provides:</p>
    <ul>
        <li><strong>Explicit buffer tank modeling</strong>: Captures intermediate thermal storage dynamics</li>
        <li><strong>Physical parameters</strong>: Time constants and heat transfer ratios with clear interpretation</li>
        <li><strong>Constrained estimation</strong>: Parameters bounded to physically plausible ranges</li>
    </ul>

    <figure>
        <img src="fig17b_greybox_model.png" alt="Grey-Box Thermal Model Results">
        <figcaption><strong>Figure 17b:</strong> Grey-box thermal model: state trajectories (top-left),
        actual vs predicted scatter (top-right), residual distribution (bottom-left),
        model comparison (bottom-right).</figcaption>
    </figure>
    </section>
    """

    return html


def main():
    """Main function for grey-box thermal model estimation."""
    print("=" * 60)
    print("Phase 3, Step 1b: Grey-Box Thermal Model")
    print("=" * 60)

    # Load and prepare data
    df = load_data()
    x_obs, u_inputs, timestamps = prepare_data(df)

    # Train/test split
    split = train_test_split(x_obs, u_inputs, timestamps, test_fraction=0.33)
    print(f"\nTrain/test split:")
    print(f"  Training: {len(split['x_train']):,} points ({split['t_train'].min().date()} to {split['t_train'].max().date()})")
    print(f"  Testing:  {len(split['x_test']):,} points ({split['t_test'].min().date()} to {split['t_test'].max().date()})")

    # Estimate parameters on training data
    stats = estimate_parameters(split['x_train'], split['u_train'], n_restarts=5)

    # Validate on test data
    params = np.array([stats['params'][p] for p in PARAM_NAMES])
    validation = validate_model(params, split)
    print(f"\nTest set performance:")
    print(f"  R² (room):   {validation['r2_room_test']:.3f}")
    print(f"  RMSE (room): {validation['rmse_room_test']:.3f}°C")

    # Re-fit on full dataset for final model
    print("\n" + "=" * 60)
    print("Re-fitting on full dataset...")
    stats_full = estimate_parameters(x_obs, u_inputs, n_restarts=5)

    # Compute residual diagnostics
    x_pred_full = stats_full['x_pred']
    residuals = x_obs[:, 1] - x_pred_full[:, 1]
    diagnostics = compute_residual_diagnostics(residuals)

    # Create full dataset split for visualization
    split_full = train_test_split(x_obs, u_inputs, timestamps, test_fraction=0.33)
    stats_full['x_pred'] = simulate_forward(
        np.array([stats_full['params'][p] for p in PARAM_NAMES]),
        x_obs[0], u_inputs
    )

    # Visualization
    plot_greybox_results(stats_full, validation, split_full, timestamps, diagnostics)

    # Save results
    results = {
        'params': stats_full['params'],
        'param_std': stats_full['param_std'],
        'param_bounds': PARAM_BOUNDS,
        'fit_stats': {
            'r2_buffer': stats_full['r2_buffer'],
            'r2_room': stats_full['r2_room'],
            'rmse_buffer': stats_full['rmse_buffer'],
            'rmse_room': stats_full['rmse_room'],
            'mae_buffer': stats_full['mae_buffer'],
            'mae_room': stats_full['mae_room'],
            'bias_buffer': stats_full['bias_buffer'],
            'bias_room': stats_full['bias_room'],
            'n_points': stats_full['n_points'],
        },
        'validation': {
            'r2_buffer_test': validation['r2_buffer_test'],
            'r2_room_test': validation['r2_room_test'],
            'rmse_buffer_test': validation['rmse_buffer_test'],
            'rmse_room_test': validation['rmse_room_test'],
        },
        'diagnostics': {
            'durbin_watson': diagnostics['durbin_watson'],
            'decorr_lag_hours': diagnostics['decorr_lag_hours'],
            'residual_std': diagnostics['residual_std'],
        },
        'sensors': {
            'buffer': BUFFER_COL,
            'room': ROOM_COL,
            'hk2': HK2_COL,
            'outdoor': OUTDOOR_COL,
            'pv': PV_COL,
        },
        'time_step_hours': DT_HOURS,
    }

    with open(OUTPUT_DIR / 'greybox_model_params.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: greybox_model_params.json")

    # Save predictions CSV
    pred_df = pd.DataFrame({
        'timestamp': timestamps,
        'T_buffer_actual': x_obs[:, 0],
        'T_buffer_pred': stats_full['x_pred'][:, 0],
        'T_room_actual': x_obs[:, 1],
        'T_room_pred': stats_full['x_pred'][:, 1],
        'T_HK2': u_inputs[:, 0],
        'T_outdoor': u_inputs[:, 1],
        'PV_kwh': u_inputs[:, 2],
    })
    pred_df.to_csv(OUTPUT_DIR / 'greybox_model_results.csv', index=False)
    print("Saved: greybox_model_results.csv")

    # Generate HTML report
    report_html = generate_report(stats_full, validation, diagnostics)
    with open(OUTPUT_DIR / 'greybox_report_section.html', 'w') as f:
        f.write(report_html)
    print("Saved: greybox_report_section.html")

    # Summary
    print("\n" + "=" * 60)
    print("GREY-BOX MODEL SUMMARY")
    print("=" * 60)

    print(f"\nPhysical Parameters:")
    print(f"  Buffer time constant (tau_buf):  {stats_full['params']['tau_buf']:.2f} h")
    print(f"  Building time constant (tau_room): {stats_full['params']['tau_room']:.1f} h")
    print(f"  Emitter coupling (r_emit):       {stats_full['params']['r_emit']:.2f}")
    print(f"  Heat transfer ratio (r_heat):    {stats_full['params']['r_heat']:.2f}")
    print(f"  Solar gain (k_solar):            {stats_full['params']['k_solar']:.3f} K/kWh")

    print(f"\nFit Quality:")
    print(f"  Training R² (room): {stats_full['r2_room']:.3f}")
    print(f"  Test R² (room):     {validation['r2_room_test']:.3f}")
    print(f"  RMSE (room):        {stats_full['rmse_room']:.3f}°C")

    print(f"\nResidual Diagnostics:")
    print(f"  Durbin-Watson: {diagnostics['durbin_watson']:.2f}")
    print(f"  Decorr. lag:   {diagnostics['decorr_lag_hours']:.1f} h")


if __name__ == '__main__':
    main()
