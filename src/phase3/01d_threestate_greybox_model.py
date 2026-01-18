#!/usr/bin/env python3
"""
Phase 3, Step 1d: Three-State Grey-Box Thermal Model

Extends the 2-state model by adding building envelope as separate thermal mass.

States:
  1. T_buffer  - Heat pump buffer tank (fast dynamics, ~1-4h)
  2. T_envelope - Building envelope/walls/floor (slow dynamics, ~24-96h)
  3. T_room    - Room air temperature (medium dynamics, ~4-12h)

Physics:
  - HP heats buffer, buffer heats room air
  - Room air exchanges heat with envelope (bidirectional)
  - Both room and envelope lose heat to outdoor
  - Solar gains directly heat room air

The envelope state captures slow thermal mass dynamics that the 2-state
model misses, potentially reducing long-term drift.

Discrete-time equations (semi-implicit):
  T_buf[k+1] = f(T_buf[k], T_room[k], T_HK2[k])
  T_env[k+1] = f(T_env[k], T_room[k], T_out[k])
  T_room[k+1] = f(T_room[k], T_buf[k+1], T_env[k], T_out[k], PV[k])
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize
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
HK2_COL = 'stiebel_eltron_isg_target_temperature_hk_2'
ROOM_COL = 'davis_inside_temperature'
OUTDOOR_COL = 'stiebel_eltron_isg_outdoor_temperature'
PV_COL = 'pv_generation_kwh'

# Time step
DT_HOURS = 0.25  # 15 minutes

# Parameter bounds for 3-state model
PARAM_BOUNDS = {
    # Time constants
    'tau_buf': (1.0, 6.0),       # Buffer: 1-6 hours (fast)
    'tau_env': (24.0, 120.0),    # Envelope: 1-5 days (slow)
    'tau_room': (4.0, 24.0),     # Room air: 4-24 hours (medium)
    # Coupling ratios
    'r_emit': (0.1, 2.0),        # Buffer → room emission
    'r_heat': (0.1, 2.0),        # Buffer heating ratio
    'r_env': (0.1, 2.0),         # Envelope ↔ room coupling
    'r_env_out': (0.05, 1.0),    # Envelope → outdoor loss
    # Gains
    'k_solar': (0.0, 0.5),       # Solar gain (K/kWh)
    'c_offset': (-0.5, 0.5),     # Bias correction
}

PARAM_NAMES = list(PARAM_BOUNDS.keys())

# Stability margin
MAX_EIGENVALUE = 0.998


def load_data() -> pd.DataFrame:
    """Load integrated dataset."""
    print("Loading data for 3-state grey-box thermal model...")
    df = pd.read_parquet(PROCESSED_DIR / 'integrated_dataset.parquet')
    df.index = pd.to_datetime(df.index)
    print(f"  Dataset: {len(df):,} rows ({df.index.min().date()} to {df.index.max().date()})")
    return df


def prepare_data(df: pd.DataFrame) -> tuple:
    """Prepare aligned data for 3-state model."""
    print("\nPreparing data...")

    required = [BUFFER_COL, HK2_COL, ROOM_COL, OUTDOOR_COL, PV_COL]
    data = df[required].copy()
    data = data.ffill(limit=4)
    data = data.dropna()

    print(f"  Valid observations: {len(data):,}")
    print(f"  Period: {data.index.min().date()} to {data.index.max().date()}")

    # Observations: [T_buffer, T_room] (we don't observe T_envelope directly)
    x_obs = data[[BUFFER_COL, ROOM_COL]].values
    u_inputs = data[[HK2_COL, OUTDOOR_COL, PV_COL]].values
    timestamps = data.index

    return x_obs, u_inputs, timestamps


def simulate_forward_3state(params: np.ndarray, x0_2state: np.ndarray, u_inputs: np.ndarray,
                             dt: float = DT_HOURS) -> tuple:
    """
    Forward simulate the 3-state model using semi-implicit integration.

    Args:
        params: Model parameters
        x0_2state: Initial observed state [T_buffer_0, T_room_0]
        u_inputs: Inputs [T_HK2, T_outdoor, PV]
        dt: Time step

    Returns:
        x_pred_3state: Full state [T_buffer, T_envelope, T_room] shape (n, 3)
        x_pred_2state: Observable states [T_buffer, T_room] shape (n, 2)
    """
    tau_buf, tau_env, tau_room, r_emit, r_heat, r_env, r_env_out, k_solar, c_offset = params

    n = len(u_inputs)
    x_3state = np.zeros((n, 3))

    # Initialize: buffer and room from observations, envelope = room (equilibrium assumption)
    x_3state[0, 0] = x0_2state[0]  # T_buffer
    x_3state[0, 1] = x0_2state[1]  # T_envelope (initialize to room temp)
    x_3state[0, 2] = x0_2state[1]  # T_room

    # Pre-compute semi-implicit factors
    alpha_buf = dt / tau_buf
    alpha_env = dt / tau_env
    alpha_room = dt / tau_room

    for k in range(n - 1):
        T_buf_k = x_3state[k, 0]
        T_env_k = x_3state[k, 1]
        T_room_k = x_3state[k, 2]
        T_hk2, T_out, pv = u_inputs[k]

        # Semi-implicit buffer update
        # dT_buf/dt = (1/tau_buf) * [(T_HK2 - T_buf) - r_emit*(T_buf - T_room)]
        denom_buf = 1 + alpha_buf * (1 + r_emit)
        T_buf_new = (T_buf_k + alpha_buf * (T_hk2 + r_emit * T_room_k)) / denom_buf

        # Semi-implicit envelope update
        # dT_env/dt = (1/tau_env) * [r_env*(T_room - T_env) - r_env_out*(T_env - T_out)]
        denom_env = 1 + alpha_env * (r_env + r_env_out)
        T_env_new = (T_env_k + alpha_env * (r_env * T_room_k + r_env_out * T_out)) / denom_env

        # Semi-implicit room update (using updated buffer and old envelope)
        # dT_room/dt = (1/tau_room) * [r_heat*(T_buf - T_room) + r_env*(T_env - T_room) - (T_room - T_out)] + k_solar*PV
        denom_room = 1 + alpha_room * (r_heat + r_env + 1)
        T_room_new = (T_room_k + alpha_room * (r_heat * T_buf_new + r_env * T_env_k + T_out) + k_solar * pv) / denom_room
        T_room_new += c_offset * dt

        x_3state[k + 1, 0] = T_buf_new
        x_3state[k + 1, 1] = T_env_new
        x_3state[k + 1, 2] = T_room_new

    # Extract observable states
    x_2state = x_3state[:, [0, 2]]  # [T_buffer, T_room]

    return x_3state, x_2state


def simulate_rolling_horizon_3state(params: np.ndarray, x_obs: np.ndarray, u_inputs: np.ndarray,
                                     horizon_steps: int = 96, dt: float = DT_HOURS) -> np.ndarray:
    """
    Rolling horizon simulation for 3-state model.

    Resets observable states from observations, but maintains envelope state
    which helps capture slow dynamics.
    """
    tau_buf, tau_env, tau_room, r_emit, r_heat, r_env, r_env_out, k_solar, c_offset = params

    n = len(u_inputs)
    x_pred = np.zeros((n, 2))  # Only [T_buffer, T_room] for comparison
    x_pred[0] = x_obs[0]

    # Full 3-state tracking
    T_buf = x_obs[0, 0]
    T_env = x_obs[0, 1]  # Initialize envelope to room
    T_room = x_obs[0, 1]

    # Pre-compute factors
    alpha_buf = dt / tau_buf
    alpha_env = dt / tau_env
    alpha_room = dt / tau_room

    denom_buf = 1 + alpha_buf * (1 + r_emit)
    denom_env = 1 + alpha_env * (r_env + r_env_out)
    denom_room = 1 + alpha_room * (r_heat + r_env + 1)

    for k in range(n - 1):
        # Reset observable states at horizon boundaries
        if k % horizon_steps == 0:
            T_buf = x_obs[k, 0]
            T_room = x_obs[k, 1]
            # Keep T_env continuous - this is the key difference!

        T_hk2, T_out, pv = u_inputs[k]

        # Semi-implicit updates
        T_buf_new = (T_buf + alpha_buf * (T_hk2 + r_emit * T_room)) / denom_buf
        T_env_new = (T_env + alpha_env * (r_env * T_room + r_env_out * T_out)) / denom_env
        T_room_new = (T_room + alpha_room * (r_heat * T_buf_new + r_env * T_env + T_out) + k_solar * pv) / denom_room
        T_room_new += c_offset * dt

        T_buf = T_buf_new
        T_env = T_env_new
        T_room = T_room_new

        x_pred[k + 1, 0] = T_buf
        x_pred[k + 1, 1] = T_room

    return x_pred


def compute_eigenvalues_3state(params: np.ndarray, dt: float = DT_HOURS) -> tuple:
    """Compute eigenvalues of 3-state transition matrix."""
    tau_buf, tau_env, tau_room, r_emit, r_heat, r_env, r_env_out, k_solar, c_offset = params

    alpha_buf = dt / tau_buf
    alpha_env = dt / tau_env
    alpha_room = dt / tau_room

    # Linearized state transition matrix (approximate)
    # States: [T_buf, T_env, T_room]
    A = np.array([
        [(1 - alpha_buf * (1 + r_emit)) / (1 + alpha_buf * (1 + r_emit)),
         0,
         alpha_buf * r_emit / (1 + alpha_buf * (1 + r_emit))],
        [0,
         (1 - alpha_env * (r_env + r_env_out)) / (1 + alpha_env * (r_env + r_env_out)),
         alpha_env * r_env / (1 + alpha_env * (r_env + r_env_out))],
        [alpha_room * r_heat / (1 + alpha_room * (r_heat + r_env + 1)),
         alpha_room * r_env / (1 + alpha_room * (r_heat + r_env + 1)),
         1 / (1 + alpha_room * (r_heat + r_env + 1))]
    ])

    eigenvalues = np.linalg.eigvals(A)
    max_eig = np.max(np.abs(eigenvalues))
    is_stable = max_eig < 1.0

    return is_stable, max_eig, eigenvalues


def stability_penalty_3state(params: np.ndarray, dt: float = DT_HOURS) -> float:
    """Compute stability penalty for 3-state model."""
    _, max_eig, _ = compute_eigenvalues_3state(params, dt)

    if max_eig < MAX_EIGENVALUE:
        return 0.0
    else:
        return 1000 * np.exp(10 * (max_eig - MAX_EIGENVALUE))


def objective_3state(params: np.ndarray, x_obs: np.ndarray, u_inputs: np.ndarray,
                     dt: float = DT_HOURS, lambda_stability: float = 10.0) -> float:
    """
    Objective function for 3-state model optimization.

    Optimizes for forward simulation performance on room temperature.
    """
    # Check bounds
    lower = np.array([PARAM_BOUNDS[p][0] for p in PARAM_NAMES])
    upper = np.array([PARAM_BOUNDS[p][1] for p in PARAM_NAMES])

    if np.any(params < lower) or np.any(params > upper):
        return 1e10

    # Stability penalty
    stab_penalty = stability_penalty_3state(params, dt)

    # Forward simulation
    try:
        _, x_pred_2state = simulate_forward_3state(params, x_obs[0], u_inputs, dt)
        mse_room = np.mean((x_obs[:, 1] - x_pred_2state[:, 1]) ** 2)
    except Exception:
        return 1e10

    return mse_room + lambda_stability * stab_penalty


def estimate_parameters_3state(x_obs: np.ndarray, u_inputs: np.ndarray,
                                n_restarts: int = 15, seed: int = 42) -> dict:
    """
    Estimate 3-state model parameters with stability constraints.
    """
    print("\nEstimating 3-state model parameters...")

    rng = np.random.RandomState(seed)

    lower = np.array([PARAM_BOUNDS[p][0] for p in PARAM_NAMES])
    upper = np.array([PARAM_BOUNDS[p][1] for p in PARAM_NAMES])
    bounds = [(l, u) for l, u in zip(lower, upper)]

    # Initial guess based on physics
    p0_physics = np.array([
        2.0,    # tau_buf: 2 hours
        48.0,   # tau_env: 2 days
        8.0,    # tau_room: 8 hours
        0.5,    # r_emit
        0.5,    # r_heat
        0.5,    # r_env
        0.2,    # r_env_out
        0.1,    # k_solar
        0.0,    # c_offset
    ])

    best_params = None
    best_obj = np.inf

    print(f"\n  Running {n_restarts} optimization restarts...")

    for i in range(n_restarts):
        if i == 0:
            p0 = p0_physics
        elif i == 1:
            # Try faster room dynamics
            p0 = np.array([1.5, 36.0, 6.0, 0.8, 0.8, 0.3, 0.15, 0.05, 0.0])
        elif i == 2:
            # Try slower envelope
            p0 = np.array([2.5, 96.0, 12.0, 0.4, 0.6, 0.6, 0.3, 0.1, 0.0])
        else:
            # Random initialization
            p0 = lower + (upper - lower) * rng.rand(len(lower))

        try:
            result = minimize(
                objective_3state,
                p0,
                args=(x_obs, u_inputs, DT_HOURS, 10.0),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )

            is_stable, max_eig, _ = compute_eigenvalues_3state(result.x)

            if is_stable and result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x
                print(f"    Restart {i + 1}: MSE = {result.fun:.4f}, max_eig = {max_eig:.4f} (stable) *BEST*")
            elif i < 5:  # Only print first few
                print(f"    Restart {i + 1}: MSE = {result.fun:.4f}, max_eig = {max_eig:.4f} {'(stable)' if is_stable else '(UNSTABLE)'}")

        except Exception as e:
            if i < 5:
                print(f"    Restart {i + 1}: failed ({e})")

    if best_params is None:
        print("  WARNING: No stable solution found!")
        return None

    # Compute final statistics
    params = best_params
    param_dict = {name: params[i] for i, name in enumerate(PARAM_NAMES)}

    # Forward simulation
    x_3state, x_pred_forward = simulate_forward_3state(params, x_obs[0], u_inputs)

    # Rolling horizon simulations
    x_pred_24h = simulate_rolling_horizon_3state(params, x_obs, u_inputs, horizon_steps=96)
    x_pred_6h = simulate_rolling_horizon_3state(params, x_obs, u_inputs, horizon_steps=24)
    x_pred_2h = simulate_rolling_horizon_3state(params, x_obs, u_inputs, horizon_steps=8)

    # Stability analysis
    is_stable, max_eig, eigenvalues = compute_eigenvalues_3state(params)

    stats = {
        'params': param_dict,
        'is_stable': is_stable,
        'max_eigenvalue': max_eig,
        'eigenvalues': eigenvalues.tolist(),
        # Forward simulation
        'r2_room_forward': r2_score(x_obs[:, 1], x_pred_forward[:, 1]),
        'rmse_room_forward': np.sqrt(mean_squared_error(x_obs[:, 1], x_pred_forward[:, 1])),
        'bias_room_forward': (x_pred_forward[:, 1] - x_obs[:, 1]).mean(),
        'r2_buffer_forward': r2_score(x_obs[:, 0], x_pred_forward[:, 0]),
        # Rolling horizon
        'r2_room_24h': r2_score(x_obs[:, 1], x_pred_24h[:, 1]),
        'rmse_room_24h': np.sqrt(mean_squared_error(x_obs[:, 1], x_pred_24h[:, 1])),
        'r2_room_6h': r2_score(x_obs[:, 1], x_pred_6h[:, 1]),
        'rmse_room_6h': np.sqrt(mean_squared_error(x_obs[:, 1], x_pred_6h[:, 1])),
        'r2_room_2h': r2_score(x_obs[:, 1], x_pred_2h[:, 1]),
        'rmse_room_2h': np.sqrt(mean_squared_error(x_obs[:, 1], x_pred_2h[:, 1])),
        # Predictions
        'x_3state': x_3state,
        'x_pred_forward': x_pred_forward,
        'x_pred_24h': x_pred_24h,
        'n_points': len(x_obs),
    }

    # Print results
    print(f"\n{'=' * 60}")
    print("3-STATE MODEL RESULTS")
    print('=' * 60)

    print(f"\nParameters:")
    print(f"  Time constants:")
    print(f"    tau_buf:  {param_dict['tau_buf']:.2f} h (buffer)")
    print(f"    tau_env:  {param_dict['tau_env']:.1f} h (envelope = {param_dict['tau_env']/24:.1f} days)")
    print(f"    tau_room: {param_dict['tau_room']:.2f} h (room air)")
    print(f"  Coupling ratios:")
    print(f"    r_emit:    {param_dict['r_emit']:.3f} (buffer→room)")
    print(f"    r_heat:    {param_dict['r_heat']:.3f} (heating ratio)")
    print(f"    r_env:     {param_dict['r_env']:.3f} (envelope↔room)")
    print(f"    r_env_out: {param_dict['r_env_out']:.3f} (envelope→outdoor)")
    print(f"  Gains:")
    print(f"    k_solar:  {param_dict['k_solar']:.4f} K/kWh")
    print(f"    c_offset: {param_dict['c_offset']:.4f} K/h")

    print(f"\nStability: max|λ| = {max_eig:.4f} {'(STABLE)' if is_stable else '(UNSTABLE)'}")

    print(f"\nForward Simulation:")
    print(f"  R² (room):   {stats['r2_room_forward']:.4f}")
    print(f"  RMSE (room): {stats['rmse_room_forward']:.3f}°C")

    print(f"\nRolling Horizon:")
    print(f"  24h: R² = {stats['r2_room_24h']:.4f}, RMSE = {stats['rmse_room_24h']:.3f}°C")
    print(f"   6h: R² = {stats['r2_room_6h']:.4f}, RMSE = {stats['rmse_room_6h']:.3f}°C")
    print(f"   2h: R² = {stats['r2_room_2h']:.4f}, RMSE = {stats['rmse_room_2h']:.3f}°C")

    return stats


def plot_results_3state(stats: dict, x_obs: np.ndarray, timestamps: pd.DatetimeIndex) -> None:
    """Create visualization for 3-state model."""
    print("\nCreating visualization...")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    x_3state = stats['x_3state']
    x_pred_forward = stats['x_pred_forward']

    step = max(1, len(timestamps) // 500)

    # Panel 1: Room temperature - forward simulation
    ax1 = axes[0, 0]
    ax1.plot(timestamps[::step], x_obs[::step, 1], 'b-', linewidth=0.8, alpha=0.8, label='Actual')
    ax1.plot(timestamps[::step], x_pred_forward[::step, 1], 'r-', linewidth=0.8, alpha=0.7, label='3-state model')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Room Temperature (°C)')
    ax1.set_title(f'Forward Simulation (R²={stats["r2_room_forward"]:.3f})')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Panel 2: Envelope temperature (hidden state)
    ax2 = axes[0, 1]
    ax2.plot(timestamps[::step], x_3state[::step, 1], 'purple', linewidth=0.8, label='T_envelope (hidden)')
    ax2.plot(timestamps[::step], x_3state[::step, 2], 'orange', linewidth=0.8, alpha=0.7, label='T_room (predicted)')
    ax2.plot(timestamps[::step], x_obs[::step, 1], 'b--', linewidth=0.5, alpha=0.5, label='T_room (actual)')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Temperature (°C)')
    ax2.set_title('Hidden Envelope State')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Panel 3: Scatter plot
    ax3 = axes[0, 2]
    step_scatter = max(1, len(x_obs) // 1000)
    ax3.scatter(x_obs[::step_scatter, 1], x_pred_forward[::step_scatter, 1],
                alpha=0.4, s=10, c='blue')
    temp_range = [x_obs[:, 1].min() - 0.5, x_obs[:, 1].max() + 0.5]
    ax3.plot(temp_range, temp_range, 'r--', linewidth=1.5, label='Perfect fit')
    ax3.set_xlabel('Actual Room Temperature (°C)')
    ax3.set_ylabel('Predicted Room Temperature (°C)')
    ax3.set_title(f'RMSE = {stats["rmse_room_forward"]:.2f}°C')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Residuals
    ax4 = axes[1, 0]
    residuals = x_obs[:, 1] - x_pred_forward[:, 1]
    ax4.plot(timestamps[::step], residuals[::step], 'b-', linewidth=0.5, alpha=0.7)
    ax4.axhline(0, color='red', linestyle='--', linewidth=1.5)
    ax4.fill_between(timestamps[::step], -1, 1, alpha=0.2, color='green', label='±1°C')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Residual (°C)')
    ax4.set_title(f'Residuals (mean={residuals.mean():.2f}°C, std={residuals.std():.2f}°C)')
    ax4.grid(True, alpha=0.3)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Panel 5: Eigenvalues
    ax5 = axes[1, 1]
    theta = np.linspace(0, 2 * np.pi, 100)
    ax5.plot(np.cos(theta), np.sin(theta), 'k--', linewidth=1, alpha=0.5)
    ax5.plot(MAX_EIGENVALUE * np.cos(theta), MAX_EIGENVALUE * np.sin(theta),
             'g--', linewidth=1, alpha=0.5, label=f'Stability ({MAX_EIGENVALUE})')
    for i, eig in enumerate(stats['eigenvalues']):
        color = 'green' if abs(eig) < MAX_EIGENVALUE else 'red'
        ax5.scatter(np.real(eig), np.imag(eig), s=100, c=color, marker='x', linewidths=3)
        ax5.annotate(f'λ{i+1}={eig:.3f}', (np.real(eig), np.imag(eig)),
                     xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax5.set_xlim(-0.2, 1.1)
    ax5.set_ylim(-0.5, 0.5)
    ax5.set_xlabel('Real')
    ax5.set_ylabel('Imaginary')
    ax5.set_title(f'Eigenvalues (max |λ| = {stats["max_eigenvalue"]:.4f})')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # Panel 6: Model comparison
    ax6 = axes[1, 2]

    # Load 2-state results for comparison
    twostate_file = OUTPUT_DIR / 'stable_greybox_params.json'
    if twostate_file.exists():
        with open(twostate_file) as f:
            twostate = json.load(f)
        r2_2state = twostate['forward_sim_metrics']['r2_room']
        r2_2state_24h = twostate['rolling_horizon_metrics']['24h']['r2_room']
    else:
        r2_2state = 0.58
        r2_2state_24h = 0.66

    # Transfer function reference
    tf_file = OUTPUT_DIR / 'thermal_model_results.csv'
    if tf_file.exists():
        tf_df = pd.read_csv(tf_file)
        r2_tf = tf_df[tf_df['room'] == 'davis_inside']['r2'].values[0]
    else:
        r2_tf = 0.68

    models = ['Transfer\nFunction', '2-State\n(forward)', '2-State\n(24h)', '3-State\n(forward)', '3-State\n(24h)']
    r2_values = [r2_tf, r2_2state, r2_2state_24h, stats['r2_room_forward'], stats['r2_room_24h']]

    colors = ['steelblue', 'coral', 'coral', 'green', 'green']
    bars = ax6.bar(models, r2_values, color=colors, alpha=0.7)
    ax6.bar_label(bars, fmt='%.3f', padding=3, fontsize=9)
    ax6.set_ylabel('R²')
    ax6.set_title('Model Comparison')
    ax6.set_ylim(0, 1.0)
    ax6.axhline(r2_tf, color='steelblue', linestyle='--', alpha=0.5)
    ax6.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig3.01e_threestate_greybox_model.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: fig3.01e_threestate_greybox_model.png")


def main():
    """Main function for 3-state grey-box model."""
    print("=" * 60)
    print("Phase 3, Step 1d: Three-State Grey-Box Thermal Model")
    print("=" * 60)

    # Load and prepare data
    df = load_data()
    x_obs, u_inputs, timestamps = prepare_data(df)

    # Estimate parameters
    stats = estimate_parameters_3state(x_obs, u_inputs, n_restarts=15)

    if stats is None:
        print("ERROR: Failed to find stable parameters")
        return

    # Visualization
    plot_results_3state(stats, x_obs, timestamps)

    # Save results
    results = {
        'params': {k: float(v) for k, v in stats['params'].items()},
        'stability': {
            'is_stable': bool(stats['is_stable']),
            'max_eigenvalue': float(stats['max_eigenvalue']),
            'eigenvalues': [float(e) for e in stats['eigenvalues']],
        },
        'forward_sim_metrics': {
            'r2_room': float(stats['r2_room_forward']),
            'rmse_room': float(stats['rmse_room_forward']),
            'bias_room': float(stats['bias_room_forward']),
            'r2_buffer': float(stats['r2_buffer_forward']),
        },
        'rolling_horizon_metrics': {
            '24h': {'r2_room': float(stats['r2_room_24h']), 'rmse_room': float(stats['rmse_room_24h'])},
            '6h': {'r2_room': float(stats['r2_room_6h']), 'rmse_room': float(stats['rmse_room_6h'])},
            '2h': {'r2_room': float(stats['r2_room_2h']), 'rmse_room': float(stats['rmse_room_2h'])},
        },
        'n_points': int(stats['n_points']),
    }

    with open(OUTPUT_DIR / 'threestate_greybox_params.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: threestate_greybox_params.json")

    # Save predictions
    pred_df = pd.DataFrame({
        'timestamp': timestamps,
        'T_buffer_actual': x_obs[:, 0],
        'T_buffer_pred': stats['x_pred_forward'][:, 0],
        'T_envelope_pred': stats['x_3state'][:, 1],
        'T_room_actual': x_obs[:, 1],
        'T_room_pred': stats['x_pred_forward'][:, 1],
    })
    pred_df.to_csv(OUTPUT_DIR / 'threestate_greybox_results.csv', index=False)
    print("Saved: threestate_greybox_results.csv")

    # Summary comparison
    print("\n" + "=" * 60)
    print("COMPARISON: 2-STATE vs 3-STATE")
    print("=" * 60)

    twostate_file = OUTPUT_DIR / 'stable_greybox_params.json'
    if twostate_file.exists():
        with open(twostate_file) as f:
            twostate = json.load(f)

        print(f"\n{'Metric':<25} {'2-State':>10} {'3-State':>10} {'Diff':>10}")
        print("-" * 55)

        r2_2s = twostate['forward_sim_metrics']['r2_room']
        r2_3s = stats['r2_room_forward']
        print(f"{'Forward sim R²':<25} {r2_2s:>10.3f} {r2_3s:>10.3f} {r2_3s - r2_2s:>+10.3f}")

        r2_2s_24h = twostate['rolling_horizon_metrics']['24h']['r2_room']
        r2_3s_24h = stats['r2_room_24h']
        print(f"{'24h rolling R²':<25} {r2_2s_24h:>10.3f} {r2_3s_24h:>10.3f} {r2_3s_24h - r2_2s_24h:>+10.3f}")

        r2_2s_6h = twostate['rolling_horizon_metrics']['6h']['r2_room']
        r2_3s_6h = stats['r2_room_6h']
        print(f"{'6h rolling R²':<25} {r2_2s_6h:>10.3f} {r2_3s_6h:>10.3f} {r2_3s_6h - r2_2s_6h:>+10.3f}")

        rmse_2s = twostate['forward_sim_metrics']['rmse_room']
        rmse_3s = stats['rmse_room_forward']
        print(f"{'Forward sim RMSE (°C)':<25} {rmse_2s:>10.2f} {rmse_3s:>10.2f} {rmse_3s - rmse_2s:>+10.2f}")


if __name__ == '__main__':
    main()
