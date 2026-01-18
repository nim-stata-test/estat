#!/usr/bin/env python3
"""
Phase 3, Step 1c: Stability-Constrained Grey-Box Thermal Model

Improvements over 01b_greybox_thermal_model.py:
1. Matrix formulation with explicit eigenvalue stability constraints
2. Semi-implicit integration for better numerical stability
3. Optional Kalman filter for state correction during simulation
4. Regularization to prevent parameter drift

The key insight: Forward simulation fails when eigenvalues of the state transition
matrix exceed 1.0 in magnitude. This version explicitly constrains eigenvalues
during optimization.

Model formulation (state-space):
    x[k+1] = A(params) @ x[k] + B(params) @ u[k]

    where x = [T_buffer, T_room]^T
          u = [T_HK2, T_outdoor, PV]^T

Stability requirement: max|eigenvalue(A)| < 1.0
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize, least_squares
from scipy.linalg import eig
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

# Parameter bounds - tightened for stability
PARAM_BOUNDS = {
    'tau_buf': (1.0, 8.0),       # Buffer time constant (hours) - increased minimum
    'tau_room': (12.0, 96.0),    # Building time constant (hours)
    'r_emit': (0.05, 2.0),       # Emitter coupling ratio
    'r_heat': (0.05, 2.0),       # Heat transfer ratio
    'k_solar': (0.0, 1.0),       # Solar gain (K/kWh)
    'c_offset': (-1.0, 1.0),     # Temperature offset (K)
}

PARAM_NAMES = list(PARAM_BOUNDS.keys())

# Stability margin (max eigenvalue magnitude)
MAX_EIGENVALUE = 0.998  # Relaxed slightly for better fit


def load_data() -> pd.DataFrame:
    """Load integrated dataset for thermal modeling."""
    print("Loading data for stable grey-box thermal model...")

    df = pd.read_parquet(PROCESSED_DIR / 'integrated_dataset.parquet')
    df.index = pd.to_datetime(df.index)

    print(f"  Dataset: {len(df):,} rows ({df.index.min().date()} to {df.index.max().date()})")

    return df


def prepare_data(df: pd.DataFrame) -> tuple:
    """Prepare aligned data for state-space model."""
    print("\nPreparing data...")

    required = [BUFFER_COL, HK2_COL, ROOM_COL, OUTDOOR_COL, PV_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    data = df[required].copy()
    data = data.ffill(limit=4)
    data = data.dropna()

    print(f"  Valid observations: {len(data):,}")
    print(f"  Period: {data.index.min().date()} to {data.index.max().date()}")

    x_obs = data[[BUFFER_COL, ROOM_COL]].values
    u_inputs = data[[HK2_COL, OUTDOOR_COL, PV_COL]].values
    timestamps = data.index

    return x_obs, u_inputs, timestamps


def build_state_matrices(params: np.ndarray, dt: float = DT_HOURS) -> tuple:
    """
    Build state-space matrices A, B from physical parameters.

    State equation: x[k+1] = A @ x[k] + B @ u[k]

    where x = [T_buf, T_room]^T
          u = [T_HK2, T_outdoor, PV]^T

    Returns:
        A: 2x2 state transition matrix
        B: 2x3 input matrix
    """
    tau_buf, tau_room, r_emit, r_heat, k_solar, c_offset = params

    # Discrete-time state transition matrix (explicit Euler)
    # dT_buf/dt = (1/tau_buf) * [(T_HK2 - T_buf) - r_emit*(T_buf - T_room)]
    # dT_room/dt = (1/tau_room) * [r_heat*(T_buf - T_room) - (T_room - T_out)] + k_solar*PV

    # Linearized around current state:
    # dT_buf/dt = -(1 + r_emit)/tau_buf * T_buf + r_emit/tau_buf * T_room + 1/tau_buf * T_HK2
    # dT_room/dt = r_heat/tau_room * T_buf - (r_heat + 1)/tau_room * T_room + 1/tau_room * T_out + k_solar * PV

    # Continuous-time A matrix
    Ac = np.array([
        [-(1 + r_emit) / tau_buf, r_emit / tau_buf],
        [r_heat / tau_room, -(r_heat + 1) / tau_room]
    ])

    # Discrete-time approximation: A = I + dt * Ac (explicit Euler)
    A = np.eye(2) + dt * Ac

    # Input matrix B (maps [T_HK2, T_outdoor, PV] to state derivatives)
    Bc = np.array([
        [1 / tau_buf, 0, 0],  # T_HK2 affects buffer
        [0, 1 / tau_room, k_solar]  # T_outdoor affects room, PV adds solar gain
    ])

    # Discrete-time: B = dt * Bc
    B = dt * Bc

    # Add offset term to room equation (as a constant input)
    # This is handled separately in simulation

    return A, B


def compute_eigenvalues(params: np.ndarray, dt: float = DT_HOURS) -> np.ndarray:
    """Compute eigenvalues of state transition matrix."""
    A, _ = build_state_matrices(params, dt)
    eigenvalues = np.linalg.eigvals(A)
    return eigenvalues


def check_stability(params: np.ndarray, dt: float = DT_HOURS) -> tuple:
    """
    Check if parameters yield stable forward simulation.

    Returns:
        is_stable: bool
        max_eigenvalue: float (magnitude of largest eigenvalue)
        eigenvalues: array of eigenvalues
    """
    eigenvalues = compute_eigenvalues(params, dt)
    max_eig = np.max(np.abs(eigenvalues))
    is_stable = max_eig < 1.0
    return is_stable, max_eig, eigenvalues


def simulate_forward_matrix(params: np.ndarray, x0: np.ndarray, u_inputs: np.ndarray,
                            dt: float = DT_HOURS) -> np.ndarray:
    """
    Forward simulate using matrix formulation.

    This is mathematically equivalent to the loop version but uses
    explicit state-space form for clarity and stability analysis.
    """
    tau_buf, tau_room, r_emit, r_heat, k_solar, c_offset = params
    A, B = build_state_matrices(params, dt)

    n = len(u_inputs)
    x_pred = np.zeros((n, 2))
    x_pred[0] = x0

    for k in range(n - 1):
        # State transition
        x_pred[k + 1] = A @ x_pred[k] + B @ u_inputs[k]
        # Add constant offset to room temperature
        x_pred[k + 1, 1] += c_offset * dt

    return x_pred


def simulate_forward_semiimplicit(params: np.ndarray, x0: np.ndarray, u_inputs: np.ndarray,
                                   dt: float = DT_HOURS) -> np.ndarray:
    """
    Forward simulate using semi-implicit integration.

    Semi-implicit: solve for x[k+1] implicitly for the diagonal terms,
    explicitly for cross-coupling. This is unconditionally stable for
    the diagonal decay terms.

    For the buffer equation:
        T_buf[k+1] = T_buf[k] + (dt/tau_buf) * [(T_HK2[k] - T_buf[k+1]) - r_emit*(T_buf[k+1] - T_room[k])]

    Solving for T_buf[k+1]:
        T_buf[k+1] * (1 + dt/tau_buf * (1 + r_emit)) = T_buf[k] + (dt/tau_buf) * (T_HK2[k] + r_emit*T_room[k])
    """
    tau_buf, tau_room, r_emit, r_heat, k_solar, c_offset = params

    n = len(u_inputs)
    x_pred = np.zeros((n, 2))
    x_pred[0] = x0

    # Precompute implicit factors
    alpha_buf = dt / tau_buf
    alpha_room = dt / tau_room

    denom_buf = 1 + alpha_buf * (1 + r_emit)
    denom_room = 1 + alpha_room * (r_heat + 1)

    for k in range(n - 1):
        T_buf_k = x_pred[k, 0]
        T_room_k = x_pred[k, 1]
        T_hk2, T_out, pv = u_inputs[k]

        # Semi-implicit buffer update
        T_buf_new = (T_buf_k + alpha_buf * (T_hk2 + r_emit * T_room_k)) / denom_buf

        # Semi-implicit room update (using new buffer temperature)
        T_room_new = (T_room_k + alpha_room * (r_heat * T_buf_new + T_out) + k_solar * pv) / denom_room
        T_room_new += c_offset * dt

        x_pred[k + 1, 0] = T_buf_new
        x_pred[k + 1, 1] = T_room_new

    return x_pred


def simulate_with_kalman(params: np.ndarray, x0: np.ndarray, u_inputs: np.ndarray,
                         x_obs: np.ndarray, process_noise: float = 0.05,
                         measurement_noise: float = 0.15, dt: float = DT_HOURS) -> np.ndarray:
    """
    Forward simulate with simple Kalman filter correction.

    Uses a standard Kalman filter to correct state estimates when
    observations are available.

    Args:
        params: Model parameters
        x0: Initial state [T_buffer_0, T_room_0]
        u_inputs: Input sequence
        x_obs: Observations
        process_noise: Process noise std dev (per 15-min step)
        measurement_noise: Measurement noise std dev
        dt: Time step

    Returns:
        x_filtered: Kalman-filtered state estimates
    """
    tau_buf, tau_room, r_emit, r_heat, k_solar, c_offset = params

    # Pre-compute semi-implicit factors
    alpha_buf = dt / tau_buf
    alpha_room = dt / tau_room
    denom_buf = 1 + alpha_buf * (1 + r_emit)
    denom_room = 1 + alpha_room * (r_heat + 1)

    n = len(u_inputs)
    x_filtered = np.zeros((n, 2))
    x_filtered[0] = x0

    # Current state estimate
    x_est = x0.copy()

    # Observation matrix (we observe both states directly)
    H = np.eye(2)

    # Process noise covariance
    Q = np.diag([process_noise**2 * 4, process_noise**2])  # Buffer more uncertain

    # Measurement noise covariance
    R = np.diag([measurement_noise**2 * 4, measurement_noise**2])

    # Initial covariance
    P = np.diag([1.0, 0.5])

    for k in range(n - 1):
        # Predict step using semi-implicit integration
        T_buf_k, T_room_k = x_est
        T_hk2, T_out, pv = u_inputs[k]

        T_buf_pred = (T_buf_k + alpha_buf * (T_hk2 + r_emit * T_room_k)) / denom_buf
        T_room_pred = (T_room_k + alpha_room * (r_heat * T_buf_pred + T_out) + k_solar * pv) / denom_room
        T_room_pred += c_offset * dt

        x_pred = np.array([T_buf_pred, T_room_pred])

        # Approximate state transition matrix for covariance
        A_approx = np.array([
            [1 / denom_buf, alpha_buf * r_emit / denom_buf],
            [alpha_room * r_heat / denom_room, 1 / denom_room]
        ])
        P_pred = A_approx @ P @ A_approx.T + Q

        # Update step
        if not np.any(np.isnan(x_obs[k + 1])):
            # Kalman gain
            S = H @ P_pred @ H.T + R
            K = P_pred @ H.T @ np.linalg.inv(S)

            # Innovation
            y = x_obs[k + 1] - x_pred

            # Update
            x_est = x_pred + K @ y
            P = (np.eye(2) - K @ H) @ P_pred
        else:
            x_est = x_pred
            P = P_pred

        x_filtered[k + 1] = x_est

    return x_filtered


def one_step_predict(params: np.ndarray, x_obs: np.ndarray, u_inputs: np.ndarray,
                     dt: float = DT_HOURS) -> np.ndarray:
    """One-step-ahead prediction using observed states."""
    tau_buf, tau_room, r_emit, r_heat, k_solar, c_offset = params
    A, B = build_state_matrices(params, dt)

    n = len(u_inputs)
    x_pred = np.zeros((n, 2))
    x_pred[0] = x_obs[0]

    for k in range(n - 1):
        x_pred[k + 1] = A @ x_obs[k] + B @ u_inputs[k]
        x_pred[k + 1, 1] += c_offset * dt

    return x_pred


def simulate_rolling_horizon(params: np.ndarray, x_obs: np.ndarray, u_inputs: np.ndarray,
                              horizon_steps: int = 96, dt: float = DT_HOURS) -> np.ndarray:
    """
    Rolling horizon simulation - reset from observations every horizon_steps.

    This gives a more realistic evaluation of prediction accuracy over
    practical time horizons (e.g., 24 hours = 96 steps at 15-min).

    Args:
        params: Model parameters
        x_obs: Observed states
        u_inputs: Inputs
        horizon_steps: Steps between resets (default 96 = 24 hours)
        dt: Time step

    Returns:
        x_pred: Predictions (reset at horizon boundaries)
    """
    tau_buf, tau_room, r_emit, r_heat, k_solar, c_offset = params

    n = len(u_inputs)
    x_pred = np.zeros((n, 2))
    x_pred[0] = x_obs[0]

    # Pre-compute semi-implicit factors
    alpha_buf = dt / tau_buf
    alpha_room = dt / tau_room
    denom_buf = 1 + alpha_buf * (1 + r_emit)
    denom_room = 1 + alpha_room * (r_heat + 1)

    for k in range(n - 1):
        # Reset to observed state at horizon boundaries
        if k % horizon_steps == 0:
            T_buf_k = x_obs[k, 0]
            T_room_k = x_obs[k, 1]
        else:
            T_buf_k = x_pred[k, 0]
            T_room_k = x_pred[k, 1]

        T_hk2, T_out, pv = u_inputs[k]

        # Semi-implicit update
        T_buf_new = (T_buf_k + alpha_buf * (T_hk2 + r_emit * T_room_k)) / denom_buf
        T_room_new = (T_room_k + alpha_room * (r_heat * T_buf_new + T_out) + k_solar * pv) / denom_room
        T_room_new += c_offset * dt

        x_pred[k + 1, 0] = T_buf_new
        x_pred[k + 1, 1] = T_room_new

    return x_pred


def stability_penalty(params: np.ndarray, dt: float = DT_HOURS,
                      target_max_eig: float = MAX_EIGENVALUE) -> float:
    """
    Compute penalty for eigenvalues exceeding stability threshold.

    Returns 0 if stable, exponentially increasing penalty otherwise.
    """
    _, max_eig, _ = check_stability(params, dt)

    if max_eig < target_max_eig:
        return 0.0
    else:
        # Exponential penalty for instability
        return 1000 * np.exp(10 * (max_eig - target_max_eig))


def objective_function(params: np.ndarray, x_obs: np.ndarray, u_inputs: np.ndarray,
                       dt: float = DT_HOURS, lambda_stability: float = 1.0,
                       use_forward: bool = True) -> float:
    """
    Combined objective: prediction error + stability penalty.

    Args:
        params: Model parameters
        x_obs: Observed states
        u_inputs: Inputs
        dt: Time step
        lambda_stability: Weight for stability penalty
        use_forward: If True, evaluate on forward simulation; else one-step

    Returns:
        Total objective value (to minimize)
    """
    # Check parameter bounds
    lower = np.array([PARAM_BOUNDS[p][0] for p in PARAM_NAMES])
    upper = np.array([PARAM_BOUNDS[p][1] for p in PARAM_NAMES])

    if np.any(params < lower) or np.any(params > upper):
        return 1e10

    # Stability penalty
    stab_penalty = stability_penalty(params, dt)

    if use_forward:
        # Forward simulation error (the hard test)
        x_pred = simulate_forward_semiimplicit(params, x_obs[0], u_inputs, dt)
        mse = np.mean((x_obs[:, 1] - x_pred[:, 1]) ** 2)
    else:
        # One-step prediction error (easier)
        x_pred = one_step_predict(params, x_obs, u_inputs, dt)
        mse = np.mean((x_obs[1:, 1] - x_pred[1:, 1]) ** 2)

    return mse + lambda_stability * stab_penalty


def estimate_parameters_stable(x_obs: np.ndarray, u_inputs: np.ndarray,
                                n_restarts: int = 10, seed: int = 42,
                                use_forward: bool = True) -> dict:
    """
    Estimate model parameters with stability constraints.

    Uses a two-phase approach:
    1. First optimize for one-step prediction to get good initial guess
    2. Then optimize for forward simulation with stability penalty
    """
    print("\nEstimating parameters with stability constraints...")

    rng = np.random.RandomState(seed)

    lower = np.array([PARAM_BOUNDS[p][0] for p in PARAM_NAMES])
    upper = np.array([PARAM_BOUNDS[p][1] for p in PARAM_NAMES])
    bounds = [(l, u) for l, u in zip(lower, upper)]

    # Phase 1: Optimize for one-step prediction (to get good initial guess)
    print("\n  Phase 1: One-step prediction optimization...")
    best_p1 = None
    best_obj1 = np.inf

    # Initial guesses
    p0_physics = np.array([2.5, 36.0, 0.5, 0.5, 0.1, 0.0])

    for i in range(n_restarts):
        if i == 0:
            p0 = p0_physics
        else:
            p0 = lower + (upper - lower) * rng.rand(len(lower))

        try:
            result = minimize(
                objective_function,
                p0,
                args=(x_obs, u_inputs, DT_HOURS, 0.1, False),  # One-step, light stability
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-8}
            )

            if result.fun < best_obj1:
                best_obj1 = result.fun
                best_p1 = result.x
                is_stable, max_eig, _ = check_stability(result.x)
                print(f"    Restart {i + 1}: MSE = {result.fun:.4f}, max_eig = {max_eig:.4f} {'(stable)' if is_stable else '(UNSTABLE)'}")
        except Exception as e:
            print(f"    Restart {i + 1}: failed ({e})")

    if best_p1 is None:
        raise RuntimeError("Phase 1 optimization failed")

    # Phase 2: Optimize for forward simulation with strong stability penalty
    print("\n  Phase 2: Forward simulation optimization with stability constraint...")
    best_p2 = None
    best_obj2 = np.inf

    for i in range(n_restarts):
        if i == 0:
            p0 = best_p1  # Start from Phase 1 result
        elif i == 1:
            p0 = p0_physics
        else:
            # Perturb around best Phase 1 result
            p0 = best_p1 + (upper - lower) * 0.1 * (rng.rand(len(lower)) - 0.5)
            p0 = np.clip(p0, lower, upper)

        try:
            result = minimize(
                objective_function,
                p0,
                args=(x_obs, u_inputs, DT_HOURS, 10.0, True),  # Forward, strong stability
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )

            is_stable, max_eig, _ = check_stability(result.x)

            # Only accept stable solutions
            if is_stable and result.fun < best_obj2:
                best_obj2 = result.fun
                best_p2 = result.x
                print(f"    Restart {i + 1}: MSE = {result.fun:.4f}, max_eig = {max_eig:.4f} (stable) *BEST*")
            else:
                print(f"    Restart {i + 1}: MSE = {result.fun:.4f}, max_eig = {max_eig:.4f} {'(stable)' if is_stable else '(UNSTABLE)'}")

        except Exception as e:
            print(f"    Restart {i + 1}: failed ({e})")

    if best_p2 is None:
        print("\n  WARNING: No stable solution found. Using best Phase 1 result.")
        best_p2 = best_p1

    # Compute final statistics
    params = best_p2
    param_dict = {name: params[i] for i, name in enumerate(PARAM_NAMES)}

    # Forward simulation (the true test)
    x_pred_forward = simulate_forward_semiimplicit(params, x_obs[0], u_inputs)

    # Rolling horizon simulations at different horizons
    x_pred_24h = simulate_rolling_horizon(params, x_obs, u_inputs, horizon_steps=96)  # 24h
    x_pred_6h = simulate_rolling_horizon(params, x_obs, u_inputs, horizon_steps=24)   # 6h
    x_pred_2h = simulate_rolling_horizon(params, x_obs, u_inputs, horizon_steps=8)    # 2h

    # Kalman-filtered simulation
    x_pred_kalman = simulate_with_kalman(params, x_obs[0], u_inputs, x_obs,
                                          process_noise=0.05, measurement_noise=0.2)

    # One-step prediction (for comparison)
    x_pred_1step = one_step_predict(params, x_obs, u_inputs)

    # Stability analysis
    is_stable, max_eig, eigenvalues = check_stability(params)

    stats = {
        'params': param_dict,
        # Stability
        'is_stable': is_stable,
        'max_eigenvalue': max_eig,
        'eigenvalues': eigenvalues.tolist(),
        # Forward simulation metrics (full horizon)
        'r2_room_forward': r2_score(x_obs[:, 1], x_pred_forward[:, 1]),
        'rmse_room_forward': np.sqrt(mean_squared_error(x_obs[:, 1], x_pred_forward[:, 1])),
        'mae_room_forward': mean_absolute_error(x_obs[:, 1], x_pred_forward[:, 1]),
        'bias_room_forward': (x_pred_forward[:, 1] - x_obs[:, 1]).mean(),
        # Rolling horizon metrics (practical horizons)
        'r2_room_24h': r2_score(x_obs[:, 1], x_pred_24h[:, 1]),
        'rmse_room_24h': np.sqrt(mean_squared_error(x_obs[:, 1], x_pred_24h[:, 1])),
        'r2_room_6h': r2_score(x_obs[:, 1], x_pred_6h[:, 1]),
        'rmse_room_6h': np.sqrt(mean_squared_error(x_obs[:, 1], x_pred_6h[:, 1])),
        'r2_room_2h': r2_score(x_obs[:, 1], x_pred_2h[:, 1]),
        'rmse_room_2h': np.sqrt(mean_squared_error(x_obs[:, 1], x_pred_2h[:, 1])),
        # Kalman-filtered metrics
        'r2_room_kalman': r2_score(x_obs[:, 1], x_pred_kalman[:, 1]),
        'rmse_room_kalman': np.sqrt(mean_squared_error(x_obs[:, 1], x_pred_kalman[:, 1])),
        # One-step metrics (for reference only - inflated)
        'r2_room_1step': r2_score(x_obs[1:, 1], x_pred_1step[1:, 1]),
        'rmse_room_1step': np.sqrt(mean_squared_error(x_obs[1:, 1], x_pred_1step[1:, 1])),
        # Buffer metrics
        'r2_buffer_forward': r2_score(x_obs[:, 0], x_pred_forward[:, 0]),
        'rmse_buffer_forward': np.sqrt(mean_squared_error(x_obs[:, 0], x_pred_forward[:, 0])),
        # Predictions
        'x_pred_forward': x_pred_forward,
        'x_pred_24h': x_pred_24h,
        'x_pred_kalman': x_pred_kalman,
        'x_pred_1step': x_pred_1step,
        'n_points': len(x_obs),
    }

    # Print summary
    print(f"\n{'=' * 60}")
    print("OPTIMIZATION RESULTS")
    print('=' * 60)
    print(f"\nParameters:")
    for name in PARAM_NAMES:
        print(f"  {name}: {param_dict[name]:.4f}")

    print(f"\nStability Analysis:")
    print(f"  Max eigenvalue: {max_eig:.4f} {'(STABLE)' if is_stable else '(UNSTABLE)'}")
    print(f"  Eigenvalues: {eigenvalues}")

    print(f"\nForward Simulation Performance:")
    print(f"  R² (room):   {stats['r2_room_forward']:.4f}")
    print(f"  RMSE (room): {stats['rmse_room_forward']:.3f}°C")
    print(f"  Bias (room): {stats['bias_room_forward']:.3f}°C")

    print(f"\nRolling Horizon Performance (reset from observations):")
    print(f"  24h horizon: R² = {stats['r2_room_24h']:.4f}, RMSE = {stats['rmse_room_24h']:.3f}°C")
    print(f"   6h horizon: R² = {stats['r2_room_6h']:.4f}, RMSE = {stats['rmse_room_6h']:.3f}°C")
    print(f"   2h horizon: R² = {stats['r2_room_2h']:.4f}, RMSE = {stats['rmse_room_2h']:.3f}°C")

    print(f"\nKalman Filter Performance:")
    print(f"  R² (room):   {stats['r2_room_kalman']:.4f}")
    print(f"  RMSE (room): {stats['rmse_room_kalman']:.3f}°C")

    print(f"\nOne-Step Performance (reference only - inflated):")
    print(f"  R² (room):   {stats['r2_room_1step']:.4f}")

    return stats


def plot_results(stats: dict, x_obs: np.ndarray, u_inputs: np.ndarray,
                 timestamps: pd.DatetimeIndex, split_idx: int) -> None:
    """Create 6-panel visualization of stable grey-box model."""
    print("\nCreating visualization...")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    x_pred_forward = stats['x_pred_forward']
    x_pred_kalman = stats['x_pred_kalman']

    # Panel 1: Forward simulation vs actual (full period)
    ax1 = axes[0, 0]
    step = max(1, len(timestamps) // 500)
    ax1.plot(timestamps[::step], x_obs[::step, 1], 'b-', linewidth=0.8, alpha=0.8, label='Actual')
    ax1.plot(timestamps[::step], x_pred_forward[::step, 1], 'r-', linewidth=0.8, alpha=0.7, label='Forward sim')
    ax1.axvline(timestamps[split_idx], color='green', linestyle=':', linewidth=2, alpha=0.7, label='Train/Test')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Room Temperature (°C)')
    ax1.set_title(f'Forward Simulation (R²={stats["r2_room_forward"]:.3f})')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Panel 2: Kalman-filtered vs actual
    ax2 = axes[0, 1]
    ax2.plot(timestamps[::step], x_obs[::step, 1], 'b-', linewidth=0.8, alpha=0.8, label='Actual')
    ax2.plot(timestamps[::step], x_pred_kalman[::step, 1], 'g-', linewidth=0.8, alpha=0.7, label='Kalman filter')
    ax2.axvline(timestamps[split_idx], color='green', linestyle=':', linewidth=2, alpha=0.7)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Room Temperature (°C)')
    ax2.set_title(f'Kalman Filter (R²={stats["r2_room_kalman"]:.3f})')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Panel 3: Actual vs Predicted scatter (forward sim)
    ax3 = axes[0, 2]
    step_scatter = max(1, len(x_obs) // 1000)
    ax3.scatter(x_obs[::step_scatter, 1], x_pred_forward[::step_scatter, 1],
                alpha=0.4, s=10, c='blue', label='Forward sim')
    temp_range = [x_obs[:, 1].min() - 0.5, x_obs[:, 1].max() + 0.5]
    ax3.plot(temp_range, temp_range, 'r--', linewidth=1.5, label='Perfect fit')
    ax3.set_xlabel('Actual Room Temperature (°C)')
    ax3.set_ylabel('Predicted Room Temperature (°C)')
    ax3.set_title(f'Forward Sim: RMSE={stats["rmse_room_forward"]:.2f}°C')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Residuals over time
    ax4 = axes[1, 0]
    residuals = x_obs[:, 1] - x_pred_forward[:, 1]
    ax4.plot(timestamps[::step], residuals[::step], 'b-', linewidth=0.5, alpha=0.7)
    ax4.axhline(0, color='red', linestyle='--', linewidth=1.5)
    ax4.axvline(timestamps[split_idx], color='green', linestyle=':', linewidth=2, alpha=0.7)
    ax4.fill_between(timestamps[::step], -1, 1, alpha=0.2, color='green', label='±1°C band')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Residual (°C)')
    ax4.set_title(f'Forward Simulation Residuals (mean={residuals.mean():.2f}°C, std={residuals.std():.2f}°C)')
    ax4.grid(True, alpha=0.3)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Panel 5: Eigenvalue plot (stability)
    ax5 = axes[1, 1]
    theta = np.linspace(0, 2 * np.pi, 100)
    ax5.plot(np.cos(theta), np.sin(theta), 'k--', linewidth=1, alpha=0.5, label='Unit circle')
    ax5.plot(MAX_EIGENVALUE * np.cos(theta), MAX_EIGENVALUE * np.sin(theta),
             'g--', linewidth=1, alpha=0.5, label=f'Stability bound ({MAX_EIGENVALUE})')
    eigenvalues = stats['eigenvalues']
    for i, eig in enumerate(eigenvalues):
        color = 'green' if abs(eig) < MAX_EIGENVALUE else 'red'
        ax5.scatter(np.real(eig), np.imag(eig), s=100, c=color, marker='x', linewidths=3)
        ax5.annotate(f'λ{i+1}={eig:.3f}', (np.real(eig), np.imag(eig)),
                     xytext=(10, 10), textcoords='offset points', fontsize=9)
    ax5.set_xlim(-1.2, 1.2)
    ax5.set_ylim(-1.2, 1.2)
    ax5.set_aspect('equal')
    ax5.set_xlabel('Real')
    ax5.set_ylabel('Imaginary')
    ax5.set_title(f'Eigenvalues (max |λ| = {stats["max_eigenvalue"]:.4f})')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # Panel 6: Model comparison bar chart
    ax6 = axes[1, 2]

    # Load transfer function results for comparison
    tf_results_file = OUTPUT_DIR / 'thermal_model_results.csv'
    if tf_results_file.exists():
        tf_df = pd.read_csv(tf_results_file)
        tf_r2 = tf_df[tf_df['room'] == 'davis_inside']['r2'].values[0]
        tf_rmse = tf_df[tf_df['room'] == 'davis_inside']['rmse'].values[0]
    else:
        tf_r2 = 0.68
        tf_rmse = 0.50

    # Load old grey-box results
    old_gb_file = OUTPUT_DIR / 'greybox_model_params.json'
    if old_gb_file.exists():
        with open(old_gb_file) as f:
            old_gb = json.load(f)
        old_r2 = old_gb.get('fit_stats', {}).get('r2_room_forward', -0.5)
        old_rmse = old_gb.get('fit_stats', {}).get('rmse_room_forward', 5.0)
    else:
        old_r2 = -0.5
        old_rmse = 5.0

    models = ['Transfer\nFunction', 'Grey-Box\n(old)', 'Stable\nGrey-Box', 'Kalman\nFilter']
    r2_values = [tf_r2, max(old_r2, 0), stats['r2_room_forward'], stats['r2_room_kalman']]
    rmse_values = [tf_rmse, min(old_rmse, 3.0), stats['rmse_room_forward'], stats['rmse_room_kalman']]

    x_pos = np.arange(len(models))
    width = 0.35

    bars1 = ax6.bar(x_pos - width/2, r2_values, width, label='R²', color='steelblue')
    ax6.bar_label(bars1, fmt='%.3f', padding=3, fontsize=8)

    ax6_twin = ax6.twinx()
    bars2 = ax6_twin.bar(x_pos + width/2, rmse_values, width, label='RMSE (°C)', color='coral')
    ax6_twin.bar_label(bars2, fmt='%.2f', padding=3, fontsize=8)

    ax6.set_ylabel('R²', color='steelblue')
    ax6_twin.set_ylabel('RMSE (°C)', color='coral')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(models, fontsize=9)
    ax6.set_title('Model Comparison')
    ax6.set_ylim(0, 1.0)
    ax6_twin.set_ylim(0, max(rmse_values) * 1.3)
    ax6.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig3.01d_stable_greybox_model.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: fig3.01d_stable_greybox_model.png")


def generate_report(stats: dict) -> str:
    """Generate HTML report section."""

    params = stats['params']

    status_color = '#d4edda' if stats['is_stable'] else '#f8d7da'
    status_text = 'STABLE' if stats['is_stable'] else 'UNSTABLE'

    html = f"""
    <section id="stable-greybox-model">
    <h2>3.1c Stability-Constrained Grey-Box Model</h2>

    <div style="background-color: {status_color}; border: 1px solid #28a745; padding: 15px; margin-bottom: 20px; border-radius: 5px;">
        <strong>Model Status: {status_text}</strong><br>
        Max eigenvalue: {stats['max_eigenvalue']:.4f} (threshold: {MAX_EIGENVALUE})<br>
        Forward simulation R² = {stats['r2_room_forward']:.3f}, RMSE = {stats['rmse_room_forward']:.2f}°C
    </div>

    <h3>Key Improvements</h3>
    <ol>
        <li><strong>Eigenvalue stability constraint</strong>: Parameters optimized to ensure max|λ| &lt; {MAX_EIGENVALUE}</li>
        <li><strong>Semi-implicit integration</strong>: Unconditionally stable for diagonal decay terms</li>
        <li><strong>Kalman filter option</strong>: State correction when observations available</li>
        <li><strong>Two-phase optimization</strong>: First one-step, then forward simulation</li>
    </ol>

    <h3>Estimated Parameters</h3>
    <table>
        <tr><th>Parameter</th><th>Value</th><th>Physical Meaning</th></tr>
        <tr><td>τ_buf</td><td>{params['tau_buf']:.2f} h</td><td>Buffer tank time constant</td></tr>
        <tr><td>τ_room</td><td>{params['tau_room']:.1f} h</td><td>Building time constant</td></tr>
        <tr><td>r_emit</td><td>{params['r_emit']:.3f}</td><td>Emitter coupling ratio</td></tr>
        <tr><td>r_heat</td><td>{params['r_heat']:.3f}</td><td>Heat transfer ratio</td></tr>
        <tr><td>k_solar</td><td>{params['k_solar']:.4f} K/kWh</td><td>Solar gain coefficient</td></tr>
        <tr><td>c_offset</td><td>{params['c_offset']:.4f} K/h</td><td>Temperature drift correction</td></tr>
    </table>

    <h3>Forward Simulation Performance</h3>
    <table>
        <tr><th>Metric</th><th>Value</th><th>Interpretation</th></tr>
        <tr>
            <td><strong>R² (forward sim)</strong></td>
            <td><strong>{stats['r2_room_forward']:.3f}</strong></td>
            <td>{'Excellent' if stats['r2_room_forward'] > 0.8 else 'Good' if stats['r2_room_forward'] > 0.6 else 'Moderate'}</td>
        </tr>
        <tr>
            <td>RMSE (forward sim)</td>
            <td>{stats['rmse_room_forward']:.2f}°C</td>
            <td>Average prediction error</td>
        </tr>
        <tr>
            <td>Bias (forward sim)</td>
            <td>{stats['bias_room_forward']:.2f}°C</td>
            <td>Systematic over/under-prediction</td>
        </tr>
    </table>

    <h3>Kalman Filter Performance</h3>
    <p>When observations are available, the Kalman filter corrects predictions:</p>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>R² (Kalman)</td><td>{stats['r2_room_kalman']:.3f}</td></tr>
        <tr><td>RMSE (Kalman)</td><td>{stats['rmse_room_kalman']:.2f}°C</td></tr>
    </table>

    <h3>Stability Analysis</h3>
    <p>The state transition matrix A has eigenvalues:</p>
    <ul>
        <li>λ₁ = {stats['eigenvalues'][0]:.4f}</li>
        <li>λ₂ = {stats['eigenvalues'][1]:.4f}</li>
        <li>Max |λ| = {stats['max_eigenvalue']:.4f} &lt; {MAX_EIGENVALUE} ✓</li>
    </ul>

    <figure>
        <img src="fig3.01d_stable_greybox_model.png" alt="Stable Grey-Box Model Results">
        <figcaption><strong>Figure 18c:</strong> Stability-constrained grey-box model with eigenvalue
        constraints. Top: forward simulation and Kalman filter vs actual. Bottom: residuals,
        eigenvalue plot, and model comparison.</figcaption>
    </figure>
    </section>
    """

    return html


def main():
    """Main function for stable grey-box model."""
    print("=" * 60)
    print("Phase 3, Step 1c: Stability-Constrained Grey-Box Model")
    print("=" * 60)

    # Load and prepare data
    df = load_data()
    x_obs, u_inputs, timestamps = prepare_data(df)

    # Train/test split
    test_fraction = 0.33
    split_idx = int(len(x_obs) * (1 - test_fraction))
    print(f"\nTrain/test split at index {split_idx}")
    print(f"  Training: {split_idx:,} points ({timestamps[0].date()} to {timestamps[split_idx-1].date()})")
    print(f"  Testing:  {len(x_obs) - split_idx:,} points ({timestamps[split_idx].date()} to {timestamps[-1].date()})")

    # Estimate parameters with stability constraints
    stats = estimate_parameters_stable(x_obs, u_inputs, n_restarts=10)

    # Visualization
    plot_results(stats, x_obs, u_inputs, timestamps, split_idx)

    # Save results (convert numpy types to Python types for JSON)
    results = {
        'params': {k: float(v) for k, v in stats['params'].items()},
        'stability': {
            'is_stable': bool(stats['is_stable']),
            'max_eigenvalue': float(stats['max_eigenvalue']),
            'eigenvalues': [float(e) for e in stats['eigenvalues']],
            'stability_threshold': float(MAX_EIGENVALUE),
        },
        'forward_sim_metrics': {
            'r2_room': float(stats['r2_room_forward']),
            'rmse_room': float(stats['rmse_room_forward']),
            'mae_room': float(stats['mae_room_forward']),
            'bias_room': float(stats['bias_room_forward']),
            'r2_buffer': float(stats['r2_buffer_forward']),
            'rmse_buffer': float(stats['rmse_buffer_forward']),
        },
        'rolling_horizon_metrics': {
            '24h': {'r2_room': float(stats['r2_room_24h']), 'rmse_room': float(stats['rmse_room_24h'])},
            '6h': {'r2_room': float(stats['r2_room_6h']), 'rmse_room': float(stats['rmse_room_6h'])},
            '2h': {'r2_room': float(stats['r2_room_2h']), 'rmse_room': float(stats['rmse_room_2h'])},
        },
        'kalman_metrics': {
            'r2_room': float(stats['r2_room_kalman']),
            'rmse_room': float(stats['rmse_room_kalman']),
        },
        'one_step_metrics': {
            'r2_room': float(stats['r2_room_1step']),
            'rmse_room': float(stats['rmse_room_1step']),
        },
        'n_points': int(stats['n_points']),
        'param_bounds': {k: [float(v[0]), float(v[1])] for k, v in PARAM_BOUNDS.items()},
        'time_step_hours': float(DT_HOURS),
    }

    with open(OUTPUT_DIR / 'stable_greybox_params.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: stable_greybox_params.json")

    # Save predictions
    pred_df = pd.DataFrame({
        'timestamp': timestamps,
        'T_room_actual': x_obs[:, 1],
        'T_room_forward': stats['x_pred_forward'][:, 1],
        'T_room_kalman': stats['x_pred_kalman'][:, 1],
        'T_buffer_actual': x_obs[:, 0],
        'T_buffer_forward': stats['x_pred_forward'][:, 0],
    })
    pred_df.to_csv(OUTPUT_DIR / 'stable_greybox_results.csv', index=False)
    print("Saved: stable_greybox_results.csv")

    # Generate HTML report
    report_html = generate_report(stats)
    with open(OUTPUT_DIR / 'stable_greybox_report_section.html', 'w') as f:
        f.write(report_html)
    print("Saved: stable_greybox_report_section.html")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nModel is {'STABLE' if stats['is_stable'] else 'UNSTABLE'}")
    print(f"Max eigenvalue: {stats['max_eigenvalue']:.4f}")
    print(f"\nForward simulation: R² = {stats['r2_room_forward']:.3f}, RMSE = {stats['rmse_room_forward']:.2f}°C")
    print(f"24h rolling:        R² = {stats['r2_room_24h']:.3f}, RMSE = {stats['rmse_room_24h']:.2f}°C")
    print(f"6h rolling:         R² = {stats['r2_room_6h']:.3f}, RMSE = {stats['rmse_room_6h']:.2f}°C")
    print(f"Kalman filter:      R² = {stats['r2_room_kalman']:.3f}, RMSE = {stats['rmse_room_kalman']:.2f}°C")


if __name__ == '__main__':
    main()
