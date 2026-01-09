#!/usr/bin/env python3
"""
Phase 3, Step 1: Building Thermal Model

Estimates building thermal characteristics from sensor data:
- Heat loss coefficient (UA, W/K) from temperature decay curves
- Thermal mass (C, J/K) from heating response time
- Room temperature dynamics model

Model: RC network (resistance-capacitance)
    C * dT_in/dt = Q_heat + UA_solar * S - UA * (T_in - T_out)

where:
    T_in = indoor temperature (°C)
    T_out = outdoor temperature (°C)
    Q_heat = heating power (W)
    S = solar irradiance proxy (from PV generation)
    UA = heat loss coefficient (W/K)
    UA_solar = solar gain coefficient (W/(W/m²))
    C = thermal mass (J/K)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / 'output' / 'phase1'
OUTPUT_DIR = PROJECT_ROOT / 'output' / 'phase3'
OUTPUT_DIR.mkdir(exist_ok=True)

# Target sensors for thermal model (weighted combination)
TARGET_SENSORS = [
    'davis_inside_temperature',  # 40%
    'office1_temperature',       # 30%
    'atelier_temperature',       # 10%
    'studio_temperature',        # 10%
    'simlab_temperature',        # 10%
]

# Weights for weighted objective
SENSOR_WEIGHTS = {
    'davis_inside_temperature': 0.40,
    'office1_temperature': 0.30,
    'atelier_temperature': 0.10,
    'studio_temperature': 0.10,
    'simlab_temperature': 0.10,
}

# Reference room for primary analysis (most data, occupied space)
PRIMARY_ROOM = 'office1_temperature'


def load_data():
    """Load and prepare data for thermal modeling."""
    print("Loading data for thermal modeling...")

    # Load integrated dataset (has everything merged at 15-min intervals)
    df = pd.read_parquet(PROCESSED_DIR / 'integrated_overlap_only.parquet')
    df.index = pd.to_datetime(df.index)

    # Load raw sensor data for better coverage
    heating_raw = pd.read_parquet(PROCESSED_DIR / 'sensors_heating.parquet')
    heating_raw['datetime'] = pd.to_datetime(heating_raw['datetime'], utc=True)

    print(f"  Integrated dataset: {len(df):,} rows ({df.index.min()} to {df.index.max()})")

    return df, heating_raw


def pivot_sensor_data(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot sensor data from long to wide format."""
    if df.empty:
        return pd.DataFrame()

    pivoted = df.pivot_table(
        values='value',
        index='datetime',
        columns='entity_id',
        aggfunc='mean'
    )
    return pivoted


def prepare_thermal_data(df: pd.DataFrame, heating_raw: pd.DataFrame) -> pd.DataFrame:
    """Prepare dataset for thermal analysis."""
    print("\nPreparing thermal analysis data...")

    # Get heating sensors pivoted
    heating = pivot_sensor_data(heating_raw)

    # Resample heating data to 15-min to match integrated dataset
    heating_15min = heating.resample('15min').mean()

    # Key columns
    thermal_data = pd.DataFrame(index=df.index)

    # Outdoor temperature from heat pump sensor
    outdoor_col = 'stiebel_eltron_isg_outdoor_temperature'
    if outdoor_col in heating_15min.columns:
        thermal_data['T_out'] = heating_15min.loc[df.index, outdoor_col].values
    else:
        print(f"  Warning: {outdoor_col} not found")
        return pd.DataFrame()

    # Room temperatures - only include target sensors
    room_temp_cols = [c for c in TARGET_SENSORS if c in df.columns]

    for col in room_temp_cols:
        thermal_data[col] = df[col]

    print(f"  Target sensors found: {room_temp_cols}")

    # Heating status and power
    if 'stiebel_eltron_isg_is_heating' in df.columns:
        thermal_data['is_heating'] = df['stiebel_eltron_isg_is_heating']

    # Flow temperature (proxy for heating power)
    flow_col = 'stiebel_eltron_isg_flow_temperature_wp1'
    if flow_col in df.columns:
        thermal_data['T_flow'] = df[flow_col]

    # Heating circuit 2 temperature - key heating effort indicator
    hk2_col = 'stiebel_eltron_isg_actual_temperature_hk_2'
    if hk2_col in df.columns:
        thermal_data['T_hk2'] = df[hk2_col]
    else:
        print(f"  Warning: {hk2_col} not found - using T_flow as fallback")
        if 'T_flow' in thermal_data.columns:
            thermal_data['T_hk2'] = thermal_data['T_flow']

    # Buffer tank temperature
    buffer_col = 'stiebel_eltron_isg_actual_temperature_buffer'
    if buffer_col in df.columns:
        thermal_data['T_buffer'] = df[buffer_col]

    # Heating energy consumed (cumulative, need to diff)
    consumed_col = 'stiebel_eltron_isg_consumed_heating_today'
    if consumed_col in df.columns:
        thermal_data['heating_energy_today'] = df[consumed_col]

    # PV generation as solar proxy
    if 'pv_generation_kwh' in df.columns:
        thermal_data['pv_generation'] = df['pv_generation_kwh']

    print(f"  Room temperature columns: {len(room_temp_cols)}")
    print(f"  Total columns: {len(thermal_data.columns)}")
    print(f"  Valid outdoor temp: {thermal_data['T_out'].notna().sum():,} / {len(thermal_data):,}")

    return thermal_data


def compute_weighted_temperature(thermal_data: pd.DataFrame) -> pd.Series:
    """
    Compute weighted average indoor temperature from target sensors.

    Weights:
      - davis_inside_temperature: 40%
      - office1_temperature: 30%
      - atelier_temperature: 10%
      - studio_temperature: 10%
      - simlab_temperature: 10%
    """
    print("\nComputing weighted indoor temperature...")

    weighted_sum = pd.Series(0.0, index=thermal_data.index)
    weight_sum = pd.Series(0.0, index=thermal_data.index)

    for sensor, weight in SENSOR_WEIGHTS.items():
        if sensor in thermal_data.columns:
            valid_mask = thermal_data[sensor].notna()
            weighted_sum[valid_mask] += thermal_data.loc[valid_mask, sensor] * weight
            weight_sum[valid_mask] += weight
            valid_count = valid_mask.sum()
            print(f"  {sensor}: {valid_count:,} valid points (weight={weight:.0%})")

    # Normalize by actual weight sum (handles missing sensors)
    T_weighted = weighted_sum / weight_sum
    T_weighted[weight_sum == 0] = np.nan

    print(f"  Weighted temperature: {T_weighted.notna().sum():,} valid points")
    print(f"  Mean: {T_weighted.mean():.2f}°C, Std: {T_weighted.std():.2f}°C")

    return T_weighted


def estimate_heat_loss_with_heating(thermal_data: pd.DataFrame, room_col: str) -> dict:
    """
    Estimate thermal parameters using heating circuit temperature as heating input.

    Model: dT_room/dt = a*(T_hk2 - T_room) - b*(T_room - T_out) + c*PV

    Where T_hk2 (heating circuit 2 temp) is a proxy for heating effort.
    Higher T_hk2 = more heat being delivered to the room.

    This approach accounts for continuous heating operation (day and night).
    """
    print(f"\nEstimating thermal parameters for {room_col}...")

    # Required columns
    hk2_col = 'T_hk2'
    required = [room_col, 'T_out', hk2_col, 'pv_generation']
    available = [c for c in required if c in thermal_data.columns]

    if len(available) < 3:
        print(f"  Missing columns: {set(required) - set(available)}")
        return {}

    df = thermal_data[available].copy().dropna()

    if len(df) < 100:
        print(f"  Not enough data: {len(df)} points")
        return {}

    # Temperature change (next step - current)
    df['dT'] = df[room_col].diff().shift(-1)

    # Driving forces
    df['delta_T_hk2'] = df[hk2_col] - df[room_col]  # Heating effort (HK2 - room)
    df['delta_T_out'] = df[room_col] - df['T_out']   # Heat loss (room - outdoor)

    df = df.dropna()

    if len(df) < 50:
        print(f"  Not enough valid data after diff: {len(df)} points")
        return {}

    # Build feature matrix
    features = ['delta_T_hk2', 'delta_T_out']
    if 'pv_generation' in df.columns:
        features.append('pv_generation')

    X = df[features].values
    y = df['dT'].values

    # Remove extreme outliers (> 3 std)
    mask = np.abs(y - y.mean()) < 3 * y.std()
    X = X[mask]
    y = y[mask]

    if len(y) < 30:
        print(f"  Not enough data after outlier removal: {len(y)} points")
        return {}

    # Regression
    reg = LinearRegression().fit(X, y)
    y_pred = reg.predict(X)

    # Extract coefficients
    heating_coef = reg.coef_[0]  # Response to heating (T_hk2 - T_room)
    loss_coef = reg.coef_[1]     # Response to outdoor delta (should be negative)
    solar_coef = reg.coef_[2] if len(features) > 2 else 0

    # Time constant from loss coefficient
    # dT = -loss_coef * (T_room - T_out) * dt
    # For loss: dT/dt = -(1/tau) * (T_room - T_out)
    # So: loss_coef = dt / tau, tau = dt / loss_coef
    dt_hours = 0.25  # 15 min
    abs_loss = abs(loss_coef)
    if abs_loss > 0.0001:
        time_constant_hours = dt_hours / abs_loss
    else:
        time_constant_hours = 100.0  # Cap for very stable temps

    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    result = {
        'room': room_col,
        'heating_coef': heating_coef,
        'loss_coef': loss_coef,
        'solar_coef': solar_coef,
        'intercept': reg.intercept_,
        'time_constant_hours': time_constant_hours,
        'r2': r2,
        'rmse': rmse,
        'n_points': len(y)
    }

    print(f"  Heating coef (T_hk2 - T_room): {heating_coef:.6f} K/(15min)/K")
    print(f"  Loss coef (T_room - T_out): {loss_coef:.6f} K/(15min)/K")
    print(f"  Solar coef: {solar_coef:.6f} K/(15min)/kWh")
    print(f"  Time constant: {time_constant_hours:.1f} hours")
    print(f"  R²: {r2:.3f}")
    print(f"  RMSE: {rmse:.4f} K")
    print(f"  Data points: {len(y)}")

    return result


def estimate_thermal_response(thermal_data: pd.DataFrame, room_col: str) -> dict:
    """
    Estimate thermal response characteristics during heating periods.

    Looks at heating rise rate when heat pump is active.
    """
    print(f"\nEstimating thermal response for {room_col}...")

    df = thermal_data[[room_col, 'T_out', 'T_flow', 'is_heating']].dropna()

    if len(df) < 50:
        print("  Not enough data")
        return {}

    # During heating: temperature rise rate
    df['dT'] = df[room_col].diff()
    df['delta_T_out'] = df[room_col] - df['T_out']
    df['delta_T_flow'] = df['T_flow'] - df[room_col]  # Flow - room (driving force)

    heating_on = df['is_heating'] > 0.5
    heating_data = df[heating_on].dropna()

    if len(heating_data) < 20:
        print("  Not enough heating data")
        return {}

    # During heating: dT/dt = (UA_flow * (T_flow - T_in) - UA_out * (T_in - T_out)) / C
    # Simplified: dT/dt = a * (T_flow - T_in) + b * (T_out - T_in)

    X = heating_data[['delta_T_flow', 'delta_T_out']].values
    y = heating_data['dT'].values

    # Remove outliers
    mask = np.abs(y) < np.percentile(np.abs(y), 95)
    X = X[mask]
    y = y[mask]

    if len(y) < 10:
        return {}

    reg = LinearRegression().fit(X, y)
    y_pred = reg.predict(X)

    result = {
        'room': room_col,
        'heating_coef': reg.coef_[0],  # Response to flow-room delta
        'loss_coef': reg.coef_[1],     # Response to outdoor-room delta
        'intercept': reg.intercept_,
        'r2': r2_score(y, y_pred),
        'rmse': np.sqrt(mean_squared_error(y, y_pred)),
        'n_points': len(y),
        'mean_rise_rate': y.mean() * 4  # K per hour (from per 15 min)
    }

    print(f"  Heating coefficient: {reg.coef_[0]:.6f}")
    print(f"  Loss coefficient: {reg.coef_[1]:.6f}")
    print(f"  R²: {result['r2']:.3f}")
    print(f"  Mean rise rate: {result['mean_rise_rate']:.2f} K/h when heating")

    return result


def build_simple_rc_model(thermal_data: pd.DataFrame, room_col: str) -> dict:
    """
    Build a simple RC (resistance-capacitance) thermal model.

    State equation (discrete, 15-min steps):
        T[k+1] = T[k] + dt/C * (Q_heat[k] - UA*(T[k] - T_out[k]) + S*PV[k])

    Parameters to estimate: UA (heat loss), response_time (C/UA)
    """
    print(f"\nBuilding RC model for {room_col}...")

    df = thermal_data[[room_col, 'T_out', 'T_flow', 'pv_generation']].copy()
    df = df.dropna()

    if len(df) < 100:
        print("  Not enough data")
        return {}

    # Estimate heating power from flow temperature
    # Q_heat ~ k * max(T_flow - T_room, 0)
    df['delta_T_flow'] = np.maximum(df['T_flow'] - df[room_col], 0)
    df['delta_T_out'] = df[room_col] - df['T_out']
    df['dT'] = df[room_col].diff().shift(-1)  # Next step change

    df = df.dropna()

    # Simple linear model: dT = a * delta_T_flow - b * delta_T_out + c * PV
    X = df[['delta_T_flow', 'delta_T_out', 'pv_generation']].values
    y = df['dT'].values

    # Remove extreme outliers
    mask = np.abs(y) < np.percentile(np.abs(y), 98)
    X = X[mask]
    y = y[mask]

    reg = LinearRegression().fit(X, y)
    y_pred = reg.predict(X)

    # Coefficients interpretation:
    # dT = a * (T_flow - T_in) - b * (T_in - T_out) + c * PV
    # a = response to heating (heating gain / C)
    # b = heat loss rate (UA / C)
    # c = solar gain coefficient

    heating_coef = reg.coef_[0]
    loss_coef = reg.coef_[1]  # Already negative in formulation
    solar_coef = reg.coef_[2]

    # Time constant estimation: tau = C / UA
    # From loss_coef (per 15 min): loss_coef = dt * UA / C
    # tau = dt / loss_coef (in 15-min units)
    # Note: loss_coef should be positive for physical meaning
    dt_hours = 0.25
    abs_loss = abs(loss_coef)
    if abs_loss > 0.001:
        time_constant_hours = dt_hours / abs_loss
    else:
        time_constant_hours = 100.0  # Cap at 100 hours for very stable temps

    result = {
        'room': room_col,
        'heating_coef': heating_coef,
        'loss_coef': loss_coef,
        'solar_coef': solar_coef,
        'intercept': reg.intercept_,
        'time_constant_hours': time_constant_hours,
        'r2': r2_score(y, y_pred),
        'rmse': np.sqrt(mean_squared_error(y, y_pred)),
        'n_points': len(y)
    }

    print(f"  Heating coef: {heating_coef:.6f} K/(15min)/K")
    print(f"  Loss coef: {loss_coef:.6f} K/(15min)/K")
    print(f"  Solar coef: {solar_coef:.4f} K/(15min)/kWh")
    print(f"  Time constant: {time_constant_hours:.1f} hours")
    print(f"  R²: {result['r2']:.3f}")
    print(f"  RMSE: {result['rmse']:.4f} K")

    return result


def simulate_temperature(thermal_data: pd.DataFrame, room_col: str,
                        model_params: dict) -> pd.DataFrame:
    """Simulate temperature using the RC model and compare with actual."""
    df = thermal_data[[room_col, 'T_out', 'T_flow', 'pv_generation']].copy()
    df = df.dropna()

    if len(df) < 10:
        return pd.DataFrame()

    # Extract parameters
    a = model_params['heating_coef']
    b = model_params['loss_coef']
    c = model_params['solar_coef']

    # Initialize simulation
    T_sim = np.zeros(len(df))
    T_sim[0] = df[room_col].iloc[0]

    T_out = df['T_out'].values
    T_flow = df['T_flow'].values
    PV = df['pv_generation'].values

    # Simulate with stability bounds
    T_actual = df[room_col].values

    for k in range(len(df) - 1):
        delta_flow = max(T_flow[k] - T_sim[k], 0)
        delta_out = T_sim[k] - T_out[k]

        dT = a * delta_flow - b * delta_out + c * PV[k]

        # Stability: limit change to reasonable bounds
        dT = np.clip(dT, -2.0, 2.0)  # Max 2K change per 15 min

        T_sim[k+1] = T_sim[k] + dT

        # Reset if simulation diverges too far
        if abs(T_sim[k+1] - T_actual[k+1]) > 10:
            T_sim[k+1] = T_actual[k+1]

    result = pd.DataFrame({
        'actual': df[room_col].values,
        'simulated': T_sim,
        'T_out': T_out,
        'T_flow': T_flow
    }, index=df.index)

    return result


def plot_thermal_analysis(thermal_data: pd.DataFrame, room_results: dict,
                         simulation: pd.DataFrame) -> None:
    """Create visualization of thermal analysis results."""
    print("\nCreating thermal analysis plots...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    room_col = room_results.get('room', PRIMARY_ROOM)

    # Panel 1: Temperature time series with simulation
    ax = axes[0, 0]
    if not simulation.empty:
        ax.plot(simulation.index, simulation['actual'],
                label='Actual', color='blue', alpha=0.7, linewidth=0.8)
        ax.plot(simulation.index, simulation['simulated'],
                label='Simulated', color='red', alpha=0.7, linewidth=0.8, linestyle='--')
        ax.plot(simulation.index, simulation['T_out'],
                label='Outdoor', color='green', alpha=0.5, linewidth=0.8)
        ax.legend(loc='upper right')

        rmse = np.sqrt(mean_squared_error(simulation['actual'], simulation['simulated']))
        ax.text(0.02, 0.98, f'RMSE: {rmse:.2f}°C', transform=ax.transAxes,
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_ylabel('Temperature (°C)')
    ax.set_title(f'Temperature Simulation: {room_col}')
    ax.grid(True, alpha=0.3)

    # Panel 2: Actual vs Predicted scatter
    ax = axes[0, 1]
    if not simulation.empty:
        ax.scatter(simulation['actual'], simulation['simulated'],
                   alpha=0.3, s=5, c='blue')

        # Perfect prediction line
        temp_range = [simulation['actual'].min(), simulation['actual'].max()]
        ax.plot(temp_range, temp_range, 'r--', linewidth=2, label='Perfect fit')

        # Regression line
        slope, intercept, r_value, _, _ = stats.linregress(
            simulation['actual'], simulation['simulated'])
        ax.plot(temp_range, [slope*t + intercept for t in temp_range],
                'g-', linewidth=1.5, label=f'Fit (R²={r_value**2:.3f})')

        ax.legend()

    ax.set_xlabel('Actual Temperature (°C)')
    ax.set_ylabel('Simulated Temperature (°C)')
    ax.set_title('Model Validation')
    ax.grid(True, alpha=0.3)

    # Panel 3: Temperature response during heating
    ax = axes[1, 0]
    if 'T_flow' in thermal_data.columns and room_col in thermal_data.columns:
        df = thermal_data[[room_col, 'T_flow', 'is_heating']].dropna()
        heating_on = df['is_heating'] > 0.5 if 'is_heating' in df.columns else pd.Series(True, index=df.index)

        delta_T = df['T_flow'] - df[room_col]
        dT = df[room_col].diff() * 4  # K per hour

        if heating_on.any():
            ax.scatter(delta_T[heating_on], dT[heating_on],
                       alpha=0.3, s=10, c='red', label='Heating ON')
        if (~heating_on).any():
            ax.scatter(delta_T[~heating_on], dT[~heating_on],
                       alpha=0.2, s=5, c='blue', label='Heating OFF')

        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.legend()

    ax.set_xlabel('Flow - Room Temperature (K)')
    ax.set_ylabel('Room Temp Change Rate (K/h)')
    ax.set_title('Heating Response')
    ax.grid(True, alpha=0.3)

    # Panel 4: Heat loss vs temperature difference
    ax = axes[1, 1]
    if 'T_out' in thermal_data.columns and room_col in thermal_data.columns:
        df = thermal_data[[room_col, 'T_out', 'is_heating', 'pv_generation']].dropna()

        # Night, heating off periods
        night_off = (df['pv_generation'] < 0.01) & (df['is_heating'] < 0.5)

        if night_off.any():
            delta_T_out = df.loc[night_off, room_col] - df.loc[night_off, 'T_out']
            dT = df.loc[night_off, room_col].diff() * 4  # K per hour

            ax.scatter(delta_T_out, dT, alpha=0.4, s=10, c='blue')

            # Regression line
            mask = delta_T_out.notna() & dT.notna()
            if mask.sum() > 10:
                slope, intercept, r_value, _, _ = stats.linregress(
                    delta_T_out[mask], dT[mask])
                x_range = np.array([delta_T_out.min(), delta_T_out.max()])
                ax.plot(x_range, slope * x_range + intercept, 'r-', linewidth=2,
                        label=f'Slope: {slope:.4f} (R²={r_value**2:.3f})')

                # Estimate time constant
                if slope < 0:
                    tau = -1 / slope
                    ax.text(0.02, 0.02, f'Time constant: {tau:.1f} h',
                            transform=ax.transAxes, fontsize=10,
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

                ax.legend()

        ax.axhline(y=0, color='black', linewidth=0.5)

    ax.set_xlabel('Indoor - Outdoor Temperature (K)')
    ax.set_ylabel('Room Temp Change Rate (K/h)')
    ax.set_title('Heat Loss During Night (Heating Off)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig13_thermal_model.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("  Saved: fig13_thermal_model.png")


def analyze_all_rooms(thermal_data: pd.DataFrame) -> pd.DataFrame:
    """Analyze thermal characteristics for target sensors only."""
    print("\n" + "="*60)
    print("Analyzing thermal characteristics for target sensors")
    print("="*60)

    # Only analyze target sensors that exist in the data
    room_cols = [c for c in TARGET_SENSORS if c in thermal_data.columns]

    results = []

    for room_col in room_cols:
        valid_count = thermal_data[room_col].notna().sum()
        weight = SENSOR_WEIGHTS.get(room_col, 0)
        if valid_count < 500:  # Need at least ~3 days of data
            print(f"\nSkipping {room_col}: only {valid_count} valid points")
            continue

        # Estimate thermal parameters using HK2 as heating input
        thermal_result = estimate_heat_loss_with_heating(thermal_data, room_col)

        if thermal_result:
            results.append({
                'room': room_col.replace('_temperature', ''),
                'weight': weight,
                'data_points': valid_count,
                'time_constant_h': thermal_result.get('time_constant_hours'),
                'heating_coef': thermal_result.get('heating_coef'),
                'loss_coef': thermal_result.get('loss_coef'),
                'solar_coef': thermal_result.get('solar_coef'),
                'r2': thermal_result.get('r2'),
                'rmse': thermal_result.get('rmse')
            })

    if results:
        df_results = pd.DataFrame(results)
        # Sort by weight (highest first)
        df_results = df_results.sort_values('weight', ascending=False)
        return df_results

    return pd.DataFrame()


def analyze_weighted_temperature(thermal_data: pd.DataFrame, T_weighted: pd.Series) -> dict:
    """
    Analyze thermal characteristics using the weighted indoor temperature.

    This provides a single combined model using the weighted objective.
    """
    print("\n" + "="*60)
    print("Analyzing weighted indoor temperature model")
    print("="*60)

    # Add weighted temperature to thermal data
    thermal_data_with_weighted = thermal_data.copy()
    thermal_data_with_weighted['T_weighted'] = T_weighted

    # Use estimate_heat_loss_with_heating with weighted temperature
    result = estimate_heat_loss_with_heating(thermal_data_with_weighted, 'T_weighted')

    if result:
        result['room'] = 'weighted_indoor'

    return result


def generate_report(room_results: pd.DataFrame, weighted_model: dict,
                   simulation_rmse: float) -> str:
    """Generate HTML report section for thermal model."""

    # Weighted model results (primary)
    if weighted_model:
        weighted_time_constant = weighted_model.get('time_constant_hours', 'N/A')
        weighted_r2 = weighted_model.get('r2', 'N/A')
        weighted_heating_coef = weighted_model.get('heating_coef', 0)
        weighted_loss_coef = weighted_model.get('loss_coef', 0)
        weighted_rmse = weighted_model.get('rmse', 'N/A')
    else:
        weighted_time_constant = 'N/A'
        weighted_r2 = 'N/A'
        weighted_heating_coef = 0
        weighted_loss_coef = 0
        weighted_rmse = 'N/A'

    # Build weights description
    weights_desc = ", ".join([f"{k.replace('_temperature', '')}: {v:.0%}"
                              for k, v in SENSOR_WEIGHTS.items()])

    html = f"""
    <section id="thermal-model">
    <h2>3.1 Building Thermal Model</h2>

    <h3>Methodology</h3>
    <p>Estimated building thermal characteristics using heating circuit temperature (T_hk2) as a proxy for heating effort.
    This approach accounts for <strong>continuous heating operation</strong> (the heat pump runs day and night, not just during daytime).</p>

    <p><strong>Weighted Indoor Temperature:</strong> The model uses a weighted combination of target sensors:<br>
    {weights_desc}</p>

    <pre>
    dT_room/dt = a × (T_hk2 - T_room) - b × (T_room - T_out) + c × PV

    where:
        T_room = weighted indoor temperature (°C)
        T_out  = outdoor temperature (°C)
        T_hk2  = heating circuit 2 temperature (proxy for heating effort)
        PV     = solar gain proxy (from PV generation)
        a      = heating coefficient (response to heating effort)
        b      = loss coefficient (heat loss to outdoor)
        c      = solar gain coefficient
        tau    = 1/b = thermal time constant (hours)
    </pre>

    <p><strong>Key insight:</strong> T_hk2 varies from ~30°C at night (eco mode) to ~36°C in morning (comfort mode),
    providing a continuous measure of heating input.</p>

    <h3>Weighted Model Results (Primary)</h3>
    <table>
        <tr><th>Metric</th><th>Value</th><th>Interpretation</th></tr>
        <tr>
            <td>Thermal time constant</td>
            <td>{weighted_time_constant:.1f} hours</td>
            <td>Time for temperature to decay to 37% of initial difference</td>
        </tr>
        <tr>
            <td>Heating coefficient (a)</td>
            <td>{weighted_heating_coef:.6f}</td>
            <td>Weighted temp rise per K of (T_hk2 - T_room) per 15 min</td>
        </tr>
        <tr>
            <td>Loss coefficient (b)</td>
            <td>{weighted_loss_coef:.6f}</td>
            <td>Weighted temp drop per K of (T_room - T_out) per 15 min</td>
        </tr>
        <tr>
            <td>Model R²</td>
            <td>{weighted_r2:.3f}</td>
            <td>Variance explained by weighted model</td>
        </tr>
        <tr>
            <td>Model RMSE</td>
            <td>{weighted_rmse:.4f}°C</td>
            <td>Typical prediction error</td>
        </tr>
        <tr>
            <td>Simulation RMSE</td>
            <td>{simulation_rmse:.2f}°C</td>
            <td>Cumulative simulation error</td>
        </tr>
    </table>

    <h3>Individual Sensor Results</h3>
    """

    if not room_results.empty:
        html += "<table>\n"
        html += "<tr><th>Sensor</th><th>Weight</th><th>Data Points</th><th>Time Constant (h)</th>"
        html += "<th>Heating Coef</th><th>Loss Coef</th><th>R²</th></tr>\n"

        for _, row in room_results.iterrows():
            html += f"<tr>"
            html += f"<td>{row['room']}</td>"
            html += f"<td>{row['weight']:.0%}</td>"
            html += f"<td>{row['data_points']:,}</td>"
            html += f"<td>{row['time_constant_h']:.1f}</td>"
            html += f"<td>{row['heating_coef']:.6f}</td>"
            html += f"<td>{row['loss_coef']:.6f}</td>"
            html += f"<td>{row['r2']:.3f}</td>"
            html += f"</tr>\n"

        html += "</table>\n"

    html += """
    <h3>Model Coefficients Interpretation</h3>
    <ul>
        <li><strong>Heating coefficient (a)</strong>: Response rate to heating circuit temperature.
            Higher values indicate faster heating response.</li>
        <li><strong>Loss coefficient</strong>: Heat loss rate per degree temperature difference.
            Higher values indicate poorer insulation.</li>
        <li><strong>Solar coefficient</strong>: Temperature rise per kWh of PV generation.
            Captures passive solar gains through windows.</li>
        <li><strong>Time constant</strong>: Building thermal inertia.
            Longer = more stable temperatures, slower response to heating/cooling.</li>
    </ul>
    """

    # Add implications with weighted time constant
    tc = weighted_time_constant if isinstance(weighted_time_constant, (int, float)) else 50

    html += f"""
    <h3>Implications for Optimization</h3>
    <ul>
        <li><strong>Pre-heating timing</strong>: With ~{tc:.0f}h time constant,
            rooms need 2-3× this time to fully respond to setpoint changes.</li>
        <li><strong>Solar preheating</strong>: Can reduce heating demand by timing comfort periods
            to coincide with solar availability.</li>
        <li><strong>Night setback recovery</strong>: Recovery from eco to comfort mode takes
            approximately {tc*0.7:.0f}-{tc*1.5:.0f} hours depending on
            outdoor temperature.</li>
    </ul>

    <figure>
        <img src="fig13_thermal_model.png" alt="Thermal Model Analysis">
        <figcaption>Thermal model validation: temperature simulation (top-left),
        actual vs predicted (top-right), heating response (bottom-left),
        heat loss characterization (bottom-right).</figcaption>
    </figure>
    </section>
    """

    return html


def main():
    """Main function for thermal model estimation."""
    print("="*60)
    print("Phase 3, Step 1: Building Thermal Model")
    print("="*60)

    # Load data
    df, heating_raw = load_data()

    # Prepare thermal analysis data
    thermal_data = prepare_thermal_data(df, heating_raw)

    if thermal_data.empty:
        print("ERROR: Could not prepare thermal data")
        return

    # Compute weighted indoor temperature
    T_weighted = compute_weighted_temperature(thermal_data)

    # Analyze weighted temperature model (primary)
    weighted_result = analyze_weighted_temperature(thermal_data, T_weighted)

    # Build RC model for simulation using weighted temperature
    thermal_data_with_weighted = thermal_data.copy()
    thermal_data_with_weighted['T_weighted'] = T_weighted

    rc_result = build_simple_rc_model(thermal_data_with_weighted, 'T_weighted')

    # Simulate temperature using weighted model
    simulation = pd.DataFrame()
    simulation_rmse = float('nan')
    if rc_result:
        simulation = simulate_temperature(thermal_data_with_weighted, 'T_weighted', rc_result)
        if not simulation.empty:
            simulation_rmse = np.sqrt(mean_squared_error(
                simulation['actual'], simulation['simulated']))

    # Analyze individual target sensors
    room_results = analyze_all_rooms(thermal_data)

    # Create visualizations using weighted temperature
    plot_thermal_analysis(thermal_data_with_weighted, weighted_result if weighted_result else rc_result, simulation)

    # Save results
    if not room_results.empty:
        room_results.to_csv(OUTPUT_DIR / 'thermal_model_results.csv', index=False)
        print(f"\nSaved: thermal_model_results.csv")

    # Generate report section with weighted model
    report_html = generate_report(room_results, weighted_result, simulation_rmse)
    with open(OUTPUT_DIR / 'thermal_model_report_section.html', 'w') as f:
        f.write(report_html)
    print("Saved: thermal_model_report_section.html")

    # Summary
    print("\n" + "="*60)
    print("THERMAL MODEL SUMMARY")
    print("="*60)

    if weighted_result:
        tc = weighted_result.get('time_constant_hours', 100)
        r2 = weighted_result.get('r2', 0)
        heating_coef = weighted_result.get('heating_coef', 0)
        loss_coef = weighted_result.get('loss_coef', 0)
        print(f"\nWeighted Indoor Temperature Model:")
        print(f"  Weights: davis_inside=40%, office1=30%, atelier/studio/simlab=10% each")
        print(f"  Time constant: {tc:.1f} hours")
        print(f"  Heating coef (T_hk2 - T_room): {heating_coef:.6f}")
        print(f"  Loss coef (T_room - T_out): {loss_coef:.6f}")
        print(f"  Model R²: {r2:.3f}")
        if not np.isnan(simulation_rmse) and simulation_rmse < 100:
            print(f"  Simulation RMSE: {simulation_rmse:.2f}°C")

    if not room_results.empty:
        print(f"\nIndividual sensors analyzed: {len(room_results)}")
        for _, row in room_results.iterrows():
            print(f"  {row['room']} (weight={row['weight']:.0%}): "
                  f"tau={row['time_constant_h']:.1f}h, R²={row['r2']:.3f}")


if __name__ == '__main__':
    main()
