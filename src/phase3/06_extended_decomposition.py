#!/usr/bin/env python3
"""
Phase 3: Extended Model Decomposition

Creates comprehensive decomposition figures showing:
1. Room temperature (actual vs predicted)
2. Outdoor temperature
3. Heating effort contribution
4. Solar/PV contribution
5. Battery state of charge
6. Power consumption
7. Grid feed-in
8. Grid import
9. Heat pump COP

This replaces fig3.01c and is placed after fig3.04.
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import json

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'output' / 'phase1'
OUTPUT_DIR = PROJECT_ROOT / 'output' / 'phase3'
WEEKLY_DIR = OUTPUT_DIR / 'weekly_decomposition'
OUTPUT_DIR.mkdir(exist_ok=True)
WEEKLY_DIR.mkdir(exist_ok=True)

# Import shared constants
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
from shared import ANALYSIS_START_DATE

# Figure style
plt.rcParams.update({
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 12,
})

COLORS = {
    'actual': '#1e4557',
    'predicted': '#2a9749',
    'outdoor': '#079bca',
    'heating': '#E85A4F',
    'solar': '#FFB800',
    'battery': '#2a9749',
    'consumption': '#9156b4',
    'grid_import': '#E85A4F',
    'grid_export': '#2a9749',
    'cop': '#1e4557',
    'baseline': '#bababa',
}

# ============================================================================
# INTRA-DAY MODEL PARAMETERS (estimated from data)
# ============================================================================
BATTERY_PARAMS = {
    'capacity_kwh': 11.0,        # Battery capacity
    'max_charge_kw': 5.0,        # Max charge rate
    'max_discharge_kw': 5.0,     # Max discharge rate
    'efficiency': 0.84,          # Round-trip efficiency (84%)
    'initial_soc_pct': 50.0,     # Initial SoC as % of capacity
}

# COP model coefficients (from Phase 3 heat pump model)
COP_MODEL = {
    'intercept': 5.93,
    'coef_t_outdoor': 0.13,
    'coef_t_hk2': -0.08,
}

# Heating curve parameters (from Phase 2)
HEATING_CURVE = {
    't_ref_comfort': 21.32,
    't_ref_eco': 19.18,
    'default_setpoint': 20.0,
    'default_curve_rise': 1.08,
}

# Tariff parameters (Primeo Energie)
TARIFF_PARAMS = {
    'high_rate_rp': 32.6,        # High tariff (Rp/kWh)
    'low_rate_rp': 26.0,         # Low tariff (Rp/kWh) - approximate
    'feedin_rate_rp': 13.0,      # Feed-in rate (Rp/kWh) with HKN
}


def load_data():
    """Load all required datasets."""
    print("Loading data...")

    # Load integrated dataset
    integrated = pd.read_parquet(DATA_DIR / 'integrated_dataset.parquet')
    integrated.index = pd.to_datetime(integrated.index)
    if integrated.index.tz is not None:
        integrated.index = integrated.index.tz_localize(None)

    # Load energy balance data
    energy = pd.read_parquet(DATA_DIR / 'energy_balance_15min.parquet')
    energy.index = pd.to_datetime(energy.index)
    if energy.index.tz is not None:
        energy.index = energy.index.tz_localize(None)

    # Load heating sensors - need to pivot to wide format
    heating_raw = pd.read_parquet(DATA_DIR / 'sensors_heating.parquet')
    if 'datetime' in heating_raw.columns:
        # Pivot to wide format with datetime as index
        heating = heating_raw.pivot_table(
            index='datetime', columns='entity_id', values='value', aggfunc='first'
        )
        heating.index = pd.to_datetime(heating.index)
        if heating.index.tz is not None:
            heating.index = heating.index.tz_localize(None)
    else:
        heating = heating_raw
        heating.index = pd.to_datetime(heating.index)
        if heating.index.tz is not None:
            heating.index = heating.index.tz_localize(None)

    print(f"  Integrated: {len(integrated)} rows")
    print(f"  Energy: {len(energy)} rows")
    print(f"  Heating: {len(heating)} rows")

    return integrated, energy, heating


def load_thermal_model_params():
    """Load thermal model parameters from CSV."""
    params_file = OUTPUT_DIR / 'thermal_model_results.csv'
    if params_file.exists():
        df = pd.read_csv(params_file)
        # Use davis_inside parameters
        row = df[df['room'] == 'davis_inside'].iloc[0]
        return {
            'c0': row['offset'],
            'tau_out': row['tau_outdoor_h'],
            'tau_eff': row['tau_effort_h'],
            'tau_pv': row['tau_pv_h'],
            'g_out': row['gain_outdoor'],
            'g_eff': row['gain_effort'],
            'g_pv': row['gain_pv'],
        }
    return None


def load_heating_curve_params():
    """Load heating curve parameters from Phase 3 output (same as weekly decomposition)."""
    # Use the same source as 05_weekly_decomposition.py
    hc_path = OUTPUT_DIR / 'heating_curve.csv'
    if hc_path.exists():
        import pandas as pd
        hc = pd.read_csv(hc_path).iloc[0]
        return {
            'intercept': hc['baseline'],
            'slope': hc['slope'],
        }
    # Fallback to Phase 2 JSON
    params_file = OUTPUT_DIR.parent / 'phase2' / 'heating_curve_params.json'
    if params_file.exists():
        with open(params_file) as f:
            return json.load(f)
    return {'intercept': 39.9, 'slope': -0.869}


def apply_lpf(x, tau_hours, dt_hours=0.25):
    """Apply first-order low-pass filter."""
    tau_samples = tau_hours / dt_hours
    alpha = 1 - np.exp(-1 / tau_samples) if tau_samples > 0 else 1

    y = np.zeros_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        if np.isnan(x[i]):
            y[i] = y[i-1]
        else:
            y[i] = alpha * x[i] + (1 - alpha) * y[i-1]
    return y


def compute_model_terms(df, params, hc_params):
    """Compute thermal model terms for decomposition."""
    # Get outdoor temperature
    t_out = df['stiebel_eltron_isg_outdoor_temperature'].values

    # Get HK2 target temperature (flow temperature setpoint)
    t_hk2 = df['stiebel_eltron_isg_target_temperature_hk_2'].values

    # Compute heating effort (deviation from heating curve)
    t_hk2_expected = hc_params.get('intercept', 39.9) + hc_params.get('slope', -0.869) * t_out
    effort = t_hk2 - t_hk2_expected

    # Get PV generation (column name is pv_generation_kwh)
    pv = df['pv_generation_kwh'].values if 'pv_generation_kwh' in df.columns else np.zeros(len(df))

    # Apply low-pass filters
    lpf_out = apply_lpf(t_out, params['tau_out'])
    lpf_eff = apply_lpf(effort, params['tau_eff'])
    lpf_pv = apply_lpf(pv, params['tau_pv'])

    # Compute contributions
    contrib_out = params['g_out'] * lpf_out
    contrib_eff = params['g_eff'] * lpf_eff
    contrib_pv = params['g_pv'] * lpf_pv

    # Predicted temperature
    t_pred = params['c0'] + contrib_out + contrib_eff + contrib_pv

    return {
        't_out': t_out,
        'effort': effort,
        'pv': pv,
        'lpf_out': lpf_out,
        'lpf_eff': lpf_eff,
        'lpf_pv': lpf_pv,
        'contrib_out': contrib_out,
        'contrib_eff': contrib_eff,
        'contrib_pv': contrib_pv,
        't_pred': t_pred,
    }


# ============================================================================
# INTRA-DAY MODEL FUNCTIONS
# ============================================================================

def predict_cop_intraday(t_outdoor, t_hk2):
    """Predict COP at 15-minute resolution using heat pump model.

    COP = intercept + coef_outdoor × T_outdoor + coef_hk2 × T_HK2
    """
    cop = (COP_MODEL['intercept'] +
           COP_MODEL['coef_t_outdoor'] * t_outdoor +
           COP_MODEL['coef_t_hk2'] * t_hk2)
    # Clip to reasonable range
    return np.clip(cop, 1.5, 8.0)


def predict_t_hk2(t_outdoor, setpoint=None, curve_rise=None, is_comfort=None):
    """Predict flow temperature (T_HK2) from heating curve.

    T_HK2 = setpoint + curve_rise × (T_ref - T_outdoor)
    """
    if setpoint is None:
        setpoint = HEATING_CURVE['default_setpoint']
    if curve_rise is None:
        curve_rise = HEATING_CURVE['default_curve_rise']

    # Use comfort or eco reference temperature
    if is_comfort is None:
        t_ref = HEATING_CURVE['t_ref_comfort']
    else:
        t_ref = np.where(is_comfort,
                         HEATING_CURVE['t_ref_comfort'],
                         HEATING_CURVE['t_ref_eco'])

    t_hk2 = setpoint + curve_rise * (t_ref - t_outdoor)
    return np.clip(t_hk2, 20, 55)  # Reasonable flow temp range


def simulate_battery_soc(pv, consumption, dt_hours=0.25):
    """Simulate battery state of charge with capacity constraints.

    Strategy: charge from excess PV, discharge to cover deficit.

    Args:
        pv: PV generation (kWh per interval)
        consumption: Total consumption (kWh per interval)
        dt_hours: Time step in hours (0.25 for 15-min)

    Returns:
        dict with soc, charge, discharge, grid_import, grid_export arrays
    """
    n = len(pv)
    cap = BATTERY_PARAMS['capacity_kwh']
    max_charge = BATTERY_PARAMS['max_charge_kw'] * dt_hours  # kWh per interval
    max_discharge = BATTERY_PARAMS['max_discharge_kw'] * dt_hours
    eff = np.sqrt(BATTERY_PARAMS['efficiency'])  # One-way efficiency

    # Initialize arrays
    soc = np.zeros(n)
    charge = np.zeros(n)
    discharge = np.zeros(n)
    grid_import = np.zeros(n)
    grid_export = np.zeros(n)

    # Initial SoC
    soc[0] = cap * BATTERY_PARAMS['initial_soc_pct'] / 100

    for i in range(n):
        net = pv[i] - consumption[i]  # Positive = excess

        if net > 0:
            # Excess PV - try to charge battery, export rest
            available_capacity = cap - soc[max(0, i-1) if i > 0 else 0]
            charge_possible = min(net * eff, max_charge, available_capacity)
            charge[i] = charge_possible
            grid_export[i] = net - charge_possible / eff  # Excess goes to grid
        else:
            # Deficit - try to discharge battery, import rest
            deficit = -net
            available_energy = soc[max(0, i-1) if i > 0 else 0]
            discharge_possible = min(deficit, max_discharge, available_energy)
            discharge[i] = discharge_possible
            grid_import[i] = deficit - discharge_possible

        # Update SoC
        if i > 0:
            soc[i] = soc[i-1] + charge[i] - discharge[i]
        else:
            soc[i] = cap * BATTERY_PARAMS['initial_soc_pct'] / 100 + charge[i] - discharge[i]

        soc[i] = np.clip(soc[i], 0, cap)

    return {
        'soc': soc,
        'soc_pct': soc / cap * 100,
        'charge': charge,
        'discharge': discharge,
        'grid_import': grid_import,
        'grid_export': grid_export,
    }


def is_high_tariff(timestamps):
    """Determine if each timestamp is in high tariff period.

    High tariff: Mon-Fri 06:00-21:00, Sat 06:00-12:00
    Low tariff: All other times
    """
    hour = timestamps.hour
    dayofweek = timestamps.dayofweek  # Monday=0, Sunday=6

    # Weekday (Mon-Fri): 06:00-21:00
    weekday_high = (dayofweek < 5) & (hour >= 6) & (hour < 21)
    # Saturday: 06:00-12:00
    saturday_high = (dayofweek == 5) & (hour >= 6) & (hour < 12)

    return weekday_high | saturday_high


def calculate_costs(grid_import, grid_export, timestamps, dt_hours=0.25):
    """Calculate costs at 15-minute resolution with tariff awareness.

    Returns costs in CHF.
    """
    high_tariff = is_high_tariff(timestamps)

    # Rates in CHF/kWh (convert from Rp)
    import_rate = np.where(high_tariff,
                           TARIFF_PARAMS['high_rate_rp'] / 100,
                           TARIFF_PARAMS['low_rate_rp'] / 100)
    export_rate = TARIFF_PARAMS['feedin_rate_rp'] / 100

    import_cost = grid_import * import_rate
    export_revenue = grid_export * export_rate
    net_cost = import_cost - export_revenue

    return {
        'import_cost': import_cost,
        'export_revenue': export_revenue,
        'net_cost': net_cost,
        'high_tariff': high_tariff,
    }


def predict_intraday_energy(energy_week, t_outdoor, consumption_model):
    """Full intra-day energy prediction with battery and costs.

    Args:
        energy_week: DataFrame with energy data
        t_outdoor: Outdoor temperature array (cleaned)
        consumption_model: Dict with base_load and heating_coef

    Returns:
        Dict with all predicted values at 15-minute resolution
    """
    # Use energy_week length as reference (may differ from t_outdoor for partial weeks)
    n = len(energy_week)
    dt = 0.25  # 15-minute intervals

    # Align t_outdoor to energy_week length
    if len(t_outdoor) != n:
        # Interpolate or truncate t_outdoor to match energy_week
        t_outdoor = t_outdoor[:n] if len(t_outdoor) > n else np.pad(t_outdoor, (0, n - len(t_outdoor)), mode='edge')

    # 1. Predict consumption from temperature (HDD model)
    if consumption_model:
        hdd = np.maximum(0, consumption_model['t_ref'] - t_outdoor)
        consumption_pred = consumption_model['base_load'] + consumption_model['heating_coef'] * hdd
        consumption_pred = np.maximum(0, consumption_pred)
    else:
        consumption_pred = np.full(n, 0.5)  # Default 0.5 kW

    # 2. Get observed PV (we don't model PV, use actual)
    if 'pv_generation_kwh' in energy_week.columns:
        pv = energy_week['pv_generation_kwh'].fillna(0).values
    else:
        pv = np.zeros(n)

    # 3. Predict T_HK2 from heating curve (assume comfort mode during day)
    hour = energy_week.index.hour if hasattr(energy_week.index, 'hour') else np.zeros(n)
    is_comfort = (hour >= 6) & (hour < 20)  # Simplified schedule
    t_hk2_pred = predict_t_hk2(t_outdoor, is_comfort=is_comfort)

    # 4. Predict COP from T_outdoor and T_HK2
    cop_pred = predict_cop_intraday(t_outdoor, t_hk2_pred)

    # 5. Simulate battery with constraints
    battery_sim = simulate_battery_soc(pv, consumption_pred, dt)

    # 6. Calculate costs with tariff awareness
    costs = calculate_costs(
        battery_sim['grid_import'],
        battery_sim['grid_export'],
        energy_week.index,
        dt
    )

    return {
        'consumption': consumption_pred,
        'pv': pv,
        't_hk2': t_hk2_pred,
        'cop': cop_pred,
        'is_comfort': is_comfort,
        **battery_sim,
        **costs,
    }


def fit_energy_models(energy_df, integrated_df):
    """Fit simple energy models for prediction.

    Returns dict with model coefficients for:
    - consumption: base_load + temp_coef * (T_ref - T_outdoor)
    - Battery/grid flows from energy balance
    """
    from sklearn.linear_model import LinearRegression

    # Merge energy and temperature data
    t_out_col = 'stiebel_eltron_isg_outdoor_temperature'
    if t_out_col not in integrated_df.columns:
        return None

    # Align data
    common_idx = energy_df.index.intersection(integrated_df.index)
    if len(common_idx) < 100:
        return None

    consumption = energy_df.loc[common_idx, 'total_consumption_kwh'].fillna(0)
    t_out = integrated_df.loc[common_idx, t_out_col].ffill()
    pv = energy_df.loc[common_idx, 'pv_generation_kwh'].fillna(0)

    # Fit consumption model: consumption = base + heating_coef * max(0, T_ref - T_out)
    # Use HDD approach with reference temp 18°C
    T_REF = 18.0
    hdd = np.maximum(0, T_REF - t_out.values)

    valid = ~(np.isnan(consumption.values) | np.isnan(hdd))
    if valid.sum() < 50:
        return None

    X = hdd[valid].reshape(-1, 1)
    y = consumption.values[valid]

    model = LinearRegression()
    model.fit(X, y)

    return {
        'base_load': model.intercept_,
        'heating_coef': model.coef_[0],
        't_ref': T_REF,
        'r2': model.score(X, y),
    }


def predict_energy(energy_week, t_out, energy_model):
    """Predict energy flows using fitted model."""
    if energy_model is None:
        return {}

    # Handle NaN in t_out by forward-filling
    t_out_clean = pd.Series(t_out).ffill().bfill().values

    # Predict consumption from temperature
    hdd = np.maximum(0, energy_model['t_ref'] - t_out_clean)
    consumption_pred = energy_model['base_load'] + energy_model['heating_coef'] * hdd
    consumption_pred = np.maximum(0, consumption_pred)

    # Get PV (observed - no model for this)
    pv = energy_week['pv_generation_kwh'].fillna(0).values if 'pv_generation_kwh' in energy_week.columns else np.zeros(len(t_out))

    # Energy balance predictions
    # Net = PV - Consumption
    net = pv - consumption_pred

    # Grid feed-in = excess (positive net)
    feedin_pred = np.maximum(0, net)

    # Grid import = deficit (negative net)
    import_pred = np.maximum(0, -net)

    # Battery absorbs some of the excess/provides some of deficit
    # Simple model: battery smooths ~50% of imbalance
    battery_charge_pred = 0.5 * feedin_pred
    battery_discharge_pred = 0.5 * import_pred

    return {
        'consumption': consumption_pred,
        'feedin': feedin_pred - battery_charge_pred,  # Reduced by battery charging
        'import': import_pred - battery_discharge_pred,  # Reduced by battery discharge
        'battery_charge': battery_charge_pred,
        'battery_discharge': battery_discharge_pred,
    }


def create_extended_decomposition(df, energy_df, heating_df, params, hc_params,
                                   start_date, end_date, output_path, title_suffix='',
                                   energy_model=None):
    """Create extended decomposition figure with all panels in single column layout."""

    # Filter data to date range
    mask = (df.index >= start_date) & (df.index < end_date)
    df_week = df[mask].copy()

    if len(df_week) < 10:
        print(f"  Insufficient data for {start_date}")
        return None

    # Get energy data for the same period
    energy_mask = (energy_df.index >= start_date) & (energy_df.index < end_date)
    energy_week = energy_df[energy_mask].copy()

    # Get heating data for the same period
    heating_mask = (heating_df.index >= start_date) & (heating_df.index < end_date)
    heating_week = heating_df[heating_mask].copy()

    # Filter df to overlap period with valid heating data before LPF computation
    # This ensures LPF doesn't start with NaN values
    outdoor_col = 'stiebel_eltron_isg_outdoor_temperature'
    first_valid_idx = df[outdoor_col].first_valid_index()
    if first_valid_idx is not None:
        df_valid = df.loc[first_valid_idx:].copy()
    else:
        df_valid = df

    # Compute model terms on valid data for proper LPF initialization
    terms_full = compute_model_terms(df_valid, params, hc_params)

    # Extract week portion from full terms using boolean mask
    full_mask = (df_valid.index >= start_date) & (df_valid.index < end_date)
    terms = {
        't_out': terms_full['t_out'][full_mask],
        'effort': terms_full['effort'][full_mask],
        'pv': terms_full['pv'][full_mask],
        'lpf_out': terms_full['lpf_out'][full_mask],
        'lpf_eff': terms_full['lpf_eff'][full_mask],
        'lpf_pv': terms_full['lpf_pv'][full_mask],
        'contrib_out': terms_full['contrib_out'][full_mask],
        'contrib_eff': terms_full['contrib_eff'][full_mask],
        'contrib_pv': terms_full['contrib_pv'][full_mask],
        't_pred': terms_full['t_pred'][full_mask],
    }

    # Get actual room temperature
    t_actual = df_week['davis_inside_temperature'].values

    # Create figure with 10 panels in single column
    fig, axes = plt.subplots(10, 1, figsize=(14, 28))
    fig.suptitle(f'Extended Model Decomposition{title_suffix}', fontsize=14, fontweight='bold', y=0.995)

    time_idx = df_week.index

    # === Panel 1: Room Temperature (actual vs predicted) ===
    ax = axes[0]
    ax.plot(time_idx, t_actual, color=COLORS['actual'], label='Observed', linewidth=1.5)
    ax.plot(time_idx, terms['t_pred'], color=COLORS['predicted'], label='Predicted',
            linewidth=1.5, linestyle='--')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('1. Room Temperature: Observed vs Predicted')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Calculate R² for this period
    r2, rmse = None, None
    t_pred_arr = np.array(terms['t_pred'])
    valid = ~(np.isnan(t_actual) | np.isnan(t_pred_arr))
    if valid.sum() > 10:
        ss_res = np.sum((t_actual[valid] - t_pred_arr[valid])**2)
        ss_tot = np.sum((t_actual[valid] - np.mean(t_actual[valid]))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        rmse = np.sqrt(np.mean((t_actual[valid] - t_pred_arr[valid])**2))
        ax.text(0.02, 0.95, f'R²={r2:.3f}, RMSE={rmse:.2f}°C',
                transform=ax.transAxes, va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # === Panel 2: Outdoor Temperature Contribution ===
    ax = axes[1]
    ax2 = ax.twinx()
    ax.plot(time_idx, terms['t_out'], color=COLORS['outdoor'], alpha=0.6,
            linewidth=1, label='T_outdoor (raw)')
    ax2.plot(time_idx, terms['contrib_out'], color=COLORS['outdoor'],
             linewidth=1.5, label=f'Contribution (g={params["g_out"]:.3f})')
    ax.set_ylabel('Outdoor Temp (°C)', color=COLORS['outdoor'])
    ax2.set_ylabel('Contribution to T_room (°C)', color=COLORS['outdoor'])
    ax.set_title(f'2. Outdoor Temperature → Room Temp (τ={params["tau_out"]:.0f}h, g={params["g_out"]:.3f})')
    ax.grid(True, alpha=0.3)

    # === Panel 3: Heating Effort Contribution ===
    ax = axes[2]
    ax2 = ax.twinx()
    ax.plot(time_idx, terms['effort'], color=COLORS['heating'], alpha=0.4,
            linewidth=0.8, label='Effort (raw)')
    ax2.plot(time_idx, terms['contrib_eff'], color=COLORS['heating'],
             linewidth=1.5, label=f'Contribution (g={params["g_eff"]:.3f})')
    ax.set_ylabel('Heating Effort (°C)', color=COLORS['heating'])
    ax2.set_ylabel('Contribution to T_room (°C)', color=COLORS['heating'])
    ax.set_title(f'3. Heating Effort → Room Temp (τ={params["tau_eff"]:.0f}h, g={params["g_eff"]:.3f})')
    ax.grid(True, alpha=0.3)

    # === Panel 4: Solar/PV Contribution ===
    ax = axes[3]
    ax2 = ax.twinx()
    ax.fill_between(time_idx, 0, terms['pv'], color=COLORS['solar'], alpha=0.3, label='PV (raw)')
    ax2.plot(time_idx, terms['contrib_pv'], color=COLORS['solar'],
             linewidth=1.5, label=f'Contribution (g={params["g_pv"]:.3f})')
    ax.set_ylabel('PV Generation (kW)', color=COLORS['solar'])
    ax2.set_ylabel('Contribution to T_room (°C)', color=COLORS['solar'])
    ax.set_title(f'4. Solar Gain → Room Temp (τ={params["tau_pv"]:.0f}h, g={params["g_pv"]:.3f})')
    ax.grid(True, alpha=0.3)

    # Get intra-day model predictions (with battery constraints and tariff awareness)
    t_out_clean = pd.Series(terms['t_out']).ffill().bfill().values
    intraday_pred = predict_intraday_energy(energy_week, t_out_clean, energy_model)

    # === Panel 5: Battery State of Charge ===
    ax = axes[4]
    if len(energy_week) > 0 and 'battery_charging_kwh' in energy_week.columns:
        # Calculate observed SoC from cumulative charge/discharge
        charge_obs = energy_week['battery_charging_kwh'].fillna(0).values
        discharge_obs = energy_week['battery_discharging_kwh'].fillna(0).values
        net_obs = charge_obs - discharge_obs
        cumsum_obs = np.cumsum(net_obs)
        # Normalize: assume starts at 50%, scale by capacity
        cap = BATTERY_PARAMS['capacity_kwh']
        soc_obs = 50 + (cumsum_obs - cumsum_obs[0]) / cap * 100
        soc_obs = np.clip(soc_obs, 0, 100)

        # Plot observed fill first
        ax.fill_between(energy_week.index, 0, soc_obs, color=COLORS['battery'],
                        alpha=0.3, label='Observed')

        # Model prediction with capacity constraints (plotted on top)
        if 'soc_pct' in intraday_pred:
            ax.plot(energy_week.index, intraday_pred['soc_pct'], color=COLORS['baseline'],
                    linewidth=2, linestyle='--', label=f'Model ({cap:.0f}kWh)', zorder=5)

        ax.set_ylabel('SoC (%)')
        ax.set_ylim(0, 100)
        ax.legend(loc='upper right')
    else:
        ax.text(0.5, 0.5, 'No battery data', ha='center', va='center', transform=ax.transAxes)
    ax.set_title(f'5. Battery SoC: Observed vs Model (Cap={BATTERY_PARAMS["capacity_kwh"]:.0f}kWh, η={BATTERY_PARAMS["efficiency"]:.0%})')
    ax.grid(True, alpha=0.3)

    # === Panel 6: Power Consumption ===
    ax = axes[5]
    if len(energy_week) > 0 and 'total_consumption_kwh' in energy_week.columns:
        consumption_obs = energy_week['total_consumption_kwh'].fillna(0).values

        # Plot observed fill first
        ax.fill_between(energy_week.index, 0, consumption_obs, color=COLORS['consumption'],
                        alpha=0.3, label='Observed')

        # HDD model prediction (on top)
        if 'consumption' in intraday_pred:
            ax.plot(energy_week.index, intraday_pred['consumption'], color=COLORS['baseline'],
                    linewidth=2, linestyle='--', label='Model (HDD)', zorder=5)

        ax.set_ylabel('Power (kWh/15min)')
        ax.legend(loc='upper right')
    else:
        ax.text(0.5, 0.5, 'No consumption data', ha='center', va='center', transform=ax.transAxes)
    ax.set_title('6. Power Consumption: Observed vs Model')
    ax.grid(True, alpha=0.3)

    # === Panel 7: Grid Feed-in ===
    ax = axes[6]
    if len(energy_week) > 0 and 'grid_feedin_kwh' in energy_week.columns:
        feedin_obs = energy_week['grid_feedin_kwh'].fillna(0).values

        # Plot observed fill first
        ax.fill_between(energy_week.index, 0, feedin_obs, color=COLORS['grid_export'],
                        alpha=0.3, label='Observed')

        # Model prediction with battery (on top)
        if 'grid_export' in intraday_pred:
            ax.plot(energy_week.index, intraday_pred['grid_export'], color=COLORS['baseline'],
                    linewidth=2, linestyle='--', label='Model', zorder=5)

        ax.set_ylabel('Power (kWh/15min)')
        ax.legend(loc='upper right')
    else:
        ax.text(0.5, 0.5, 'No feed-in data', ha='center', va='center', transform=ax.transAxes)
    ax.set_title('7. Grid Feed-in: Observed vs Model')
    ax.grid(True, alpha=0.3)

    # === Panel 8: Grid Import ===
    ax = axes[7]
    if len(energy_week) > 0 and 'external_supply_kwh' in energy_week.columns:
        import_obs = energy_week['external_supply_kwh'].fillna(0).values

        # Plot observed fill first
        ax.fill_between(energy_week.index, 0, import_obs, color=COLORS['grid_import'],
                        alpha=0.3, label='Observed')

        # Model prediction with battery (on top)
        if 'grid_import' in intraday_pred:
            ax.plot(energy_week.index, intraday_pred['grid_import'], color=COLORS['baseline'],
                    linewidth=2, linestyle='--', label='Model', zorder=5)

        # Show tariff periods as background
        if 'high_tariff' in intraday_pred:
            high = intraday_pred['high_tariff']
            ax.fill_between(energy_week.index, 0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1,
                           where=high, alpha=0.1, color='red', label='High tariff')

        ax.set_ylabel('Power (kWh/15min)')
        ax.legend(loc='upper right')
    else:
        ax.text(0.5, 0.5, 'No grid import data', ha='center', va='center', transform=ax.transAxes)
    ax.set_title('8. Grid Import: Observed vs Model (tariff-aware)')
    ax.grid(True, alpha=0.3)

    # === Panel 9: Outdoor Temperature ===
    ax = axes[8]
    ax.plot(time_idx, terms['t_out'], color=COLORS['outdoor'], linewidth=1.2, label='Observed')
    mean_temp = np.nanmean(terms['t_out'])
    ax.axhline(y=mean_temp, color=COLORS['baseline'], linestyle='--',
               label=f'Mean: {mean_temp:.1f}°C')
    ax.fill_between(time_idx, mean_temp, terms['t_out'],
                    where=terms['t_out'] > mean_temp,
                    color=COLORS['heating'], alpha=0.2, label='Above mean')
    ax.fill_between(time_idx, mean_temp, terms['t_out'],
                    where=terms['t_out'] < mean_temp,
                    color=COLORS['outdoor'], alpha=0.2, label='Below mean')
    ax.set_ylabel('Temperature (°C)')
    ax.legend(loc='upper right')
    ax.set_title('9. Outdoor Temperature')
    ax.grid(True, alpha=0.3)

    # === Panel 10: Heat Pump COP (Intra-day Model) ===
    ax = axes[9]

    # Plot intra-day COP model prediction
    if 'cop' in intraday_pred:
        cop_pred = intraday_pred['cop']
        is_comfort = intraday_pred.get('is_comfort', np.ones(len(cop_pred), dtype=bool))

        # Fill for comfort/eco periods
        ax.fill_between(energy_week.index, 1.5, cop_pred,
                        where=is_comfort, alpha=0.3, color=COLORS['heating'],
                        label='Comfort mode')
        ax.fill_between(energy_week.index, 1.5, cop_pred,
                        where=~is_comfort, alpha=0.3, color=COLORS['outdoor'],
                        label='Eco mode')

        # Model line
        ax.plot(energy_week.index, cop_pred, color=COLORS['cop'],
                linewidth=1.5, label='Model COP')

        # Add daily observed COP as markers for comparison
        if len(heating_week) > 0:
            q_col = 'stiebel_eltron_isg_produced_heating_today'
            e_col = 'stiebel_eltron_isg_consumed_heating_today'
            q_heat = heating_week.get(q_col, pd.Series())
            e_elec = heating_week.get(e_col, pd.Series())

            if len(q_heat) > 0 and len(e_elec) > 0:
                q_daily = q_heat.dropna().groupby(q_heat.dropna().index.date).max()
                e_daily = e_elec.dropna().groupby(e_elec.dropna().index.date).max()
                common_dates = q_daily.index.intersection(e_daily.index)

                if len(common_dates) > 0:
                    cop_obs = (q_daily.loc[common_dates] / e_daily.loc[common_dates]).clip(1, 8)
                    dates = pd.to_datetime(common_dates) + pd.Timedelta(hours=12)  # Plot at midday
                    ax.scatter(dates, cop_obs.values, s=80, color=COLORS['actual'],
                              marker='o', zorder=10, label='Observed (daily)', edgecolors='white')

        mean_cop = cop_pred.mean()
        ax.axhline(y=mean_cop, color=COLORS['baseline'], linestyle='--',
                  linewidth=2, label=f'Model mean: {mean_cop:.2f}')
        ax.set_ylabel('COP')
        ax.set_ylim(1.5, 6)
        ax.legend(loc='upper right', fontsize=7, ncol=2)
    else:
        ax.text(0.5, 0.5, 'No COP model prediction', ha='center', va='center',
                transform=ax.transAxes, fontsize=11)

    ax.set_title(f'10. Heat Pump COP: Model (COP = {COP_MODEL["intercept"]:.1f} + {COP_MODEL["coef_t_outdoor"]:.2f}×T_out - {abs(COP_MODEL["coef_t_hk2"]):.2f}×T_HK2)')
    ax.grid(True, alpha=0.3)

    # Format x-axes for all panels
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.tick_params(axis='x', rotation=0)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return {'success': True, 'r2': r2, 'rmse': rmse}


def main():
    """Generate extended decomposition figures."""
    print("=" * 60)
    print("Phase 3: Extended Model Decomposition")
    print("=" * 60)

    # Load data
    integrated, energy, heating = load_data()

    # Load model parameters
    params = load_thermal_model_params()
    if params is None:
        print("ERROR: Could not load thermal model parameters")
        print("Run 01_thermal_model.py first")
        return

    print(f"\nModel parameters loaded:")
    print(f"  τ_out={params['tau_out']}h, τ_eff={params['tau_eff']}h, τ_pv={params['tau_pv']}h")
    print(f"  g_out={params['g_out']:.3f}, g_eff={params['g_eff']:.3f}, g_pv={params['g_pv']:.3f}")

    # Load heating curve parameters
    hc_params = load_heating_curve_params()

    # Find overlapping data period
    overlap_start = max(integrated.index.min(), energy.index.min(), heating.index.min())
    overlap_end = min(integrated.index.max(), energy.index.max(), heating.index.max())
    print(f"\nData overlap: {overlap_start.date()} to {overlap_end.date()}")

    # Fit energy models for prediction
    print("\nFitting energy models...")
    energy_model = fit_energy_models(energy, integrated)
    if energy_model:
        print(f"  Consumption model: base={energy_model['base_load']:.3f} kW, "
              f"heating_coef={energy_model['heating_coef']:.4f} kW/°C, R²={energy_model['r2']:.3f}")
    else:
        print("  WARNING: Could not fit energy model")

    # Generate main extended decomposition figure (representative week)
    print("\nGenerating main extended decomposition figure...")

    # Find a representative week with good data
    # Use the most recent complete week
    end_date = overlap_end
    start_date = end_date - pd.Timedelta(days=7)

    output_path = OUTPUT_DIR / 'fig3.05_extended_decomposition.png'
    success = create_extended_decomposition(
        integrated, energy, heating, params, hc_params,
        start_date, end_date, output_path,
        title_suffix=f'\n{start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}',
        energy_model=energy_model
    )

    if success:
        print(f"  Saved: {output_path.name}")

    # Generate weekly decomposition figures
    print("\nGenerating weekly extended decomposition figures...")

    # Use ANALYSIS_START_DATE as the first week start, then 7-day intervals
    # This aligns with 05_weekly_decomposition.py
    valid_data = integrated[['davis_inside_temperature', 'stiebel_eltron_isg_outdoor_temperature']].dropna()

    # Handle timezone: match ANALYSIS_START_DATE to data's timezone
    start_date = ANALYSIS_START_DATE
    if valid_data.index.tz is not None and start_date.tz is None:
        start_date = start_date.tz_localize(valid_data.index.tz)
    elif valid_data.index.tz is None and start_date.tz is not None:
        start_date = start_date.tz_localize(None)

    valid_data = valid_data[valid_data.index >= start_date]
    data_end = min(overlap_end, valid_data.index.max())

    # Generate week boundaries
    weeks = []
    current = start_date
    while current < data_end:
        week_end = min(current + pd.Timedelta(days=7), data_end + pd.Timedelta(hours=1))
        # Count points in this week
        week_data = valid_data[(valid_data.index >= current) & (valid_data.index < week_end)]
        # Require at least 2 days of data
        if len(week_data) >= 2 * 96:
            weeks.append((current, week_end))
        current = current + pd.Timedelta(days=7)

    print(f"  Found {len(weeks)} weeks (starting {start_date.date()})")

    for i, (start, end) in enumerate(weeks):
        week_num = i + 1
        output_path = WEEKLY_DIR / f'week_{week_num:02d}_extended.png'

        result = create_extended_decomposition(
            integrated, energy, heating, params, hc_params,
            start, end, output_path,
            title_suffix=f' - Week {week_num}\n{start.strftime("%Y-%m-%d")} to {end.strftime("%Y-%m-%d")}',
            energy_model=energy_model
        )

        if result and result.get('success'):
            r2_str = f"R²={result['r2']:.3f}" if result.get('r2') is not None else "R²=N/A"
            rmse_str = f"RMSE={result['rmse']:.2f}°C" if result.get('rmse') is not None else ""
            print(f"  Week {week_num}: {start.strftime('%Y-%m-%d')} - {r2_str}, {rmse_str}")
        else:
            print(f"  Week {week_num}: {start.strftime('%Y-%m-%d')} - Skipped (insufficient data)")

    print("\n" + "=" * 60)
    print("EXTENDED DECOMPOSITION COMPLETE")
    print("=" * 60)
    print(f"\nOutputs:")
    print(f"  - {OUTPUT_DIR / 'fig3.05_extended_decomposition.png'}")
    print(f"  - {WEEKLY_DIR / 'week_XX_extended.png'} (multiple)")


if __name__ == '__main__':
    main()
