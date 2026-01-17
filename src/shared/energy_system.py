"""
Shared Energy System Simulation Module

Provides intra-day energy system simulation with:
- Battery model with capacity constraints (SoC tracking)
- Intra-day COP model (varies with T_outdoor and T_HK2)
- Heating curve model (T_HK2 from controllable parameters)
- Energy balance simulation (PV → consumption → battery → grid)
- Tariff-aware cost calculation (high/low rates, feed-in)

Used by:
- Phase 3: Extended decomposition (panels 5-8, 10)
- Phase 4: Strategy simulation and Pareto optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from pathlib import Path
import json


# ============================================================================
# Battery Parameters (estimated from historical data)
# ============================================================================
BATTERY_PARAMS = {
    'capacity_kwh': 11.0,        # Total capacity
    'max_charge_kw': 5.0,        # Max charging rate
    'max_discharge_kw': 5.0,     # Max discharging rate
    'efficiency': 0.77,          # Round-trip efficiency (post-degradation, was 0.84)
    'initial_soc_pct': 50.0,     # Default starting SoC
    'min_soc_pct': 20.0,         # Minimum SoC (battery protection since Mar 2025)
    'max_soc_pct': 100.0,        # Maximum SoC (could be 90% for longevity)
    # Time-of-use discharge strategy (observed: battery concentrates discharge 15:00-22:00)
    'discharge_start_hour': 15.0,   # Preferred discharge window start
    'discharge_end_hour': 22.0,     # Preferred discharge window end
    'allow_overnight_discharge': False,  # Don't discharge 00:00-06:00
}


# ============================================================================
# COP Model (from Phase 3 heat pump analysis)
# ============================================================================
COP_PARAMS = {
    'intercept': 5.93,           # Base COP
    'coef_t_outdoor': 0.13,      # COP increase per °C outdoor
    'coef_t_hk2': -0.08,         # COP decrease per °C flow temp
    'min_cop': 1.5,              # Minimum COP (safety floor)
    'max_cop': 8.0,              # Maximum COP (physical limit)
}


# ============================================================================
# Heating Curve Model (from Phase 2 analysis)
# ============================================================================
HEATING_CURVE_PARAMS = {
    't_ref_comfort': 21.32,      # Reference temp for comfort mode
    't_ref_eco': 19.18,          # Reference temp for eco mode
    'default_setpoint': 20.0,    # Default setpoint
    'default_curve_rise': 1.08,  # Default curve slope
    't_hk2_min': 20.0,           # Minimum flow temp
    't_hk2_max': 55.0,           # Maximum flow temp
}


# ============================================================================
# Tariff Parameters (Primeo Energie)
# ============================================================================
TARIFF_PARAMS = {
    'high_rate_rp': 32.6,        # High tariff (Rp/kWh)
    'low_rate_rp': 26.0,         # Low tariff (Rp/kWh)
    'feedin_rate_rp': 13.0,      # Feed-in rate with HKN (Rp/kWh)
}


# ============================================================================
# Consumption Model (HDD-based)
# ============================================================================
CONSUMPTION_PARAMS = {
    'base_load_kw': 0.024,       # Base load (non-heating)
    'heating_coef_kw': 0.037,    # Heating coefficient per HDD
    'hdd_base_temp': 18.0,       # HDD reference temperature
}


def load_heating_curve_params(phase2_dir: Path = None) -> Dict:
    """Load heating curve parameters from Phase 2 JSON."""
    if phase2_dir is None:
        phase2_dir = Path(__file__).parent.parent.parent / 'output' / 'phase2'

    params_file = phase2_dir / 'heating_curve_params.json'
    if params_file.exists():
        with open(params_file) as f:
            data = json.load(f)
        return {
            't_ref_comfort': data['t_ref_comfort'],
            't_ref_eco': data['t_ref_eco'],
        }
    return {
        't_ref_comfort': HEATING_CURVE_PARAMS['t_ref_comfort'],
        't_ref_eco': HEATING_CURVE_PARAMS['t_ref_eco'],
    }


# ============================================================================
# COP Model Functions
# ============================================================================

def predict_cop(t_outdoor: np.ndarray, t_hk2: np.ndarray,
                params: Dict = None) -> np.ndarray:
    """
    Calculate COP from outdoor temperature and flow temperature.

    COP = intercept + coef_t_outdoor × T_outdoor + coef_t_hk2 × T_HK2

    Args:
        t_outdoor: Outdoor temperature array (°C)
        t_hk2: Flow temperature array (°C)
        params: Optional COP parameters (default: COP_PARAMS)

    Returns:
        COP array, clipped to [min_cop, max_cop]
    """
    if params is None:
        params = COP_PARAMS

    cop = (params['intercept'] +
           params['coef_t_outdoor'] * np.asarray(t_outdoor) +
           params['coef_t_hk2'] * np.asarray(t_hk2))

    return np.clip(cop, params.get('min_cop', 1.5), params.get('max_cop', 8.0))


def predict_t_hk2(t_outdoor: np.ndarray, setpoint: float, curve_rise: float,
                  is_comfort: np.ndarray, params: Dict = None) -> np.ndarray:
    """
    Calculate target flow temperature from heating curve.

    T_HK2 = setpoint + curve_rise × (T_ref - T_outdoor)

    Args:
        t_outdoor: Outdoor temperature array (°C)
        setpoint: Setpoint temperature (comfort or eco)
        curve_rise: Heating curve slope (Steilheit)
        is_comfort: Boolean array (True = comfort mode, False = eco)
        params: Optional heating curve parameters

    Returns:
        T_HK2 array, clipped to [t_hk2_min, t_hk2_max]
    """
    if params is None:
        params = HEATING_CURVE_PARAMS

    t_outdoor = np.asarray(t_outdoor)
    is_comfort = np.asarray(is_comfort)

    # Reference temperature depends on mode
    t_ref = np.where(is_comfort, params['t_ref_comfort'], params['t_ref_eco'])

    # Calculate T_HK2 from heating curve
    t_hk2 = setpoint + curve_rise * (t_ref - t_outdoor)

    return np.clip(t_hk2, params.get('t_hk2_min', 20), params.get('t_hk2_max', 55))


def predict_t_hk2_variable_setpoint(t_outdoor: np.ndarray,
                                     setpoint_comfort: float, setpoint_eco: float,
                                     curve_rise: float, is_comfort: np.ndarray,
                                     params: Dict = None) -> np.ndarray:
    """
    Calculate target flow temperature with different comfort/eco setpoints.

    T_HK2 = setpoint + curve_rise × (T_ref - T_outdoor)

    Where setpoint and T_ref both depend on comfort/eco mode.

    Args:
        t_outdoor: Outdoor temperature array (°C)
        setpoint_comfort: Comfort mode setpoint (°C)
        setpoint_eco: Eco mode setpoint (°C)
        curve_rise: Heating curve slope
        is_comfort: Boolean array (True = comfort, False = eco)
        params: Optional heating curve parameters

    Returns:
        T_HK2 array
    """
    if params is None:
        params = HEATING_CURVE_PARAMS

    t_outdoor = np.asarray(t_outdoor)
    is_comfort = np.asarray(is_comfort)

    # Both setpoint and T_ref depend on mode
    setpoint = np.where(is_comfort, setpoint_comfort, setpoint_eco)
    t_ref = np.where(is_comfort, params['t_ref_comfort'], params['t_ref_eco'])

    t_hk2 = setpoint + curve_rise * (t_ref - t_outdoor)

    return np.clip(t_hk2, params.get('t_hk2_min', 20), params.get('t_hk2_max', 55))


# ============================================================================
# Battery Model Functions
# ============================================================================

def simulate_battery_soc(pv_generation: np.ndarray, consumption: np.ndarray,
                         dt_hours: float = 0.25,
                         battery_params: Dict = None,
                         initial_soc_pct: float = None,
                         timestamps: pd.DatetimeIndex = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate battery state-of-charge with capacity and time-of-use constraints.

    Energy flow logic:
    1. Net = PV - consumption
    2. If Net > 0: charge battery (up to max_soc), excess to grid
    3. If Net < 0: discharge battery (if above min_soc AND in discharge window), deficit from grid

    Improvements over simple model:
    - Min/max SoC limits (battery protection)
    - Time-of-use discharge strategy (concentrate discharge 15:00-22:00)
    - Optional overnight discharge blocking

    Args:
        pv_generation: PV generation array (kWh per interval)
        consumption: Consumption array (kWh per interval)
        dt_hours: Time step in hours (default 0.25 = 15 min)
        battery_params: Battery parameters (default: BATTERY_PARAMS)
        initial_soc_pct: Initial SoC percentage (default from params)
        timestamps: Optional DatetimeIndex for time-of-use logic

    Returns:
        Tuple of (soc_kwh, grid_import, grid_export, battery_flow)
        - soc_kwh: Battery SoC in kWh
        - grid_import: Import from grid (kWh per interval)
        - grid_export: Export to grid (kWh per interval)
        - battery_flow: Battery flow (+ = charging, - = discharging)
    """
    if battery_params is None:
        battery_params = BATTERY_PARAMS

    if initial_soc_pct is None:
        initial_soc_pct = battery_params.get('initial_soc_pct', 50.0)

    capacity = battery_params['capacity_kwh']
    max_charge = battery_params['max_charge_kw'] * dt_hours  # kWh per interval
    max_discharge = battery_params['max_discharge_kw'] * dt_hours
    efficiency = battery_params.get('efficiency', 0.77)
    one_way_eff = np.sqrt(efficiency)  # One-way efficiency

    # SoC limits
    min_soc_pct = battery_params.get('min_soc_pct', 20.0)
    max_soc_pct = battery_params.get('max_soc_pct', 100.0)
    min_soc = capacity * min_soc_pct / 100
    max_soc = capacity * max_soc_pct / 100

    # Time-of-use parameters
    discharge_start = battery_params.get('discharge_start_hour', 15.0)
    discharge_end = battery_params.get('discharge_end_hour', 22.0)
    allow_overnight = battery_params.get('allow_overnight_discharge', False)

    pv = np.asarray(pv_generation)
    cons = np.asarray(consumption)
    n = len(pv)

    # Output arrays
    soc = np.zeros(n)
    grid_import = np.zeros(n)
    grid_export = np.zeros(n)
    battery_flow = np.zeros(n)

    # Initial state (clamp to valid range)
    initial_soc = np.clip(capacity * initial_soc_pct / 100, min_soc, max_soc)
    current_soc = initial_soc

    # Determine discharge permission per interval
    if timestamps is not None:
        hours = timestamps.hour + timestamps.minute / 60
        # Primary discharge window: discharge_start to discharge_end
        in_discharge_window = (hours >= discharge_start) & (hours < discharge_end)
        # Overnight (00:00-06:00) - allow if flag set
        is_overnight = (hours >= 0) & (hours < 6)
        # Allow discharge during window OR if overnight is allowed
        can_discharge = in_discharge_window | (is_overnight & allow_overnight) | (~is_overnight & (hours >= 6) & (hours < discharge_start))
        # Actually, simpler: only block overnight if flag is False
        if not allow_overnight:
            can_discharge = ~is_overnight
        else:
            can_discharge = np.ones(n, dtype=bool)
    else:
        # No timestamps: allow discharge always (backward compatible)
        can_discharge = np.ones(n, dtype=bool)

    for i in range(n):
        net_energy = pv[i] - cons[i]

        if net_energy > 0:
            # Excess PV: charge battery (up to max_soc)
            charge_room = max_soc - current_soc
            charge_possible = min(net_energy * one_way_eff, max_charge, charge_room)

            battery_flow[i] = charge_possible  # Positive = charging
            grid_export[i] = max(0, net_energy - charge_possible / one_way_eff)
            grid_import[i] = 0

            current_soc = current_soc + charge_possible
        else:
            # Deficit: discharge battery (if above min_soc AND allowed)
            deficit = -net_energy
            usable_energy = (current_soc - min_soc) * one_way_eff  # Energy available above min_soc

            if can_discharge[i] and usable_energy > 0:
                discharge_output = min(deficit, max_discharge * one_way_eff, usable_energy)
                soc_change = discharge_output / one_way_eff

                battery_flow[i] = -soc_change  # Negative = discharging
                grid_import[i] = max(0, deficit - discharge_output)
                current_soc = current_soc - soc_change
            else:
                # Cannot discharge: import all from grid
                battery_flow[i] = 0
                grid_import[i] = deficit

            grid_export[i] = 0

        # Store SoC (clamp to valid range)
        soc[i] = np.clip(current_soc, min_soc, max_soc)
        current_soc = soc[i]

    return soc, grid_import, grid_export, battery_flow


# ============================================================================
# Tariff Functions
# ============================================================================

def is_high_tariff(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """
    Determine high/low tariff periods.

    High tariff: Mon-Fri 06:00-21:00, Sat 06:00-12:00
    Low tariff: All other times + Swiss federal holidays

    Args:
        timestamps: DatetimeIndex with timestamps

    Returns:
        Boolean array (True = high tariff)
    """
    hours = timestamps.hour + timestamps.minute / 60
    weekday = timestamps.dayofweek  # 0=Mon, 6=Sun

    # Mon-Fri 06:00-21:00
    weekday_high = (weekday < 5) & (hours >= 6) & (hours < 21)

    # Saturday 06:00-12:00
    saturday_high = (weekday == 5) & (hours >= 6) & (hours < 12)

    return weekday_high | saturday_high


def calculate_electricity_cost(grid_import: np.ndarray, grid_export: np.ndarray,
                               is_high: np.ndarray = None,
                               tariff_params: Dict = None) -> Tuple[float, float, float]:
    """
    Calculate electricity costs with tariff awareness.

    Args:
        grid_import: Grid import array (kWh per interval)
        grid_export: Grid export array (kWh per interval)
        is_high: Boolean array for high tariff periods
        tariff_params: Tariff parameters (default: TARIFF_PARAMS)

    Returns:
        Tuple of (grid_cost_chf, feedin_revenue_chf, net_cost_chf)
    """
    if tariff_params is None:
        tariff_params = TARIFF_PARAMS

    grid_import = np.asarray(grid_import)
    grid_export = np.asarray(grid_export)

    if is_high is None:
        # Assume all high tariff if not provided
        purchase_rate = tariff_params['high_rate_rp']
    else:
        is_high = np.asarray(is_high)
        high_rate = tariff_params['high_rate_rp']
        low_rate = tariff_params['low_rate_rp']
        purchase_rate = np.where(is_high, high_rate, low_rate)

    # Grid costs
    grid_cost_rp = np.sum(grid_import * purchase_rate)
    grid_cost_chf = grid_cost_rp / 100

    # Feed-in revenue
    feedin_rate = tariff_params['feedin_rate_rp']
    feedin_revenue_rp = np.sum(grid_export * feedin_rate)
    feedin_revenue_chf = feedin_revenue_rp / 100

    # Net cost
    net_cost_chf = grid_cost_chf - feedin_revenue_chf

    return grid_cost_chf, feedin_revenue_chf, net_cost_chf


# ============================================================================
# Consumption Model
# ============================================================================

def estimate_consumption(t_outdoor: np.ndarray,
                        heating_energy_per_interval: np.ndarray = None,
                        params: Dict = None) -> np.ndarray:
    """
    Estimate total consumption using HDD-based model.

    Consumption = base_load + heating_coef × HDD

    Args:
        t_outdoor: Outdoor temperature array (°C)
        heating_energy_per_interval: Optional heating energy array (overrides HDD model)
        params: Consumption parameters

    Returns:
        Consumption array (kW average per interval)
    """
    if params is None:
        params = CONSUMPTION_PARAMS

    t_outdoor = np.asarray(t_outdoor)

    # Base load
    base_load = params['base_load_kw']

    if heating_energy_per_interval is not None:
        # Use provided heating energy
        return base_load + np.asarray(heating_energy_per_interval)

    # HDD-based estimate
    hdd = np.maximum(0, params['hdd_base_temp'] - t_outdoor)
    heating = params['heating_coef_kw'] * hdd

    return base_load + heating


# ============================================================================
# Full Energy System Simulation
# ============================================================================

def simulate_energy_system(timestamps: pd.DatetimeIndex,
                           t_outdoor: np.ndarray,
                           pv_generation: np.ndarray,
                           setpoint_comfort: float,
                           setpoint_eco: float,
                           comfort_start: float,
                           comfort_end: float,
                           curve_rise: float,
                           dt_hours: float = 0.25,
                           battery_params: Dict = None,
                           cop_params: Dict = None,
                           heating_curve_params: Dict = None,
                           tariff_params: Dict = None) -> Dict:
    """
    Full energy system simulation with all components.

    Args:
        timestamps: DatetimeIndex
        t_outdoor: Outdoor temperature (°C)
        pv_generation: PV generation (kWh per interval)
        setpoint_comfort: Comfort setpoint (°C)
        setpoint_eco: Eco setpoint (°C)
        comfort_start: Comfort period start (hour)
        comfort_end: Comfort period end (hour)
        curve_rise: Heating curve slope
        dt_hours: Time step in hours
        battery_params: Battery parameters
        cop_params: COP model parameters
        heating_curve_params: Heating curve parameters
        tariff_params: Tariff parameters

    Returns:
        Dict with simulation results
    """
    t_outdoor = np.asarray(t_outdoor)
    pv_generation = np.asarray(pv_generation)
    n = len(timestamps)

    # Determine comfort/eco mode
    hours = timestamps.hour + timestamps.minute / 60
    is_comfort = (hours >= comfort_start) & (hours < comfort_end)

    # Calculate T_HK2 from heating curve
    t_hk2 = predict_t_hk2_variable_setpoint(
        t_outdoor, setpoint_comfort, setpoint_eco, curve_rise, is_comfort,
        params=heating_curve_params
    )

    # Calculate COP
    cop = predict_cop(t_outdoor, t_hk2, params=cop_params)

    # Estimate consumption (simplified: base + HDD-based heating)
    consumption = estimate_consumption(t_outdoor)

    # Simulate battery (pass timestamps for time-of-use logic)
    soc, grid_import, grid_export, battery_flow = simulate_battery_soc(
        pv_generation, consumption * dt_hours,  # Convert power to energy
        dt_hours=dt_hours,
        battery_params=battery_params,
        timestamps=timestamps
    )

    # Calculate tariff periods
    is_high = is_high_tariff(timestamps)

    # Calculate costs
    grid_cost, feedin_revenue, net_cost = calculate_electricity_cost(
        grid_import, grid_export, is_high, tariff_params
    )

    return {
        'timestamps': timestamps,
        't_outdoor': t_outdoor,
        't_hk2': t_hk2,
        'cop': cop,
        'is_comfort': is_comfort,
        'consumption': consumption,
        'pv_generation': pv_generation,
        'battery_soc': soc,
        'battery_flow': battery_flow,
        'grid_import': grid_import,
        'grid_export': grid_export,
        'is_high_tariff': is_high,
        'grid_cost_chf': grid_cost,
        'feedin_revenue_chf': feedin_revenue,
        'net_cost_chf': net_cost,
        # Summary metrics
        'total_grid_import_kwh': np.sum(grid_import),
        'total_grid_export_kwh': np.sum(grid_export),
        'total_pv_kwh': np.sum(pv_generation),
        'total_consumption_kwh': np.sum(consumption * dt_hours),
        'avg_cop': np.mean(cop[is_comfort]),  # COP during comfort hours
        'self_sufficiency': 1 - np.sum(grid_import) / max(np.sum(consumption * dt_hours), 0.1),
    }


def simulate_strategy_comparison(sim_data: pd.DataFrame,
                                  baseline_params: Dict,
                                  strategy_params: Dict,
                                  dt_hours: float = 0.25) -> Dict:
    """
    Compare two strategies on historical data.

    Args:
        sim_data: DataFrame with T_outdoor, pv_generation columns
        baseline_params: Dict with setpoint_comfort, setpoint_eco, comfort_start, comfort_end, curve_rise
        strategy_params: Dict with same keys as baseline
        dt_hours: Time step

    Returns:
        Dict with comparison metrics
    """
    timestamps = pd.DatetimeIndex(sim_data.index)
    t_outdoor = sim_data['T_outdoor'].values
    pv = sim_data.get('pv_generation', sim_data.get('pv_generation_kwh', pd.Series(0))).values

    # Simulate baseline
    baseline = simulate_energy_system(
        timestamps, t_outdoor, pv,
        baseline_params['setpoint_comfort'],
        baseline_params['setpoint_eco'],
        baseline_params['comfort_start'],
        baseline_params['comfort_end'],
        baseline_params['curve_rise'],
        dt_hours=dt_hours
    )

    # Simulate strategy
    strategy = simulate_energy_system(
        timestamps, t_outdoor, pv,
        strategy_params['setpoint_comfort'],
        strategy_params['setpoint_eco'],
        strategy_params['comfort_start'],
        strategy_params['comfort_end'],
        strategy_params['curve_rise'],
        dt_hours=dt_hours
    )

    return {
        'baseline': baseline,
        'strategy': strategy,
        'delta_grid_kwh': strategy['total_grid_import_kwh'] - baseline['total_grid_import_kwh'],
        'delta_cost_chf': strategy['net_cost_chf'] - baseline['net_cost_chf'],
        'delta_cop': strategy['avg_cop'] - baseline['avg_cop'],
        'delta_self_sufficiency': strategy['self_sufficiency'] - baseline['self_sufficiency'],
    }
