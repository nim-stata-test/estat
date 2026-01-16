"""
Shared Grey-Box Thermal Simulator Module

Provides forward simulation capabilities for Phase 4 optimization
using the grey-box state-space model from Phase 3.

Model Equations (discrete-time, dt = 0.25h):

    T_buf[k+1] = T_buf[k] + (dt/tau_buf) * [(T_HK2[k] - T_buf[k]) - r_emit*(T_buf[k] - T_room[k])]
    T_room[k+1] = T_room[k] + (dt/tau_room) * [r_heat*(T_buf[k] - T_room[k]) - (T_room[k] - T_out[k])] + k_solar*PV[k]

State Variables:
    T_buf   : Buffer tank temperature (intermediate thermal storage)
    T_room  : Room temperature (comfort objective)

Parameters:
    tau_buf : Buffer tank time constant (hours)
    tau_room: Building time constant (hours)
    r_emit  : Emitter/HP coupling ratio (dimensionless)
    r_heat  : Heat transfer ratio (dimensionless)
    k_solar : Solar gain coefficient (K/kWh)
    c_offset: Temperature drift correction (K)

Heating Curve Model (computes T_HK2 from controllable parameters):
    T_HK2 = T_setpoint + curve_rise * (T_ref - T_outdoor)

Where T_ref depends on comfort/eco mode (from Phase 2 heating curve analysis).
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Union


# Default time step (15 minutes = 0.25 hours)
DT_HOURS = 0.25

# Parameter names in order
PARAM_NAMES = ['tau_buf', 'tau_room', 'r_emit', 'r_heat', 'k_solar', 'c_offset']


def load_greybox_params(path: Union[str, Path]) -> Dict:
    """
    Load grey-box model parameters from JSON file.

    Args:
        path: Path to greybox_model_params.json

    Returns:
        Dictionary with 'params' (model parameters) and metadata
    """
    with open(path) as f:
        data = json.load(f)
    return data


def simulate_forward(params: Union[Dict, np.ndarray], x0: np.ndarray,
                     u_inputs: np.ndarray, dt: float = DT_HOURS) -> np.ndarray:
    """
    Forward simulate the two-state thermal model (recursive).

    This compounds predictions over time - each step uses the predicted
    (not observed) state from the previous step.

    Args:
        params: Model parameters as dict {'tau_buf': ..., ...} or
                array [tau_buf, tau_room, r_emit, r_heat, k_solar, c_offset]
        x0: Initial state [T_buffer_0, T_room_0]
        u_inputs: Inputs [T_HK2, T_outdoor, PV] shape (n, 3)
        dt: Time step in hours (default 0.25)

    Returns:
        x_pred: Predicted states shape (n, 2) where columns are [T_buf, T_room]
    """
    # Extract parameters
    if isinstance(params, dict):
        if 'params' in params:
            params = params['params']
        tau_buf = params['tau_buf']
        tau_room = params['tau_room']
        r_emit = params['r_emit']
        r_heat = params['r_heat']
        k_solar = params['k_solar']
        c_offset = params['c_offset']
    else:
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
        x_pred[k + 1, 1] = T_room + dT_room + c_offset * dt

    return x_pred


def compute_T_HK2_from_schedule(
    T_outdoor: np.ndarray,
    hours: np.ndarray,
    schedule: Dict,
    t_ref_comfort: float,
    t_ref_eco: float
) -> np.ndarray:
    """
    Compute target flow temperature (T_HK2) from heating curve and schedule.

    Uses the parametric heating curve model:
        T_HK2 = T_setpoint + curve_rise * (T_ref - T_outdoor)

    Args:
        T_outdoor: Outdoor temperature array (n,)
        hours: Hour of day for each timestep (0-24, can be float)
        schedule: Dict with keys:
            - setpoint_comfort: Comfort mode target temperature
            - setpoint_eco: Eco mode target temperature
            - comfort_start: Start hour of comfort period (e.g., 6.5 for 06:30)
            - comfort_end: End hour of comfort period (e.g., 20.0 for 20:00)
            - curve_rise: Heating curve slope (Steilheit)
        t_ref_comfort: Reference temperature for comfort mode (from Phase 2)
        t_ref_eco: Reference temperature for eco mode (from Phase 2)

    Returns:
        T_HK2: Target flow temperature array (n,)
    """
    setpoint_comfort = schedule['setpoint_comfort']
    setpoint_eco = schedule['setpoint_eco']
    comfort_start = schedule['comfort_start']
    comfort_end = schedule['comfort_end']
    curve_rise = schedule['curve_rise']

    n = len(T_outdoor)
    T_HK2 = np.zeros(n)

    for i in range(n):
        hour = hours[i]
        T_out = T_outdoor[i]

        # Determine if in comfort or eco mode based on schedule
        if comfort_start <= hour < comfort_end:
            # Comfort mode
            T_setpoint = setpoint_comfort
            T_ref = t_ref_comfort
        else:
            # Eco mode
            T_setpoint = setpoint_eco
            T_ref = t_ref_eco

        # Heating curve formula
        T_HK2[i] = T_setpoint + curve_rise * (T_ref - T_out)

    return T_HK2


def compute_T_HK2_from_schedule_vectorized(
    T_outdoor: np.ndarray,
    hours: np.ndarray,
    schedule: Dict,
    t_ref_comfort: float,
    t_ref_eco: float
) -> np.ndarray:
    """
    Vectorized version of compute_T_HK2_from_schedule for better performance.
    """
    setpoint_comfort = schedule['setpoint_comfort']
    setpoint_eco = schedule['setpoint_eco']
    comfort_start = schedule['comfort_start']
    comfort_end = schedule['comfort_end']
    curve_rise = schedule['curve_rise']

    # Boolean mask for comfort mode
    is_comfort = (hours >= comfort_start) & (hours < comfort_end)

    # Compute T_HK2 for both modes
    T_setpoint = np.where(is_comfort, setpoint_comfort, setpoint_eco)
    T_ref = np.where(is_comfort, t_ref_comfort, t_ref_eco)

    # Heating curve formula
    T_HK2 = T_setpoint + curve_rise * (T_ref - T_outdoor)

    return T_HK2


class ThermalSimulator:
    """
    Encapsulates grey-box thermal model for simulation.

    Provides convenient interface for Phase 4 optimization.
    """

    def __init__(self, params_path: Optional[Union[str, Path]] = None,
                 params: Optional[Dict] = None,
                 heating_curve_params: Optional[Dict] = None):
        """
        Initialize simulator with grey-box parameters.

        Args:
            params_path: Path to greybox_model_params.json
            params: Alternatively, provide params dict directly
            heating_curve_params: Dict with t_ref_comfort, t_ref_eco
        """
        if params_path is not None:
            data = load_greybox_params(params_path)
            self.params = data['params']
            self.fit_stats = data.get('fit_stats', {})
        elif params is not None:
            self.params = params
            self.fit_stats = {}
        else:
            raise ValueError("Must provide either params_path or params")

        # Heating curve reference temperatures
        if heating_curve_params is not None:
            self.t_ref_comfort = heating_curve_params['t_ref_comfort']
            self.t_ref_eco = heating_curve_params['t_ref_eco']
        else:
            # Default values from Phase 2 analysis
            self.t_ref_comfort = 21.32
            self.t_ref_eco = 19.16

        self.dt = DT_HOURS

    def simulate(self, x0: np.ndarray, u_inputs: np.ndarray,
                 warmup_steps: int = 0) -> np.ndarray:
        """
        Run forward simulation.

        Args:
            x0: Initial state [T_buffer, T_room]
            u_inputs: Input array [T_HK2, T_outdoor, PV] shape (n, 3)
            warmup_steps: Number of steps to skip at start (warmup period)

        Returns:
            Predicted states shape (n - warmup_steps, 2)
        """
        x_pred = simulate_forward(self.params, x0, u_inputs, self.dt)

        if warmup_steps > 0:
            return x_pred[warmup_steps:]
        return x_pred

    def simulate_with_schedule(
        self,
        schedule: Dict,
        T_outdoor: np.ndarray,
        hours: np.ndarray,
        PV: np.ndarray,
        x0: np.ndarray,
        warmup_steps: int = 96
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate with a specific heating schedule.

        Computes T_HK2 from heating curve and schedule, then runs simulation.

        Args:
            schedule: Dict with setpoint_comfort, setpoint_eco, comfort_start,
                      comfort_end, curve_rise
            T_outdoor: Outdoor temperature array
            hours: Hour of day for each timestep
            PV: PV generation array (kWh per 15-min interval)
            x0: Initial state [T_buffer, T_room]
            warmup_steps: Steps to skip at start (default 96 = 24h)

        Returns:
            Tuple of (T_room_sim, T_HK2_sim) after warmup period
        """
        # Compute T_HK2 from schedule
        T_HK2 = compute_T_HK2_from_schedule_vectorized(
            T_outdoor, hours, schedule,
            self.t_ref_comfort, self.t_ref_eco
        )

        # Build input array
        u_inputs = np.column_stack([T_HK2, T_outdoor, PV])

        # Run simulation
        x_pred = simulate_forward(self.params, x0, u_inputs, self.dt)

        # Extract room temperature and T_HK2 after warmup
        if warmup_steps > 0:
            T_room_sim = x_pred[warmup_steps:, 1]
            T_HK2_sim = T_HK2[warmup_steps:]
        else:
            T_room_sim = x_pred[:, 1]
            T_HK2_sim = T_HK2

        return T_room_sim, T_HK2_sim

    def get_steady_state_gain(self) -> float:
        """
        Compute steady-state gain dT_room/dT_HK2.

        This represents how much the room temperature changes in equilibrium
        for a 1-degree change in flow temperature.

        Returns:
            Steady-state gain (dimensionless, typically 0.1-0.5)
        """
        # From steady-state analysis of the model equations
        tau_buf = self.params['tau_buf']
        tau_room = self.params['tau_room']
        r_emit = self.params['r_emit']
        r_heat = self.params['r_heat']

        # Steady-state: dT_buf = 0, dT_room = 0
        # Solving the system analytically...
        # Gain = r_heat / (r_heat + 1) * 1 / (1 + r_emit * tau_buf / tau_room * (r_heat / (r_heat + 1)))
        # Simplified approximation for typical parameter ranges:
        gain = r_heat / (r_heat + 1 + r_emit)

        return gain
