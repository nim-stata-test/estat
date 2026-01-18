"""Shared utilities for ESTAT project."""

import pandas as pd

# Analysis start date - good sensor data begins Oct 29, 2025
# All model estimation and weekly analyses should use this as the start
ANALYSIS_START_DATE = pd.Timestamp('2025-10-29')

from .report_style import (
    CSS, COLORS,
    get_html_head, get_html_footer, wrap_html,
    create_metric_card, create_metrics_grid, create_info_box, create_figure
)

from .energy_system import (
    BATTERY_PARAMS,
    simulate_battery_soc,
    predict_cop,
    predict_t_hk2_variable_setpoint,
    is_high_tariff,
    calculate_electricity_cost,
    simulate_energy_system,
)

__all__ = [
    # Analysis constants
    'ANALYSIS_START_DATE',
    # Report styling (statistik.bs.ch design)
    'CSS', 'COLORS',
    'get_html_head', 'get_html_footer', 'wrap_html',
    'create_metric_card', 'create_metrics_grid', 'create_info_box', 'create_figure',
    # Energy system simulation
    'BATTERY_PARAMS',
    'simulate_battery_soc',
    'predict_cop',
    'predict_t_hk2_variable_setpoint',
    'is_high_tariff',
    'calculate_electricity_cost',
    'simulate_energy_system',
]
