"""Shared utilities for ESTAT project."""

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
