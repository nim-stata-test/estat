"""Shared utilities for ESTAT project."""

from .thermal_simulator import (
    load_greybox_params,
    simulate_forward,
    compute_T_HK2_from_schedule,
    ThermalSimulator,
)

from .report_style import (
    CSS, COLORS,
    get_html_head, get_html_footer, wrap_html,
    create_metric_card, create_metrics_grid, create_info_box, create_figure
)

__all__ = [
    # Thermal simulator
    'load_greybox_params',
    'simulate_forward',
    'compute_T_HK2_from_schedule',
    'ThermalSimulator',
    # Report styling (statistik.bs.ch design)
    'CSS', 'COLORS',
    'get_html_head', 'get_html_footer', 'wrap_html',
    'create_metric_card', 'create_metrics_grid', 'create_info_box', 'create_figure',
]
