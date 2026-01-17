#!/usr/bin/env python3
"""
Generate main report (index.html) for the ESTAT project.

This script creates a combined report with table of contents that links to
all phase reports in their subdirectories.

Uses shared CSS styles based on statistik.bs.ch design system.
"""

from pathlib import Path
from datetime import datetime
import sys

# Add src to path for shared imports
sys.path.insert(0, str(Path(__file__).parent))
from shared.report_style import CSS, COLORS

# Paths
OUTPUT_DIR = Path(__file__).parent.parent / 'output'

# Phase definitions with summaries
PHASES = [
    {
        'id': 'phase1',
        'title': 'Phase 1: Data Preprocessing',
        'report': 'phase1/phase1_report.html',
        'summary': '''
            <p>Data preprocessing pipeline for energy balance and sensor data.</p>
            <ul>
                <li>Energy balance data: 15-min intervals from 2023-03 to present</li>
                <li>Sensor data: 96 heating sensors, 27 weather sensors, room temperatures</li>
                <li>Tariff data: Primeo Energie rates with high/low tariff windows</li>
                <li>Output: Cleaned parquet files with integrated dataset</li>
            </ul>
        ''',
        'key_outputs': [
            ('integrated_dataset.parquet', 'Merged energy + sensor data'),
            ('tariff_schedule.csv', 'Electricity tariff rates'),
        ]
    },
    {
        'id': 'phase2',
        'title': 'Phase 2: Exploratory Data Analysis',
        'report': 'phase2/phase2_report.html',
        'summary': '''
            <p>Comprehensive analysis of energy patterns, heating system, and solar correlation.</p>
            <ul>
                <li>Energy: 55.9 kWh/day PV generation, 44% self-sufficiency</li>
                <li>Heat pump: COP 3.55 average, heating curve model R²=0.96</li>
                <li>Solar-heating overlap: Only 4.7% forced grid consumption</li>
                <li>Battery degradation: 5-11pp efficiency drop after deep discharge event</li>
            </ul>
        ''',
        'key_outputs': [
            ('fig01-fig11', 'Energy and heating analysis'),
            ('fig12', 'Heating curve schedule detection'),
            ('fig13', 'Weighted temperature response'),
            ('fig14-fig16', 'Tariff analysis'),
            ('fig17', 'HK2 target vs actual temperature'),
        ],
        'figures': [
            ('fig01_daily_energy_timeseries.png', 'Daily Energy Time Series'),
            ('fig05_heat_pump_cop.png', 'Heat Pump COP Analysis'),
            ('fig12_heating_curve_schedule.png', 'Heating Curve Schedule'),
            ('fig17_hk2_target_actual.png', 'HK2 Target vs Actual'),
        ]
    },
    {
        'id': 'phase3',
        'title': 'Phase 3: System Modeling',
        'report': 'phase3/phase3_report.html',
        'summary': '''
            <p>Physics-based models for thermal dynamics, heat pump, and energy system.</p>
            <ul>
                <li>Thermal model: R²=0.68, building time constant ~24h</li>
                <li>Transfer function with causal coefficients (g_eff=0.208)</li>
                <li>COP model: COP = 5.93 + 0.13×T_outdoor - 0.08×T_HK2 (R²=0.94)</li>
                <li>Self-sufficiency potential: 57% → 84% with optimization</li>
            </ul>
        ''',
        'key_outputs': [
            ('fig18', 'Transfer function thermal model'),
            ('fig19', 'Heat pump COP model'),
            ('fig20', 'Energy system analysis'),
            ('fig21', 'Tariff cost model'),
            ('causal_coefficients.json', 'Physics-based coefficients for Phase 4'),
        ],
        'figures': [
            ('fig18_thermal_model.png', 'Thermal Model'),
            ('fig19_heat_pump_model.png', 'Heat Pump Model'),
            ('fig_transfer_function_integration.png', 'Causal Coefficients'),
        ]
    },
    {
        'id': 'phase4',
        'title': 'Phase 4: Optimization',
        'report': 'phase4/phase4_report.html',
        'summary': '''
            <p>Multi-objective optimization using NSGA-II for heating strategies.</p>
            <ul>
                <li>Three objectives: Maximize comfort, minimize grid, minimize cost</li>
                <li>Uses causal coefficients from Phase 3 (g_eff=0.208)</li>
                <li>Optimal strategies: 22°C setpoint, 12:00-16:00 schedule, curve_rise 1.0-1.2</li>
                <li>Grid-minimal: 7% reduction with 0% comfort violations</li>
            </ul>
        ''',
        'key_outputs': [
            ('fig22-fig24', 'Strategy comparison and simulation'),
            ('fig25-fig27', 'Pareto optimization results'),
            ('fig28-fig31', 'Detailed strategy analysis'),
            ('selected_strategies.json', 'Pareto-optimal parameters'),
        ],
        'figures': [
            ('fig25_pareto_front.png', 'Pareto Front'),
            ('fig28_strategy_temperature_predictions.png', 'Temperature Predictions'),
        ]
    },
    {
        'id': 'phase5_pilot',
        'title': 'Phase 5 Pilot: Parameter Exploration',
        'report': 'phase5_pilot/pilot_protocol.html',
        'summary': '''
            <p>T_HK2-targeted pilot experiment (Jan-Mar 2026) to learn thermal response.</p>
            <ul>
                <li>10 blocks × 7 days with varied heating parameters</li>
                <li>T_HK2 spread: 9.5°C (comfort), 10.7°C (eco)</li>
                <li>Dynamical analysis using grey-box model (no washout needed)</li>
                <li>Goal: Learn T_indoor = f(T_HK2 history, T_outdoor, thermal_mass)</li>
            </ul>
        ''',
        'key_outputs': [
            ('pilot_schedule.json', 'Block schedule with parameters'),
            ('thk2_design.csv', 'T_HK2-targeted design matrix'),
        ]
    },
    {
        'id': 'phase5',
        'title': 'Phase 5: Intervention Study (Future)',
        'report': 'phase5/experimental_protocol.html',
        'summary': '''
            <p>Planned randomized crossover study for winter 2027-2028.</p>
            <ul>
                <li>20 weeks with 3 strategies (Baseline, Grid-Minimal, Balanced)</li>
                <li>7-day blocks with 3-day washout + 4-day measurement</li>
                <li>Statistical power: 97% to detect +0.30 COP change</li>
            </ul>
        ''',
        'key_outputs': [
            ('experimental_protocol.html', 'Full study protocol'),
        ]
    },
]

# Additional CSS for main report layout
EXTRA_CSS = f"""
/* Main report specific styles */
header {{
    text-align: center;
    margin-bottom: 3rem;
    padding-bottom: 2rem;
    border-bottom: 3px solid {COLORS['primary_green']};
}}

header h1 {{
    border-bottom: none;
    padding-bottom: 0;
}}

header .subtitle {{
    color: {COLORS['gray_dark']};
    font-size: 1.2rem;
    margin-top: 0.5rem;
}}

header .timestamp {{
    color: {COLORS['gray_medium']};
    font-size: 0.9rem;
    margin-top: 1rem;
}}

#toc {{
    background: white;
    padding: 1.5rem 2rem;
    border-radius: 4px;
    margin-bottom: 2rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}}

#toc h2 {{
    font-size: 1.2rem;
    margin-bottom: 1rem;
    margin-top: 0;
    color: {COLORS['dark_teal']};
    border-bottom: none;
}}

#toc ol {{
    columns: 2;
    column-gap: 2rem;
    padding-left: 1.5rem;
    margin-bottom: 0;
}}

#toc li {{
    margin-bottom: 0.5rem;
}}

section {{
    background: white;
    padding: 2rem;
    border-radius: 4px;
    margin-bottom: 2rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}}

section h2 {{
    margin-top: 0;
}}

.figure-gallery {{
    display: flex;
    flex-wrap: wrap;
    gap: 1rem;
    margin: 1.5rem 0;
}}

.thumbnail {{
    flex: 1 1 200px;
    max-width: 280px;
    text-align: center;
}}

.thumbnail img {{
    width: 100%;
    height: 150px;
    object-fit: cover;
    border-radius: 4px;
    border: 1px solid {COLORS['gray_border']};
    transition: transform 0.2s, box-shadow 0.2s;
}}

.thumbnail img:hover {{
    transform: scale(1.02);
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}}

.thumbnail p {{
    font-size: 0.85rem;
    color: {COLORS['gray_dark']};
    margin-top: 0.5rem;
}}

.report-link {{
    margin-top: 1.5rem;
    padding-top: 1rem;
    border-top: 1px solid {COLORS['gray_border']};
}}

.button {{
    display: inline-block;
    background: {COLORS['primary_dark_blue']};
    color: white;
    padding: 0.6rem 1.2rem;
    border-radius: 4px;
    text-decoration: none;
    font-weight: 500;
    transition: background 0.15s;
}}

.button:hover {{
    background: {COLORS['dark_teal']};
    color: white;
    text-decoration: none;
}}

.footer {{
    text-align: center;
    padding: 2rem;
    color: {COLORS['gray_dark']};
    font-size: 0.9rem;
    border-top: 1px solid {COLORS['gray_border']};
    margin-top: 2rem;
}}

@media (max-width: 768px) {{
    #toc ol {{
        columns: 1;
    }}

    .figure-gallery {{
        justify-content: center;
    }}
}}
"""


def generate_toc(phases: list) -> str:
    """Generate table of contents HTML."""
    items = []
    for phase in phases:
        items.append(f'<li><a href="#{phase["id"]}">{phase["title"]}</a></li>')

    return f'''
    <nav id="toc">
        <h2>Table of Contents</h2>
        <ol>
            {''.join(items)}
        </ol>
    </nav>
    '''


def generate_phase_section(phase: dict) -> str:
    """Generate HTML section for a phase."""

    # Key outputs list
    outputs_html = ''
    if phase.get('key_outputs'):
        items = [f'<li><code>{k}</code>: {v}</li>' for k, v in phase['key_outputs']]
        outputs_html = f'<h4>Key Outputs</h4><ul>{"".join(items)}</ul>'

    # Figure thumbnails
    figures_html = ''
    if phase.get('figures'):
        thumbs = []
        for fig_file, fig_title in phase['figures']:
            fig_path = f'{phase["id"]}/{fig_file}'
            thumbs.append(f'''
                <div class="thumbnail">
                    <a href="{fig_path}">
                        <img src="{fig_path}" alt="{fig_title}">
                    </a>
                    <p>{fig_title}</p>
                </div>
            ''')
        figures_html = f'<div class="figure-gallery">{"".join(thumbs)}</div>'

    return f'''
    <section id="{phase['id']}">
        <h2>{phase['title']}</h2>
        {phase['summary']}
        {outputs_html}
        {figures_html}
        <p class="report-link">
            <a href="{phase['report']}" class="button">View Full Report &rarr;</a>
        </p>
    </section>
    '''


def generate_main_report():
    """Generate the main index.html report."""

    # Generate sections
    toc = generate_toc(PHASES)
    sections = '\n'.join(generate_phase_section(p) for p in PHASES)

    # Get timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')

    html = f'''<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ESTAT - Energy System Analysis Report</title>
    <style>
{CSS}
{EXTRA_CSS}
    </style>
</head>
<body>
    <header>
        <h1>ESTAT Analysis Report</h1>
        <p class="subtitle">Energy System Optimization for Solar/Heat Pump Integration</p>
        <p class="timestamp">Generated: {timestamp}</p>
    </header>

    {toc}

    <main>
        {sections}
    </main>

    <div class="footer">
        <p>ESTAT - Energy Balance Data Repository</p>
    </div>
</body>
</html>
'''

    # Write report
    output_path = OUTPUT_DIR / 'index.html'
    output_path.write_text(html)
    print(f"Generated: {output_path}")

    return output_path


def main():
    """Main entry point."""
    print("=" * 60)
    print("Generating Main Report")
    print("=" * 60)

    generate_main_report()

    print("\nDone!")


if __name__ == '__main__':
    main()
