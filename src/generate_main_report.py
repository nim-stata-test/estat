#!/usr/bin/env python3
"""
Generate main report (index.html) for the ESTAT project.

This script creates a combined report with table of contents that links to
all phase reports in their subdirectories.
"""

from pathlib import Path
from datetime import datetime

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
                <li>Grey-box model: Two-state (buffer + room) with physical parameters</li>
                <li>COP model: COP = 5.93 + 0.13×T_outdoor - 0.08×T_HK2 (R²=0.94)</li>
                <li>Self-sufficiency potential: 57% → 84% with optimization</li>
            </ul>
        ''',
        'key_outputs': [
            ('fig18', 'Transfer function thermal model'),
            ('fig18b', 'Grey-box thermal model'),
            ('fig19', 'Heat pump COP model'),
            ('fig20', 'Energy system analysis'),
            ('fig21', 'Tariff cost model'),
        ],
        'figures': [
            ('fig18_thermal_model.png', 'Thermal Model'),
            ('fig18b_greybox_model.png', 'Grey-Box Model'),
            ('fig19_heat_pump_model.png', 'Heat Pump Model'),
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
                <li>Pareto-optimal strategies with 2.9% comfort violation (≤5% target)</li>
                <li>Key levers: comfort schedule, setpoint, curve rise</li>
                <li>Grid-minimal strategy: 10% reduction in grid import</li>
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
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ESTAT - Energy System Analysis Report</title>
    <style>
        :root {{
            --primary: #2563eb;
            --primary-dark: #1d4ed8;
            --bg: #f8fafc;
            --card-bg: #ffffff;
            --text: #1e293b;
            --text-muted: #64748b;
            --border: #e2e8f0;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            color: var(--text);
            background: var(--bg);
            padding: 2rem;
            max-width: 1200px;
            margin: 0 auto;
        }}

        header {{
            text-align: center;
            margin-bottom: 3rem;
            padding-bottom: 2rem;
            border-bottom: 1px solid var(--border);
        }}

        h1 {{
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            color: var(--primary);
        }}

        header p {{
            color: var(--text-muted);
            font-size: 1.1rem;
        }}

        #toc {{
            background: var(--card-bg);
            padding: 1.5rem 2rem;
            border-radius: 8px;
            margin-bottom: 2rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}

        #toc h2 {{
            font-size: 1.2rem;
            margin-bottom: 1rem;
            color: var(--text-muted);
        }}

        #toc ol {{
            columns: 2;
            column-gap: 2rem;
            padding-left: 1.5rem;
        }}

        #toc li {{
            margin-bottom: 0.5rem;
        }}

        #toc a {{
            color: var(--primary);
            text-decoration: none;
        }}

        #toc a:hover {{
            text-decoration: underline;
        }}

        section {{
            background: var(--card-bg);
            padding: 2rem;
            border-radius: 8px;
            margin-bottom: 2rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}

        section h2 {{
            font-size: 1.5rem;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid var(--primary);
        }}

        section p {{
            margin-bottom: 1rem;
        }}

        section ul {{
            margin-left: 1.5rem;
            margin-bottom: 1rem;
        }}

        section li {{
            margin-bottom: 0.25rem;
        }}

        h4 {{
            font-size: 1rem;
            color: var(--text-muted);
            margin: 1.5rem 0 0.75rem 0;
        }}

        code {{
            background: #f1f5f9;
            padding: 0.15rem 0.4rem;
            border-radius: 4px;
            font-size: 0.9em;
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
            border: 1px solid var(--border);
            transition: transform 0.2s;
        }}

        .thumbnail img:hover {{
            transform: scale(1.02);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}

        .thumbnail p {{
            font-size: 0.85rem;
            color: var(--text-muted);
            margin-top: 0.5rem;
        }}

        .report-link {{
            margin-top: 1.5rem;
            padding-top: 1rem;
            border-top: 1px solid var(--border);
        }}

        .button {{
            display: inline-block;
            background: var(--primary);
            color: white;
            padding: 0.6rem 1.2rem;
            border-radius: 6px;
            text-decoration: none;
            font-weight: 500;
            transition: background 0.2s;
        }}

        .button:hover {{
            background: var(--primary-dark);
        }}

        footer {{
            text-align: center;
            padding: 2rem;
            color: var(--text-muted);
            font-size: 0.9rem;
        }}

        @media (max-width: 768px) {{
            body {{
                padding: 1rem;
            }}

            #toc ol {{
                columns: 1;
            }}

            .figure-gallery {{
                justify-content: center;
            }}
        }}
    </style>
</head>
<body>
    <header>
        <h1>ESTAT Analysis Report</h1>
        <p>Energy System Optimization for Solar/Heat Pump Integration</p>
        <p><small>Generated: {timestamp}</small></p>
    </header>

    {toc}

    <main>
        {sections}
    </main>

    <footer>
        <p>ESTAT - Energy Balance Data Repository</p>
        <p>Report generated automatically from analysis pipeline</p>
    </footer>
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
