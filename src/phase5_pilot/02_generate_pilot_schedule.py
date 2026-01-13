#!/usr/bin/env python3
"""
Phase 5 Pilot: Generate Pilot Schedule

Takes the T_HK2-targeted design and creates a dated schedule for
the pilot experiment starting Jan 13, 2026.

Usage:
    python src/phase5_pilot/02_generate_pilot_schedule.py
    python src/phase5_pilot/02_generate_pilot_schedule.py --start 2026-01-13 --seed 42

Outputs:
    output/phase5_pilot/pilot_schedule.csv - Block schedule
    output/phase5_pilot/pilot_schedule.json - Machine-readable format
    output/phase5_pilot/pilot_protocol.html - Human-readable protocol
"""

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'output' / 'phase5_pilot'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Block structure
BLOCK_DAYS = 7
WASHOUT_DAYS = 2  # For RSM analysis only; dynamical analysis uses all data
MEASUREMENT_DAYS = 5  # For RSM analysis only

# Safety constraints
SAFETY = {
    'min_temp_floor': 17.0,  # Minimum T_weighted allowed (°C)
    'min_cop': 2.0,           # Minimum COP before checking heat pump
    'max_violation_pct': 50,  # Max % below 18.5°C (exploration allows more)
}


def load_design() -> pd.DataFrame:
    """Load the T_HK2-targeted design from previous step."""
    # Try new T_HK2 design first, fall back to old LHD design
    design_path = OUTPUT_DIR / 'thk2_design.csv'
    if not design_path.exists():
        design_path = OUTPUT_DIR / 'lhd_design.csv'
        if not design_path.exists():
            raise FileNotFoundError(
                f"Design file not found.\n"
                "Run 01_generate_thk2_design.py first."
            )

    df = pd.read_csv(design_path)
    print(f"Loaded design with {len(df)} points from {design_path}")
    return df


def randomize_sequence(n_blocks: int, seed: int) -> list:
    """
    Randomize the order of design points.

    Ensures no obvious patterns in the sequence.
    """
    rng = np.random.default_rng(seed)
    sequence = list(range(1, n_blocks + 1))
    rng.shuffle(sequence)
    return sequence


def generate_schedule(
    design_df: pd.DataFrame,
    start_date: datetime,
    seed: int,
) -> pd.DataFrame:
    """Generate the dated pilot schedule."""

    n_blocks = len(design_df)

    print(f"Generating pilot schedule:")
    print(f"  Start date: {start_date.strftime('%Y-%m-%d')}")
    print(f"  Blocks: {n_blocks}")
    print(f"  Block length: {BLOCK_DAYS} days (washout: {WASHOUT_DAYS}, measurement: {MEASUREMENT_DAYS})")
    print(f"  Random seed: {seed}")

    # Randomize order
    sequence = randomize_sequence(n_blocks, seed)

    # Build schedule
    records = []
    current_date = start_date

    for block_num, design_point in enumerate(sequence, 1):
        # Get design point parameters
        design_row = design_df[design_df['design_point'] == design_point].iloc[0]

        block_start = current_date
        block_end = current_date + timedelta(days=BLOCK_DAYS - 1)
        washout_end = current_date + timedelta(days=WASHOUT_DAYS - 1)
        measurement_start = current_date + timedelta(days=WASHOUT_DAYS)

        # Determine season based on date
        month = block_start.month
        if month in [1, 2]:
            season = 'mid-winter'
        elif month == 3:
            season = 'late-winter'
        else:
            season = 'shoulder'

        record = {
            'block': block_num,
            'design_point': design_point,
            'start_date': block_start.strftime('%Y-%m-%d'),
            'end_date': block_end.strftime('%Y-%m-%d'),
            'washout_end': washout_end.strftime('%Y-%m-%d'),
            'measurement_start': measurement_start.strftime('%Y-%m-%d'),
            'season': season,
            'comfort_setpoint': design_row['comfort_setpoint'],
            'eco_setpoint': design_row['eco_setpoint'],
            'curve_rise': design_row['curve_rise'],
            'comfort_hours': design_row['comfort_hours'],
            'comfort_start': design_row['comfort_start'],
            'comfort_end': design_row['comfort_end'],
            'notes': '',
            'status': 'scheduled',
        }

        # Add T_HK2 values if available in design
        if 'T_HK2_comfort' in design_row:
            record['T_HK2_comfort'] = design_row['T_HK2_comfort']
        if 'T_HK2_eco' in design_row:
            record['T_HK2_eco'] = design_row['T_HK2_eco']

        records.append(record)

        current_date = block_end + timedelta(days=1)

    df = pd.DataFrame(records)

    # Print summary
    end_date = df['end_date'].iloc[-1]
    print(f"\nSchedule period: {start_date.strftime('%Y-%m-%d')} to {end_date}")
    print(f"Total duration: {n_blocks * BLOCK_DAYS} days ({n_blocks} weeks)")

    return df


def generate_html_protocol(df: pd.DataFrame, seed: int, output_path: Path) -> None:
    """Generate HTML protocol for pilot experiment."""

    start_date = df['start_date'].iloc[0]
    end_date = df['end_date'].iloc[-1]
    n_blocks = len(df)

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ESTAT Phase 5 Pilot: Experimental Protocol</title>
    <style>
        :root {{
            --primary: #2563eb;
            --success: #16a34a;
            --warning: #d97706;
            --danger: #dc2626;
            --bg: #f8fafc;
            --card-bg: #ffffff;
            --text: #1e293b;
            --text-muted: #64748b;
            --border: #e2e8f0;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
            padding: 2rem;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: var(--primary); margin-bottom: 0.5rem; }}
        h2 {{ color: var(--text); margin: 2rem 0 1rem; padding-bottom: 0.5rem; border-bottom: 2px solid var(--primary); }}
        h3 {{ color: var(--text-muted); margin: 1.5rem 0 0.75rem; }}
        .meta {{ color: var(--text-muted); margin-bottom: 2rem; }}
        .card {{
            background: var(--card-bg);
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem; }}
        .stat-box {{
            background: var(--card-bg);
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .stat-box .value {{ font-size: 1.5rem; font-weight: bold; color: var(--primary); }}
        .stat-box .label {{ color: var(--text-muted); font-size: 0.85rem; }}
        table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
        th, td {{ padding: 0.5rem 0.75rem; text-align: left; border-bottom: 1px solid var(--border); }}
        th {{ background: var(--bg); font-weight: 600; }}
        .block-card {{
            background: var(--card-bg);
            border-radius: 8px;
            padding: 1rem 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            border-left: 4px solid var(--primary);
        }}
        .block-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.5rem;
        }}
        .block-num {{
            font-size: 1.25rem;
            font-weight: bold;
            color: var(--primary);
        }}
        .block-dates {{
            color: var(--text-muted);
            font-size: 0.9rem;
        }}
        .param-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.5rem;
            margin-top: 0.5rem;
        }}
        .param-item {{
            background: var(--bg);
            padding: 0.5rem;
            border-radius: 4px;
            text-align: center;
        }}
        .param-label {{ font-size: 0.75rem; color: var(--text-muted); }}
        .param-value {{ font-weight: 600; }}
        .warning {{
            background: #fef3c7;
            border: 1px solid #fcd34d;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
        }}
        .danger {{
            background: #fee2e2;
            border: 1px solid #fca5a5;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
        }}
        @media print {{
            body {{ padding: 0.5rem; }}
            .card {{ box-shadow: none; border: 1px solid var(--border); }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>ESTAT Phase 5 Pilot: Parameter Exploration</h1>
        <p class="meta">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Seed: {seed}</p>

        <h2>1. Study Overview</h2>

        <div class="grid">
            <div class="stat-box">
                <div class="value">{n_blocks}</div>
                <div class="label">Blocks</div>
            </div>
            <div class="stat-box">
                <div class="value">{BLOCK_DAYS}</div>
                <div class="label">Days/Block</div>
            </div>
            <div class="stat-box">
                <div class="value">All</div>
                <div class="label">Data Used*</div>
            </div>
            <div class="stat-box">
                <div class="value">2</div>
                <div class="label">Analysis Methods</div>
            </div>
        </div>
        <p style="color: #666; font-size: 0.85rem; margin-top: 0.5rem;">
            *Dynamical analysis uses all 7 days. RSM analysis uses days 3-7 (excludes 2-day washout).
        </p>

        <div class="card">
            <h3>Study Period</h3>
            <table>
                <tr><td>Start Date</td><td><strong>{start_date}</strong></td></tr>
                <tr><td>End Date</td><td><strong>{end_date}</strong></td></tr>
                <tr><td>Duration</td><td>{n_blocks * BLOCK_DAYS} days ({n_blocks} weeks)</td></tr>
            </table>
        </div>

        <div class="card">
            <h3>Purpose</h3>
            <p>This pilot experiment explores the heating parameter space using a <strong>T_HK2-targeted design</strong>.
            The design maximizes spread in flow temperature (T_HK2) rather than raw parameters, because
            T_HK2 is what drives the thermal response we need to learn.</p>

            <h4 style="margin-top: 1rem;">Two Analysis Approaches</h4>
            <table style="margin: 0.5rem 0;">
                <tr>
                    <th>Approach</th>
                    <th>Data Used</th>
                    <th>Best For</th>
                </tr>
                <tr>
                    <td><strong>Dynamical</strong> (preferred)</td>
                    <td>All 7 days (~6,700 points)</td>
                    <td>Learning dynamics, model validation</td>
                </tr>
                <tr>
                    <td>RSM</td>
                    <td>Days 3-7 (10 block means)</td>
                    <td>Simple parameter effects</td>
                </tr>
            </table>
            <p style="margin-top: 0.5rem;">The <strong>dynamical analysis</strong> uses the grey-box state-space model
            and treats transitions between settings as informative step responses (no washout needed).</p>
        </div>

        <div class="warning">
            <strong>Safety Constraints</strong>
            <ul style="margin: 0.5rem 0 0 1rem;">
                <li>Minimum T_weighted: <strong>{SAFETY['min_temp_floor']}°C</strong> - abort block if breached</li>
                <li>Minimum COP: <strong>{SAFETY['min_cop']}</strong> - check heat pump if sustained below</li>
                <li>Maximum violation %: <strong>{SAFETY['max_violation_pct']}%</strong> (exploration allows more than Phase 5)</li>
            </ul>
        </div>

        <h2>2. Parameter Ranges</h2>

        <div class="card">
            <table>
                <tr>
                    <th>Parameter</th>
                    <th>Min</th>
                    <th>Max</th>
                    <th>Baseline</th>
                    <th>Goal</th>
                </tr>
                <tr>
                    <td>Comfort Setpoint</td>
                    <td>19.0°C</td>
                    <td>22.0°C</td>
                    <td>20.2°C</td>
                    <td>Estimate effect on T_weighted</td>
                </tr>
                <tr>
                    <td>Eco Setpoint</td>
                    <td>14.0°C</td>
                    <td>19.0°C</td>
                    <td>18.0°C</td>
                    <td>Test if aggressive setback helps</td>
                </tr>
                <tr>
                    <td>Curve Rise</td>
                    <td>0.80</td>
                    <td>1.20</td>
                    <td>0.97</td>
                    <td>Estimate COP sensitivity</td>
                </tr>
                <tr>
                    <td>Comfort Hours</td>
                    <td>8h</td>
                    <td>16h</td>
                    <td>13.5h</td>
                    <td>Test schedule flexibility</td>
                </tr>
            </table>
        </div>

        <h2>3. Block Schedule</h2>
'''

    for _, row in df.iterrows():
        # Add T_HK2 values if available
        thk2_html = ""
        if 'T_HK2_comfort' in row and pd.notna(row.get('T_HK2_comfort')):
            thk2_html = f'''
                <div class="param-item" style="background: #dbeafe;">
                    <div class="param-label">T_HK2 Comfort</div>
                    <div class="param-value">{row['T_HK2_comfort']:.1f}°C</div>
                </div>
                <div class="param-item" style="background: #dbeafe;">
                    <div class="param-label">T_HK2 Eco</div>
                    <div class="param-value">{row['T_HK2_eco']:.1f}°C</div>
                </div>'''

        html += f'''
        <div class="block-card">
            <div class="block-header">
                <span class="block-num">Block {row['block']} (Design Point {row['design_point']})</span>
                <span class="block-dates">{row['start_date']} to {row['end_date']}</span>
            </div>
            <div class="param-grid">{thk2_html}
                <div class="param-item">
                    <div class="param-label">Comfort Setpoint</div>
                    <div class="param-value">{row['comfort_setpoint']}°C</div>
                </div>
                <div class="param-item">
                    <div class="param-label">Eco Setpoint</div>
                    <div class="param-value">{row['eco_setpoint']}°C</div>
                </div>
                <div class="param-item">
                    <div class="param-label">Curve Rise</div>
                    <div class="param-value">{row['curve_rise']}</div>
                </div>
                <div class="param-item">
                    <div class="param-label">Schedule</div>
                    <div class="param-value">{row['comfort_start']} - {row['comfort_end']}</div>
                </div>
                <div class="param-item">
                    <div class="param-label">Comfort Hours</div>
                    <div class="param-value">{row['comfort_hours']:.0f}h</div>
                </div>
                <div class="param-item">
                    <div class="param-label">Duration</div>
                    <div class="param-value">7 days</div>
                </div>
            </div>
        </div>
'''

    html += '''
        <h2>4. Daily Checklist</h2>

        <div class="card">
            <h3>At Block Start (Day 1 of each block)</h3>
            <ul style="margin: 0.5rem 0 0 1.5rem;">
                <li>Update heat pump comfort schedule (start/end times)</li>
                <li>Update Home Assistant climate setpoints (comfort/eco)</li>
                <li>Update heating curve rise (Steilheit) on heat pump</li>
                <li>Record parameter change timestamp</li>
                <li>Note current outdoor temperature and weather</li>
            </ul>
        </div>

        <div class="card">
            <h3>During Measurement (Days 3-7)</h3>
            <ul style="margin: 0.5rem 0 0 1.5rem;">
                <li>Check T_weighted stays above 17.0°C</li>
                <li>Monitor COP remains reasonable (>2.0)</li>
                <li>Note any occupancy changes or unusual events</li>
            </ul>
        </div>

        <div class="danger">
            <strong>Abort Criteria</strong>
            <p>Revert to baseline settings immediately if:</p>
            <ul style="margin: 0.5rem 0 0 1rem;">
                <li>T_weighted drops below 16.5°C for more than 2 hours</li>
                <li>COP remains below 2.0 for more than 24 hours</li>
                <li>Heat pump shows error codes</li>
            </ul>
            <p style="margin-top: 0.5rem;"><strong>Baseline:</strong> comfort=20.2°C, eco=18.0°C, curve=0.97, schedule=06:30-20:00</p>
        </div>

        <h2>5. Parameter Change Locations</h2>

        <div class="card">
            <table>
                <tr>
                    <th>Parameter</th>
                    <th>Location</th>
                    <th>Interface</th>
                </tr>
                <tr>
                    <td>Comfort Schedule (Start/End)</td>
                    <td>Heat pump scheduler</td>
                    <td>Heat pump control panel</td>
                </tr>
                <tr>
                    <td>Setpoints (Comfort/Eco)</td>
                    <td>Climate entity</td>
                    <td>Home Assistant</td>
                </tr>
                <tr>
                    <td>Curve Rise (Steilheit)</td>
                    <td>Heating curve menu</td>
                    <td>Heat pump control panel</td>
                </tr>
            </table>
        </div>

        <h2>6. Running Analysis</h2>

        <div class="card">
            <h3>After Collecting Data</h3>
            <p>Run analysis after each completed block or at the end of the pilot:</p>
            <pre style="background: #1e293b; color: #e2e8f0; padding: 1rem; border-radius: 8px; overflow-x: auto;">
# Dynamical analysis (preferred) - uses grey-box model on all data
python src/phase5_pilot/run_pilot.py --analyze

# RSM analysis (comparison) - uses block averages with washout
python src/phase5_pilot/run_pilot.py --analyze-rsm

# View results
open output/phase5_pilot/dynamical_analysis_report.html
            </pre>
        </div>

        <div class="card">
            <h3>Key Outputs</h3>
            <table>
                <tr>
                    <th>File</th>
                    <th>Description</th>
                </tr>
                <tr>
                    <td><code>dynamical_model_params.json</code></td>
                    <td>Grey-box model parameters (τ_buf, τ_room, etc.)</td>
                </tr>
                <tr>
                    <td><code>step_response_analysis.csv</code></td>
                    <td>Metrics at each parameter transition</td>
                </tr>
                <tr>
                    <td><code>fig_dynamical_model.png</code></td>
                    <td>Model fit visualization</td>
                </tr>
                <tr>
                    <td><code>fig_step_responses.png</code></td>
                    <td>Step response characteristics</td>
                </tr>
            </table>
        </div>

    </div>
</body>
</html>'''

    with open(output_path, 'w') as f:
        f.write(html)


def save_schedule(df: pd.DataFrame, seed: int) -> None:
    """Save schedule to CSV and JSON formats."""

    # CSV format
    csv_path = OUTPUT_DIR / 'pilot_schedule.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # JSON format
    json_path = OUTPUT_DIR / 'pilot_schedule.json'
    schedule_dict = {
        'generated': datetime.now().isoformat(),
        'seed': seed,
        'n_blocks': len(df),
        'block_days': BLOCK_DAYS,
        'washout_days': WASHOUT_DAYS,
        'measurement_days': MEASUREMENT_DAYS,
        'safety': SAFETY,
        'blocks': df.to_dict(orient='records'),
    }
    with open(json_path, 'w') as f:
        json.dump(schedule_dict, f, indent=2)
    print(f"Saved: {json_path}")

    # HTML protocol
    html_path = OUTPUT_DIR / 'pilot_protocol.html'
    generate_html_protocol(df, seed, html_path)
    print(f"Saved: {html_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate dated pilot schedule from T_HK2-targeted design',
    )
    parser.add_argument(
        '--start',
        type=str,
        default='2026-01-13',
        help='Start date (YYYY-MM-DD), default: 2026-01-13',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for sequence randomization (default: 42)',
    )

    args = parser.parse_args()

    # Parse start date
    try:
        start_date = datetime.strptime(args.start, '%Y-%m-%d')
    except ValueError:
        print(f"Error: Invalid date format '{args.start}'. Use YYYY-MM-DD.")
        return 1

    # Load design
    try:
        design_df = load_design()
    except FileNotFoundError as e:
        print(e)
        return 1

    # Generate schedule
    schedule_df = generate_schedule(design_df, start_date, args.seed)

    # Save
    save_schedule(schedule_df, args.seed)

    # Print preview
    print(f"\nSchedule preview:")
    preview_cols = ['block', 'start_date', 'comfort_setpoint', 'eco_setpoint',
                    'curve_rise', 'comfort_start', 'comfort_end']
    print(schedule_df[preview_cols].head(10).to_string(index=False))

    print("\nDone! Review pilot_schedule.csv and pilot_protocol.html")
    return 0


if __name__ == '__main__':
    exit(main())
