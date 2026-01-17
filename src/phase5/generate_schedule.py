#!/usr/bin/env python3
"""
Phase 5: Generate Randomization Schedule

Generates a balanced randomized block schedule for the intervention study.
Uses a constrained randomization approach to ensure:
- Each strategy appears ~equally across the study
- Strategies are balanced across early/mid/late winter
- No strategy follows itself (carryover balance)

Usage:
    python src/phase5/generate_schedule.py --start 2027-11-01 --weeks 20 --seed 42
    python src/phase5/generate_schedule.py --help

Outputs:
    output/phase5/block_schedule.csv - Complete block schedule
    output/phase5/block_schedule.json - Machine-readable format
    output/phase5/experimental_protocol.html - HTML report for study execution
"""

import argparse
import json
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'output' / 'phase5'

# Add src to path for shared imports
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
from shared.report_style import CSS, COLORS
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Strategy definitions (3 strategies from Pareto optimization - Jan 2026)
# See docs/phase5_experimental_design.md Sections 3.1-3.3 for rationale
STRATEGIES = {
    'A': {
        'name': 'Baseline',
        'comfort_start': '06:30',
        'comfort_end': '20:00',
        'setpoint_comfort': 20.2,
        'setpoint_eco': 18.5,
        'curve_rise': 1.08,
    },
    'B': {
        'name': 'Grid-Minimal',
        'comfort_start': '09:00',
        'comfort_end': '16:00',
        'setpoint_comfort': 22.0,
        'setpoint_eco': 12.0,
        'curve_rise': 0.83,
    },
    'C': {
        'name': 'Balanced',
        'comfort_start': '10:00',
        'comfort_end': '16:00',
        'setpoint_comfort': 22.0,
        'setpoint_eco': 13.1,
        'curve_rise': 0.83,
    },
}

# Block duration in days (from power analysis: 3-day washout + 4-day measurement)
# τ_effort = 12.4h (weighted avg) → 3×τ = 37h ≈ 1.5 days → rounded to 3 days for margin
# 7-day blocks enable weekly parameter changes with 97% power (vs 75% for 4-day blocks)
BLOCK_DAYS = 7
WASHOUT_DAYS = 3
MEASUREMENT_DAYS = 4


def get_season(date: datetime, study_start: datetime) -> str:
    """Determine season tercile based on study progress."""
    days_elapsed = (date - study_start).days
    total_days = 140  # 20 weeks = 20 blocks of 7 days

    if days_elapsed < total_days / 3:
        return 'early'
    elif days_elapsed < 2 * total_days / 3:
        return 'mid'
    else:
        return 'late'


def generate_balanced_sequence(n_blocks: int, seed: int) -> list:
    """
    Generate a balanced sequence of strategies.

    Ensures:
    - Approximately equal representation of each strategy
    - No consecutive repeats
    - Reasonably balanced across study phases
    """
    random.seed(seed)
    strategies = list(STRATEGIES.keys())
    n_strategies = len(strategies)

    # Calculate how many complete sets we need
    complete_sets = n_blocks // n_strategies
    remainder = n_blocks % n_strategies

    # Build base sequence with complete balanced sets
    sequence = []
    for _ in range(complete_sets):
        shuffled = strategies.copy()
        random.shuffle(shuffled)

        # Ensure no consecutive repeat with previous block
        if sequence and shuffled[0] == sequence[-1]:
            # Swap first element with another
            swap_idx = random.randint(1, n_strategies - 1)
            shuffled[0], shuffled[swap_idx] = shuffled[swap_idx], shuffled[0]

        sequence.extend(shuffled)

    # Add remainder blocks
    if remainder > 0:
        extra = strategies.copy()
        random.shuffle(extra)

        # Avoid consecutive repeat
        if sequence and extra[0] == sequence[-1]:
            swap_idx = random.randint(1, n_strategies - 1)
            extra[0], extra[swap_idx] = extra[swap_idx], extra[0]

        sequence.extend(extra[:remainder])

    return sequence


def generate_schedule(
    start_date: datetime,
    n_weeks: int,
    seed: int,
) -> pd.DataFrame:
    """Generate the complete block schedule."""

    # Calculate number of blocks
    total_days = n_weeks * 7
    n_blocks = total_days // BLOCK_DAYS

    print(f"Generating schedule:")
    print(f"  Start date: {start_date.strftime('%Y-%m-%d')}")
    print(f"  Duration: {n_weeks} weeks ({total_days} days)")
    print(f"  Block length: {BLOCK_DAYS} days")
    print(f"  Total blocks: {n_blocks}")
    print(f"  Random seed: {seed}")

    # Generate balanced sequence
    strategy_sequence = generate_balanced_sequence(n_blocks, seed)

    # Build schedule DataFrame
    records = []
    current_date = start_date

    for block_num, strategy_code in enumerate(strategy_sequence, 1):
        block_start = current_date
        block_end = current_date + timedelta(days=BLOCK_DAYS - 1)
        washout_end = current_date + timedelta(days=WASHOUT_DAYS - 1)
        measurement_start = current_date + timedelta(days=WASHOUT_DAYS)
        season = get_season(block_start, start_date)
        strategy = STRATEGIES[strategy_code]

        records.append({
            'block': block_num,
            'start_date': block_start.strftime('%Y-%m-%d'),
            'end_date': block_end.strftime('%Y-%m-%d'),
            'washout_end': washout_end.strftime('%Y-%m-%d'),
            'measurement_start': measurement_start.strftime('%Y-%m-%d'),
            'strategy_code': strategy_code,
            'strategy_name': strategy['name'],
            'season': season,
            'comfort_start': strategy['comfort_start'],
            'comfort_end': strategy['comfort_end'],
            'setpoint_comfort': strategy['setpoint_comfort'],
            'setpoint_eco': strategy['setpoint_eco'],
            'curve_rise': strategy['curve_rise'],
            'notes': '',
        })

        current_date = block_end + timedelta(days=1)

    df = pd.DataFrame(records)

    # Print summary statistics
    print(f"\nStrategy distribution:")
    for code, name in [(k, v['name']) for k, v in STRATEGIES.items()]:
        count = (df['strategy_code'] == code).sum()
        print(f"  {code} ({name}): {count} blocks")

    print(f"\nSeason distribution:")
    for season in ['early', 'mid', 'late']:
        subset = df[df['season'] == season]
        print(f"  {season.capitalize()}: {len(subset)} blocks")
        for code in STRATEGIES.keys():
            count = (subset['strategy_code'] == code).sum()
            print(f"    {code}: {count}")

    return df


# Strategy colors for visualization
STRATEGY_COLORS = {
    'A': '#6366f1',  # Indigo (Baseline)
    'B': '#22c55e',  # Green (Energy-Optimized)
    'C': '#ec4899',  # Pink (Cost-Optimized)
}


def generate_html_report(
    df: pd.DataFrame,
    schedule_dict: dict,
    seed: int,
    output_path: Path,
) -> None:
    """Generate HTML experimental protocol report."""

    start_date = df['start_date'].iloc[0]
    end_date = df['end_date'].iloc[-1]
    n_blocks = len(df)

    # Count strategies by season
    season_counts = {}
    for season in ['early', 'mid', 'late']:
        subset = df[df['season'] == season]
        season_counts[season] = {
            code: (subset['strategy_code'] == code).sum()
            for code in STRATEGIES.keys()
        }

    # Phase 5-specific CSS extensions
    extra_css = f"""
        :root {{
            --strategy-a: {COLORS['primary_dark_blue']};
            --strategy-b: {COLORS['primary_green']};
            --strategy-c: {COLORS['purple']};
        }}
        .stat-box {{
            background: white;
            border-radius: 4px;
            padding: 1rem 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .stat-box .value {{
            font-size: 1.75rem;
            font-weight: bold;
            color: {COLORS['primary_green']};
        }}
        .stat-box .label {{
            color: {COLORS['gray_dark']};
            font-size: 0.875rem;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }}
        .card {{
            background: white;
            border-radius: 4px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .meta {{
            color: {COLORS['gray_dark']};
            margin-bottom: 2rem;
        }}
        .toc {{
            background: white;
            padding: 1.5rem;
            border-radius: 4px;
            margin-bottom: 2rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .toc ul {{
            list-style: none;
            padding-left: 0;
            margin: 0;
        }}
        .toc li {{
            margin: 0.5rem 0;
        }}
        .toc a {{
            color: {COLORS['primary_green']};
            text-decoration: none;
        }}
        .toc a:hover {{
            text-decoration: underline;
        }}
        .strategy-badge {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-weight: 600;
            font-size: 0.875rem;
            color: white;
        }}
        .strategy-a {{ background: var(--strategy-a); }}
        .strategy-b {{ background: var(--strategy-b); }}
        .strategy-c {{ background: var(--strategy-c); }}
        .season-early {{ background: #fef3c7; }}
        .season-mid {{ background: {COLORS['light_green']}; }}
        .season-late {{ background: #d1fae5; }}
        .block-row {{ transition: background-color 0.2s; }}
        .block-row:hover {{ background: {COLORS['gray_light']}; }}
        .washout {{ color: {COLORS['gray_dark']}; font-size: 0.85rem; }}
        .measurement {{ font-weight: 600; }}
        .checklist {{
            background: #fffbeb;
            border: 1px solid #fcd34d;
            border-radius: 4px;
            padding: 1rem 1.5rem;
            margin: 1rem 0;
        }}
        .checklist li {{ margin: 0.5rem 0; }}
        .param-table {{ font-size: 0.9rem; }}
        .param-table td:first-child {{ font-weight: 500; width: 40%; }}
        .timeline {{
            display: flex;
            flex-wrap: wrap;
            gap: 4px;
            margin: 1rem 0;
        }}
        .timeline-block {{
            width: 28px;
            height: 28px;
            border-radius: 4px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.75rem;
            font-weight: 600;
            color: white;
            cursor: pointer;
        }}
        .timeline-block:hover {{ transform: scale(1.1); }}
        .strategy-legend {{
            display: flex;
            gap: 1.5rem;
            flex-wrap: wrap;
            margin: 1rem 0;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        .legend-color {{
            width: 16px;
            height: 16px;
            border-radius: 4px;
        }}
        .print-break {{ page-break-before: always; }}
        @media print {{
            body {{ padding: 0.5rem; }}
            .card {{ box-shadow: none; border: 1px solid {COLORS['gray_border']}; }}
            .no-print {{ display: none; }}
        }}
    """

    html = f'''<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ESTAT Phase 5: Experimental Protocol</title>
    <style>
{CSS}
{extra_css}
    </style>
</head>
<body>
    <div class="container">
        <h1>ESTAT Phase 5: Experimental Protocol</h1>
        <p class="meta">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Seed: {seed}</p>

        <div class="toc no-print">
            <strong>Contents</strong>
            <ul>
                <li><a href="#rationale">1. Study Rationale</a></li>
                <li><a href="#overview">2. Study Overview</a></li>
                <li><a href="#strategies">3. Strategy Definitions</a></li>
                <li><a href="#timeline">4. Visual Timeline</a></li>
                <li><a href="#schedule">5. Complete Block Schedule</a></li>
                <li><a href="#parameters">6. Parameter Change Protocol</a></li>
                <li><a href="#checklist">7. Daily Checklist</a></li>
                <li><a href="#missed">8. Handling Missed Block Changes</a></li>
            </ul>
        </div>

        <h2 id="rationale">1. Study Rationale</h2>

        <div class="card">
            <h4>Why This Study?</h4>
            <p>This intervention study tests whether adjusting heat pump scheduling and heating curve parameters
            can improve energy efficiency (COP) while maintaining thermal comfort. The four strategies represent
            different trade-offs between comfort, energy consumption, solar self-sufficiency, and electricity costs.</p>

            <p style="margin-top: 1rem;">Historical data analysis (Phases 1-4) suggests potential COP improvements of 0.2-0.4 through
            schedule optimization, but these predictions need validation under real operating conditions with
            varying weather, occupancy, and system dynamics.</p>
        </div>

        <div class="card">
            <h4>Crossover Design</h4>
            <p>This study uses a <strong>randomized crossover design</strong> where each strategy is tested multiple
            times throughout the heating season. Key design features:</p>
            <ul style="margin: 1rem 0 0 1.5rem;">
                <li><strong>Within-subject comparison:</strong> The same building tests all strategies, eliminating
                between-building variability</li>
                <li><strong>Seasonal balance:</strong> Each strategy appears in early, mid, and late winter to account
                for changing outdoor temperatures and solar availability</li>
                <li><strong>Randomization:</strong> Block order is randomized to prevent systematic bias from time trends</li>
                <li><strong>Washout periods:</strong> 3-day washout between strategies allows the thermal mass to
                equilibrate to new settings before measurement (based on τ_effort analysis)</li>
                <li><strong>Replication:</strong> ~6-7 blocks per strategy provides 97% statistical power to detect
                +0.30 COP changes</li>
            </ul>
        </div>

        <div class="card">
            <h4>Block Structure</h4>
            <p>Each {BLOCK_DAYS}-day block consists of:</p>
            <ul style="margin: 1rem 0 0 1.5rem;">
                <li><strong>Days 1-{WASHOUT_DAYS} (Washout):</strong> System adjusts to new parameters. Indoor temperatures
                stabilize. Data from this period is excluded from primary analysis.</li>
                <li><strong>Days {WASHOUT_DAYS + 1}-{BLOCK_DAYS} (Measurement):</strong> System operates in steady-state. COP,
                energy consumption, and comfort metrics are recorded for analysis.</li>
            </ul>
            <p style="margin-top: 1rem;">The {WASHOUT_DAYS}-day washout was determined from Phase 3 thermal model analysis showing
            the weighted τ_effort (heating response time) is ~12 hours. Three days provides >99% equilibration with margin for scheduling.</p>
        </div>

        <div class="card">
            <h4>Primary Outcomes</h4>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Definition</th>
                    <th>Expected Range</th>
                </tr>
                <tr>
                    <td>COP (Coefficient of Performance)</td>
                    <td>Heat delivered / Electricity consumed</td>
                    <td>3.5 - 5.0</td>
                </tr>
                <tr>
                    <td>Daily heating electricity</td>
                    <td>kWh consumed by heat pump per day</td>
                    <td>15 - 50 kWh</td>
                </tr>
                <tr>
                    <td>Comfort compliance</td>
                    <td>% of occupied hours (08:00-22:00) within 18-23°C</td>
                    <td>>95%</td>
                </tr>
                <tr>
                    <td>Solar self-sufficiency</td>
                    <td>% of heating energy from PV</td>
                    <td>20 - 60%</td>
                </tr>
            </table>
        </div>

        <h2 id="overview">2. Study Overview</h2>

        <div class="grid">
            <div class="stat-box">
                <div class="value">{n_blocks}</div>
                <div class="label">Total Blocks</div>
            </div>
            <div class="stat-box">
                <div class="value">{BLOCK_DAYS}</div>
                <div class="label">Days per Block</div>
            </div>
            <div class="stat-box">
                <div class="value">{WASHOUT_DAYS}</div>
                <div class="label">Washout Days</div>
            </div>
            <div class="stat-box">
                <div class="value">{MEASUREMENT_DAYS}</div>
                <div class="label">Measurement Days</div>
            </div>
        </div>

        <div class="card">
            <h4>Study Period</h4>
            <table>
                <tr><td>Start Date</td><td><strong>{start_date}</strong></td></tr>
                <tr><td>End Date</td><td><strong>{end_date}</strong></td></tr>
                <tr><td>Duration</td><td>{n_blocks * BLOCK_DAYS} days ({n_blocks * BLOCK_DAYS // 7} weeks)</td></tr>
                <tr><td>Random Seed</td><td><code>{seed}</code></td></tr>
            </table>
        </div>

        <div class="card">
            <h4>Strategy Distribution by Season</h4>
            <table>
                <tr>
                    <th>Season</th>
                    <th>A (Baseline)</th>
                    <th>B (Grid-Min)</th>
                    <th>C (Balanced)</th>
                    <th>Total</th>
                </tr>'''

    for season in ['early', 'mid', 'late']:
        counts = season_counts[season]
        total = sum(counts.values())
        html += f'''
                <tr class="season-{season}">
                    <td><strong>{season.capitalize()}</strong></td>
                    <td>{counts['A']}</td>
                    <td>{counts['B']}</td>
                    <td>{counts['C']}</td>
                    <td><strong>{total}</strong></td>
                </tr>'''

    # Totals row
    totals = {code: sum(season_counts[s][code] for s in ['early', 'mid', 'late']) for code in STRATEGIES.keys()}
    html += f'''
                <tr>
                    <td><strong>Total</strong></td>
                    <td><strong>{totals['A']}</strong></td>
                    <td><strong>{totals['B']}</strong></td>
                    <td><strong>{totals['C']}</strong></td>
                    <td><strong>{n_blocks}</strong></td>
                </tr>
            </table>
        </div>

        <h2 id="strategies">3. Strategy Definitions</h2>

        <div class="strategy-legend">
            <div class="legend-item">
                <div class="legend-color" style="background: var(--strategy-a);"></div>
                <span>A: Baseline</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: var(--strategy-b);"></div>
                <span>B: Grid-Minimal</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: var(--strategy-c);"></div>
                <span>C: Balanced</span>
            </div>
        </div>

        <div class="card">
            <table>
                <tr>
                    <th>Parameter</th>
                    <th><span class="strategy-badge strategy-a">A</span> Baseline</th>
                    <th><span class="strategy-badge strategy-b">B</span> Grid-Min</th>
                    <th><span class="strategy-badge strategy-c">C</span> Balanced</th>
                </tr>
                <tr>
                    <td>Comfort Start</td>
                    <td>{STRATEGIES['A']['comfort_start']}</td>
                    <td>{STRATEGIES['B']['comfort_start']}</td>
                    <td>{STRATEGIES['C']['comfort_start']}</td>
                </tr>
                <tr>
                    <td>Comfort End</td>
                    <td>{STRATEGIES['A']['comfort_end']}</td>
                    <td>{STRATEGIES['B']['comfort_end']}</td>
                    <td>{STRATEGIES['C']['comfort_end']}</td>
                </tr>
                <tr>
                    <td>Setpoint Comfort</td>
                    <td>{STRATEGIES['A']['setpoint_comfort']}°C</td>
                    <td>{STRATEGIES['B']['setpoint_comfort']}°C</td>
                    <td>{STRATEGIES['C']['setpoint_comfort']}°C</td>
                </tr>
                <tr>
                    <td>Setpoint Eco</td>
                    <td>{STRATEGIES['A']['setpoint_eco']}°C</td>
                    <td>{STRATEGIES['B']['setpoint_eco']}°C</td>
                    <td>{STRATEGIES['C']['setpoint_eco']}°C</td>
                </tr>
                <tr>
                    <td>Curve Rise (Steilheit)</td>
                    <td>{STRATEGIES['A']['curve_rise']}</td>
                    <td>{STRATEGIES['B']['curve_rise']}</td>
                    <td>{STRATEGIES['C']['curve_rise']}</td>
                </tr>
            </table>
        </div>

        <h2 id="timeline">4. Visual Timeline</h2>

        <div class="card">
            <p>Each square represents one block. Hover for details.</p>
            <div class="timeline">'''

    for _, row in df.iterrows():
        code = row['strategy_code']
        color = STRATEGY_COLORS[code]
        title = f"Block {row['block']}: {row['strategy_name']} ({row['start_date']} to {row['end_date']})"
        html += f'''
                <div class="timeline-block" style="background: {color};" title="{title}">{row['block']}</div>'''

    html += '''
            </div>
            <div class="strategy-legend" style="margin-top: 1rem;">
                <div class="legend-item">
                    <div class="legend-color season-early"></div>
                    <span>Early Winter</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color season-mid"></div>
                    <span>Mid Winter</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color season-late"></div>
                    <span>Late Winter</span>
                </div>
            </div>
        </div>

        <h2 id="schedule">5. Complete Block Schedule</h2>

        <div class="card">
            <table>
                <tr>
                    <th>#</th>
                    <th>Strategy</th>
                    <th>Block Period</th>
                    <th>Washout</th>
                    <th>Measurement</th>
                    <th>Season</th>
                </tr>'''

    for _, row in df.iterrows():
        code = row['strategy_code'].lower()
        season = row['season']
        html += f'''
                <tr class="block-row season-{season}">
                    <td><strong>{row['block']}</strong></td>
                    <td><span class="strategy-badge strategy-{code}">{row['strategy_code']}</span> {row['strategy_name']}</td>
                    <td>{row['start_date']} to {row['end_date']}</td>
                    <td class="washout">{row['start_date']} to {row['washout_end']}</td>
                    <td class="measurement">{row['measurement_start']} to {row['end_date']}</td>
                    <td>{season.capitalize()}</td>
                </tr>'''

    html += '''
            </table>
        </div>

        <h2 id="parameters" class="print-break">6. Parameter Change Protocol</h2>

        <p>At the start of each block, adjust the following parameters on the heat pump:</p>
'''

    # Generate parameter cards for each block
    for _, row in df.iterrows():
        code = row['strategy_code'].lower()
        html += f'''
        <div class="card">
            <h4>Block {row['block']}: <span class="strategy-badge strategy-{code}">{row['strategy_code']}</span> {row['strategy_name']}</h4>
            <p><strong>Start Date:</strong> {row['start_date']} | <strong>End Date:</strong> {row['end_date']}</p>
            <table class="param-table">
                <tr><td>Schedule: Comfort Start</td><td><strong>{row['comfort_start']}</strong></td></tr>
                <tr><td>Schedule: Comfort End</td><td><strong>{row['comfort_end']}</strong></td></tr>
                <tr><td>Climate: Setpoint Comfort</td><td><strong>{row['setpoint_comfort']}°C</strong></td></tr>
                <tr><td>Climate: Setpoint Eco</td><td><strong>{row['setpoint_eco']}°C</strong></td></tr>
                <tr><td>Heating Curve: Rise (Steilheit)</td><td><strong>{row['curve_rise']}</strong></td></tr>
            </table>
        </div>'''

    html += '''

        <h2 id="checklist">7. Daily Checklist</h2>

        <div class="checklist">
            <h4>At Block Start (Day 1)</h4>
            <ul>
                <li>Verify current block number and strategy from schedule</li>
                <li>Update heat pump comfort schedule (start/end times)</li>
                <li>Update Home Assistant climate setpoints (comfort/eco)</li>
                <li>Update heating curve rise (Steilheit) on heat pump</li>
                <li>Record parameter change timestamp in daily log</li>
                <li>Note any unusual conditions (weather events, occupancy changes)</li>
            </ul>
        </div>

        <div class="checklist">
            <h4>During Washout (Days 1-{WASHOUT_DAYS})</h4>
            <ul>
                <li>Monitor indoor temperatures stabilizing to new settings</li>
                <li>Check heat pump is responding to new schedule</li>
                <li>Note any error conditions or anomalies</li>
            </ul>
        </div>

        <div class="checklist">
            <h4>During Measurement (Days {WASHOUT_DAYS + 1}-{BLOCK_DAYS})</h4>
            <ul>
                <li>Verify system is operating in steady-state</li>
                <li>Check data logging is active (Home Assistant, InfluxDB)</li>
                <li>Record daily COP and energy consumption (optional manual check)</li>
                <li>Note weather conditions and any disruptions</li>
            </ul>
        </div>

        <div class="checklist">
            <h4>At Block End (Day {BLOCK_DAYS})</h4>
            <ul>
                <li>Complete block summary entry</li>
                <li>Export any manual observations</li>
                <li>Prepare for next block's parameter changes</li>
            </ul>
        </div>

        <div class="card">
            <h4>Parameter Change Locations</h4>
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

        <h2 id="missed">8. Handling Missed Block Changes</h2>

        <div class="card">
            <h4>What Constitutes a "Missed" Block Change?</h4>
            <p>A block change is considered missed if parameters were not updated within <strong>24 hours</strong>
            of the scheduled block start date. Minor delays (a few hours) do not invalidate a block, but
            significantly late changes affect the washout period and may compromise measurement validity.</p>
        </div>

        <div class="card" style="border-left: 4px solid var(--warning);">
            <h4>Scenario 1: Change Delayed by 1 Day</h4>
            <p><strong>Situation:</strong> Parameters changed on Day 2 instead of Day 1.</p>
            <p><strong>Impact:</strong> Washout period is shortened by 1 day.</p>
            <p><strong>Action:</strong></p>
            <ul style="margin: 0.5rem 0 0 1.5rem;">
                <li>Continue with the block as scheduled</li>
                <li>Document the delay in the block notes</li>
                <li>Mark the block as "shortened washout" in the daily log</li>
                <li>Include in analysis with a sensitivity flag</li>
            </ul>
        </div>

        <div class="card" style="border-left: 4px solid var(--danger);">
            <h4>Scenario 2: Change Delayed by {WASHOUT_DAYS}+ Days</h4>
            <p><strong>Situation:</strong> Parameters changed on Day {WASHOUT_DAYS + 1} or later.</p>
            <p><strong>Impact:</strong> Washout period is severely compromised or eliminated.</p>
            <p><strong>Action:</strong></p>
            <ul style="margin: 0.5rem 0 0 1.5rem;">
                <li>Implement the change immediately when discovered</li>
                <li>Document the delay with exact timestamps</li>
                <li>Mark the block as <strong>"excluded"</strong> in the block summary</li>
                <li>Do NOT attempt to extend the block or shift subsequent blocks</li>
                <li>Continue with the next scheduled block on its original date</li>
            </ul>
        </div>

        <div class="card" style="border-left: 4px solid var(--danger);">
            <h4>Scenario 3: Entire Block Missed</h4>
            <p><strong>Situation:</strong> No parameter change was made during the entire block period.</p>
            <p><strong>Impact:</strong> Block continues previous strategy settings.</p>
            <p><strong>Action:</strong></p>
            <ul style="margin: 0.5rem 0 0 1.5rem;">
                <li>Mark the block as <strong>"missed"</strong> in the block summary</li>
                <li>Document what settings were actually in effect</li>
                <li>In analysis, this block's data will be attributed to whatever strategy was actually running</li>
                <li>Begin the next block on schedule with correct parameters</li>
            </ul>
        </div>

        <div class="card" style="border-left: 4px solid var(--primary);">
            <h4>Scenario 4: Wrong Strategy Implemented</h4>
            <p><strong>Situation:</strong> Parameters were changed, but to the wrong strategy.</p>
            <p><strong>Impact:</strong> Block data is misattributed.</p>
            <p><strong>Action:</strong></p>
            <ul style="margin: 0.5rem 0 0 1.5rem;">
                <li>If discovered during washout (Days 1-{WASHOUT_DAYS}): Correct immediately and note the error</li>
                <li>If discovered during measurement (Days {WASHOUT_DAYS + 1}-{BLOCK_DAYS}): Complete the block with current settings</li>
                <li>Document which strategy was actually implemented</li>
                <li>In analysis, attribute data to the <em>actual</em> strategy that was running</li>
            </ul>
        </div>

        <div class="card">
            <h4>Key Principles</h4>
            <ul style="margin: 0.5rem 0 0 1.5rem;">
                <li><strong>Never shift the schedule:</strong> Subsequent blocks always start on their original dates.
                Shifting creates cascading problems and breaks seasonal balance.</li>
                <li><strong>Document everything:</strong> Detailed notes allow proper handling during analysis.
                A well-documented deviation is better than an undocumented one.</li>
                <li><strong>Intention-to-treat:</strong> The primary analysis will use the <em>intended</em> schedule.
                Sensitivity analyses will examine actual-strategy effects.</li>
                <li><strong>No makeup blocks:</strong> Do not add extra blocks to compensate for missed ones.
                The design has sufficient replication to handle occasional missing data.</li>
            </ul>
        </div>

        <div class="card">
            <h4>Recording Deviations</h4>
            <p>For any deviation from the protocol, record in the block summary:</p>
            <table>
                <tr>
                    <th>Field</th>
                    <th>Example</th>
                </tr>
                <tr>
                    <td>Deviation type</td>
                    <td>Delayed change / Missed block / Wrong strategy</td>
                </tr>
                <tr>
                    <td>Scheduled change date</td>
                    <td>2027-11-16</td>
                </tr>
                <tr>
                    <td>Actual change date</td>
                    <td>2027-11-17 14:30</td>
                </tr>
                <tr>
                    <td>Intended strategy</td>
                    <td>A (Baseline)</td>
                </tr>
                <tr>
                    <td>Actual strategy</td>
                    <td>Previous block's strategy (C)</td>
                </tr>
                <tr>
                    <td>Reason for deviation</td>
                    <td>Travel / forgot / technical issue</td>
                </tr>
                <tr>
                    <td>Block status</td>
                    <td>Valid / Shortened washout / Excluded / Missed</td>
                </tr>
            </table>
        </div>

        <div class="footer">
            <p>ESTAT - Energy System Analysis | Phase 5: Intervention Study</p>
        </div>
    </div>
</body>
</html>'''

    with open(output_path, 'w') as f:
        f.write(html)


def save_schedule(df: pd.DataFrame, seed: int) -> None:
    """Save schedule to CSV and JSON formats."""

    # CSV format
    csv_path = OUTPUT_DIR / 'block_schedule.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # JSON format (for programmatic access)
    json_path = OUTPUT_DIR / 'block_schedule.json'
    schedule_dict = {
        'generated': datetime.now().isoformat(),
        'n_blocks': len(df),
        'block_days': BLOCK_DAYS,
        'strategies': STRATEGIES,
        'blocks': df.to_dict(orient='records'),
    }
    with open(json_path, 'w') as f:
        json.dump(schedule_dict, f, indent=2)
    print(f"Saved: {json_path}")

    # HTML report
    html_path = OUTPUT_DIR / 'experimental_protocol.html'
    generate_html_report(df, schedule_dict, seed, html_path)
    print(f"Saved: {html_path}")

    # Print first few blocks as preview
    print(f"\nSchedule preview (first 8 blocks):")
    preview = df[['block', 'start_date', 'end_date', 'strategy_code', 'strategy_name', 'season']].head(8)
    print(preview.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description='Generate randomized block schedule for Phase 5 intervention study',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python src/phase5/generate_schedule.py --start 2027-11-01 --weeks 20 --seed 42
    python src/phase5/generate_schedule.py --start 2027-11-01 --weeks 16
        """,
    )
    parser.add_argument(
        '--start',
        type=str,
        default='2027-11-01',
        help='Study start date (YYYY-MM-DD), default: 2027-11-01',
    )
    parser.add_argument(
        '--weeks',
        type=int,
        default=20,
        help='Study duration in weeks, default: 20',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility (default: random)',
    )

    args = parser.parse_args()

    # Parse start date
    try:
        start_date = datetime.strptime(args.start, '%Y-%m-%d')
    except ValueError:
        print(f"Error: Invalid date format '{args.start}'. Use YYYY-MM-DD.")
        return 1

    # Use random seed if not specified
    seed = args.seed if args.seed is not None else random.randint(1, 99999)

    # Generate and save schedule
    df = generate_schedule(start_date, args.weeks, seed)
    save_schedule(df, seed)

    print("\nDone! Review the schedule and adjust manually if needed.")
    return 0


if __name__ == '__main__':
    exit(main())
