#!/usr/bin/env python3
"""
Phase 1: Data Preprocessing Pipeline

This script runs all preprocessing steps and generates an HTML report
documenting the data cleaning, transformations, and quality checks.

Steps:
1. Energy balance preprocessing (daily, monthly, yearly CSV files)
2. Sensor data preprocessing (InfluxDB export from Home Assistant)
3. Data integration (merge energy and sensor data)
4. Tariff preprocessing (electricity purchase and feed-in rates)

Output:
- Processed parquet files in output/phase1/
- HTML report: output/phase1/phase1_report.html
"""

import subprocess
import sys
import io
import re
from pathlib import Path
from datetime import datetime
from contextlib import redirect_stdout, redirect_stderr
import pandas as pd

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "output" / "phase1"
SRC_DIR = PROJECT_ROOT / "src" / "phase1"

# Add src to path for shared imports
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
from shared.report_style import CSS, COLORS

# Store logs from each step
step_logs = {}


def run_step(step_name: str, script_path: Path) -> tuple[bool, str]:
    """Run a preprocessing step and capture its output."""
    print(f"\n{'='*60}")
    print(f"Running: {step_name}")
    print(f"Script: {script_path.name}")
    print('='*60)

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=1200  # 20 minute timeout
        )

        output = result.stdout
        if result.stderr:
            output += "\n\nSTDERR:\n" + result.stderr

        # Print output to console
        print(output)

        success = result.returncode == 0
        return success, output

    except subprocess.TimeoutExpired:
        error_msg = f"ERROR: {step_name} timed out after 20 minutes"
        print(error_msg)
        return False, error_msg
    except Exception as e:
        error_msg = f"ERROR: {step_name} failed with exception: {e}"
        print(error_msg)
        return False, error_msg


def load_csv_as_html_table(filepath: Path, max_rows: int | None = 50) -> str:
    """Load a CSV file and return as HTML table.

    Args:
        filepath: Path to CSV file
        max_rows: Maximum rows to display, or None for all rows
    """
    if not filepath.exists():
        return f"<p class='error'>File not found: {filepath.name}</p>"

    try:
        df = pd.read_csv(filepath)
        if max_rows is not None and len(df) > max_rows:
            df_display = df.head(max_rows)
            truncated = f"<p class='note'>Showing first {max_rows} of {len(df)} rows</p>"
        else:
            df_display = df
            truncated = ""

        return truncated + df_display.to_html(classes='data-table', index=False)
    except Exception as e:
        return f"<p class='error'>Error loading {filepath.name}: {e}</p>"


def load_text_file(filepath: Path) -> str:
    """Load a text file and return contents."""
    if not filepath.exists():
        return f"File not found: {filepath.name}"

    try:
        return filepath.read_text()
    except Exception as e:
        return f"Error loading {filepath.name}: {e}"


def extract_stats_from_log(log: str) -> dict:
    """Extract key statistics from preprocessing log output."""
    stats = {}

    # Energy balance stats
    if match := re.search(r'Daily data: (\d+) records', log):
        stats['daily_records'] = int(match.group(1))
    if match := re.search(r'Monthly data: (\d+) records', log):
        stats['monthly_records'] = int(match.group(1))
    if match := re.search(r'Yearly data: (\d+) records', log):
        stats['yearly_records'] = int(match.group(1))
    if match := re.search(r'Corrected (\d+) threshold violations', log):
        stats['threshold_corrections'] = int(match.group(1))
    if match := re.search(r'match rate.*?(\d+\.?\d*)%', log):
        stats['validation_match_rate'] = float(match.group(1))

    # Sensor stats
    if match := re.search(r'heating: ([\d,]+) records, (\d+) sensors', log):
        stats['heating_records'] = int(match.group(1).replace(',', ''))
        stats['heating_sensors'] = int(match.group(2))
    if match := re.search(r'weather: ([\d,]+) records, (\d+) sensors', log):
        stats['weather_records'] = int(match.group(1).replace(',', ''))
        stats['weather_sensors'] = int(match.group(2))
    if match := re.search(r'rooms: ([\d,]+) records, (\d+) sensors', log):
        stats['rooms_records'] = int(match.group(1).replace(',', ''))
        stats['rooms_sensors'] = int(match.group(2))
    if match := re.search(r'energy: ([\d,]+) records, (\d+) sensors', log):
        stats['energy_records'] = int(match.group(1).replace(',', ''))
        stats['energy_sensors'] = int(match.group(2))
    if match := re.search(r'Removed (\d+) spurious readings', log):
        stats['spurious_readings_removed'] = int(match.group(1))

    # Integration stats
    if match := re.search(r'Merged dataset: (\d+) records, (\d+) columns', log):
        stats['merged_records'] = int(match.group(1))
        stats['merged_columns'] = int(match.group(2))
    if match := re.search(r'Duration: (\d+) days', log):
        stats['overlap_days'] = int(match.group(1))

    # Tariff stats
    if match := re.search(r'Created (\d+) tariff entries', log):
        stats['tariff_entries'] = int(match.group(1))
    if match := re.search(r'High tariff hours: ([\d,]+)', log):
        stats['high_tariff_hours'] = int(match.group(1).replace(',', ''))
    if match := re.search(r'Low tariff hours: ([\d,]+)', log):
        stats['low_tariff_hours'] = int(match.group(1).replace(',', ''))
    if match := re.search(r'Purchase tariffs: (\d+) entries', log):
        stats['purchase_tariff_entries'] = int(match.group(1))
    if match := re.search(r'Feed-in tariffs: (\d+) entries', log):
        stats['feedin_tariff_entries'] = int(match.group(1))

    return stats


def generate_html_report(step_logs: dict, stats: dict) -> str:
    """Generate comprehensive HTML preprocessing report."""

    # Load CSV summaries
    corrections_html = load_csv_as_html_table(PROCESSED_DIR / "corrections_log.csv")
    validation_html = load_csv_as_html_table(PROCESSED_DIR / "validation_results_fuzzy.csv")
    sensor_summary_html = load_csv_as_html_table(PROCESSED_DIR / "sensor_summary.csv", max_rows=None)
    overlap_html = load_csv_as_html_table(PROCESSED_DIR / "data_overlap_summary.csv")
    energy_summary = load_text_file(PROCESSED_DIR / "energy_balance_summary.txt")
    tariff_schedule_html = load_csv_as_html_table(PROCESSED_DIR / "tariff_schedule.csv", max_rows=None)
    tariff_report_section = load_text_file(PROCESSED_DIR / "tariff_report_section.html")

    # Format logs for HTML
    def format_log(log: str) -> str:
        return log.replace('<', '&lt;').replace('>', '&gt;')

    # Phase 1-specific CSS extensions
    extra_css = f"""
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
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
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
        .data-table {{
            font-size: 0.8rem;
        }}
        .data-table th, .data-table td {{
            padding: 0.35rem 0.5rem;
        }}
        .step-status {{
            padding: 0.25rem 0.75rem;
            border-radius: 4px;
            font-size: 0.875rem;
            font-weight: 500;
        }}
        .step-status.success {{
            background: {COLORS['light_green']};
            color: {COLORS['primary_green']};
        }}
        .step-status.failed {{
            background: #fee2e2;
            color: #dc2626;
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
        .meta {{
            color: {COLORS['gray_dark']};
            margin-bottom: 2rem;
        }}
        details {{
            margin: 1rem 0;
        }}
        summary {{
            cursor: pointer;
            padding: 0.5rem;
            background: {COLORS['gray_light']};
            border-radius: 4px;
            font-weight: 500;
        }}
        summary:hover {{
            background: {COLORS['gray_border']};
        }}
        .methodology {{
            background: {COLORS['light_green']};
            border-left: 4px solid {COLORS['primary_green']};
            padding: 1rem 1.5rem;
            margin: 1rem 0;
            border-radius: 0 4px 4px 0;
        }}
        .methodology h4 {{
            margin-top: 0;
            color: {COLORS['dark_teal']};
        }}
        .methodology ul {{
            margin-left: 1.5rem;
        }}
        .methodology li {{
            margin: 0.25rem 0;
        }}
        .success {{
            color: {COLORS['primary_green']};
        }}
        .error {{
            color: #dc2626;
        }}
        .note {{
            color: {COLORS['gray_dark']};
            font-style: italic;
            margin-bottom: 0.5rem;
        }}
    """

    html = f"""<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ESTAT Phase 1: Preprocessing Report</title>
    <style>
{CSS}
{extra_css}
    </style>
</head>
<body>
    <div class="container">
        <h1>ESTAT Phase 1: Data Preprocessing Report</h1>
        <p class="meta">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <div class="toc">
            <strong>Contents</strong>
            <ul>
                <li><a href="#overview">1. Overview</a></li>
                <li><a href="#energy-balance">2. Energy Balance Preprocessing</a></li>
                <li><a href="#sensors">3. Sensor Data Preprocessing</a></li>
                <li><a href="#integration">4. Data Integration</a></li>
                <li><a href="#tariffs">5. Electricity Tariffs</a></li>
                <li><a href="#outputs">6. Output Files</a></li>
                <li><a href="#logs">7. Detailed Logs</a></li>
            </ul>
        </div>

        <h2 id="overview">1. Overview</h2>

        <div class="grid">
            <div class="stat-box">
                <div class="value">{stats.get('daily_records', 'N/A'):,}</div>
                <div class="label">15-min Energy Records</div>
            </div>
            <div class="stat-box">
                <div class="value">{stats.get('heating_sensors', 0) + stats.get('weather_sensors', 0) + stats.get('rooms_sensors', 0) + stats.get('energy_sensors', 0)}</div>
                <div class="label">Total Sensors</div>
            </div>
            <div class="stat-box">
                <div class="value">{stats.get('overlap_days', 'N/A')}</div>
                <div class="label">Days of Overlap</div>
            </div>
            <div class="stat-box">
                <div class="value">{stats.get('merged_columns', 'N/A')}</div>
                <div class="label">Integrated Columns</div>
            </div>
        </div>

        <div class="card">
            <h4>Pipeline Steps</h4>
            <table>
                <thead>
                    <tr><th>Step</th><th>Description</th><th>Status</th></tr>
                </thead>
                <tbody>
                    <tr>
                        <td>1. Energy Balance</td>
                        <td>Parse CSV files, convert units, clean outliers, validate aggregations</td>
                        <td><span class="step-status {'success' if step_logs.get('energy_balance', (False,))[0] else 'failed'}">{'Success' if step_logs.get('energy_balance', (False,))[0] else 'Failed'}</span></td>
                    </tr>
                    <tr>
                        <td>2. Sensor Data</td>
                        <td>Parse InfluxDB export, categorize sensors, convert units, clean spikes</td>
                        <td><span class="step-status {'success' if step_logs.get('sensors', (False,))[0] else 'failed'}">{'Success' if step_logs.get('sensors', (False,))[0] else 'Failed'}</span></td>
                    </tr>
                    <tr>
                        <td>3. Integration</td>
                        <td>Merge energy and sensor data, identify overlap period</td>
                        <td><span class="step-status {'success' if step_logs.get('integration', (False,))[0] else 'failed'}">{'Success' if step_logs.get('integration', (False,))[0] else 'Failed'}</span></td>
                    </tr>
                    <tr>
                        <td>4. Tariffs</td>
                        <td>Preprocess electricity purchase and feed-in tariff rates</td>
                        <td><span class="step-status {'success' if step_logs.get('tariffs', (False,))[0] else 'failed'}">{'Success' if step_logs.get('tariffs', (False,))[0] else 'Failed'}</span></td>
                    </tr>
                </tbody>
            </table>
        </div>

        <h2 id="energy-balance">2. Energy Balance Preprocessing</h2>

        <div class="methodology">
            <h4>Data Cleaning Methodology</h4>
            <ul>
                <li><strong>Source files:</strong> Daily (15-min intervals), Monthly, Yearly CSV exports</li>
                <li><strong>Number format:</strong> European format (comma as decimal separator)</li>
                <li><strong>Unit conversion:</strong> Watts to kWh (W * 0.25h / 1000)</li>
                <li><strong>Threshold filtering:</strong> Values &gt; 20 kW (5 kWh per 15-min) are replaced with interpolated values</li>
                <li><strong>Missing values:</strong> Linear time-based interpolation (limit: 8 consecutive intervals = 2 hours)</li>
                <li><strong>Monthly corrections:</strong> Corrupted monthly values replaced with aggregated daily sums</li>
                <li><strong>Validation:</strong> Daily sums compared against monthly totals (5% tolerance)</li>
            </ul>
        </div>

        <div class="grid">
            <div class="stat-box">
                <div class="value">{stats.get('threshold_corrections', 'N/A')}</div>
                <div class="label">Threshold Corrections</div>
            </div>
            <div class="stat-box">
                <div class="value">{f"{stats['validation_match_rate']:.1f}" if 'validation_match_rate' in stats else 'N/A'}%</div>
                <div class="label">Validation Match Rate</div>
            </div>
        </div>

        <h3>Data Corrections Log</h3>
        <div class="card">
            {corrections_html}
        </div>

        <h3>Validation Mismatches (Daily vs Monthly)</h3>
        <div class="card">
            <p class="note">Showing mismatches with &ge;10% difference AND &ge;1 kWh absolute difference</p>
            {validation_html}
        </div>

        <details>
            <summary>Energy Balance Summary Statistics</summary>
            <pre>{energy_summary}</pre>
        </details>

        <h2 id="sensors">3. Sensor Data Preprocessing</h2>

        <div class="methodology">
            <h4>Data Cleaning Methodology</h4>
            <ul>
                <li><strong>Source:</strong> InfluxDB annotated CSV export from Home Assistant</li>
                <li><strong>Categorization:</strong> Sensors split into heating, weather, rooms, energy</li>
                <li><strong>Unit conversion:</strong> Davis weather station temperatures converted from Fahrenheit to Celsius</li>
                <li><strong>Spike removal:</strong> Cumulative counters (energy totals) cleaned of spurious spikes</li>
                <li><strong>Spike detection:</strong> Readings with increment &gt; 50 kWh are removed (sensor glitches)</li>
            </ul>
        </div>

        <div class="grid">
            <div class="stat-box">
                <div class="value">{stats.get('heating_sensors', 'N/A')}</div>
                <div class="label">Heating Sensors</div>
            </div>
            <div class="stat-box">
                <div class="value">{stats.get('weather_sensors', 'N/A')}</div>
                <div class="label">Weather Sensors</div>
            </div>
            <div class="stat-box">
                <div class="value">{stats.get('rooms_sensors', 'N/A')}</div>
                <div class="label">Room Sensors</div>
            </div>
            <div class="stat-box">
                <div class="value">{stats.get('energy_sensors', 'N/A')}</div>
                <div class="label">Energy Sensors</div>
            </div>
        </div>

        <h3>Sensor Summary</h3>
        <div class="card">
            {sensor_summary_html}
        </div>

        <h2 id="integration">4. Data Integration</h2>

        <div class="methodology">
            <h4>Integration Methodology</h4>
            <ul>
                <li><strong>Join type:</strong> Outer join on timestamp (preserves all data)</li>
                <li><strong>Resampling:</strong> Sensor data resampled to 15-minute intervals</li>
                <li><strong>Overlap period:</strong> Identified for combined analysis</li>
                <li><strong>Output:</strong> Full integrated dataset + overlap-only subset</li>
            </ul>
        </div>

        <h3>Data Overlap Summary</h3>
        <div class="card">
            {overlap_html}
        </div>

        <h2 id="tariffs">5. Electricity Tariffs</h2>

        <div class="methodology">
            <h4>Tariff Data Sources</h4>
            <ul>
                <li><strong>Provider:</strong> Primeo Energie (Münchenstein, Switzerland)</li>
                <li><strong>Purchase rates:</strong> Official Primeo announcements and ElCom data</li>
                <li><strong>Feed-in rates:</strong> Rückliefervergütung including HKN bonus</li>
                <li><strong>Coverage:</strong> 2023-01-01 to 2026-01-31</li>
            </ul>
        </div>

        <div class="grid">
            <div class="stat-box">
                <div class="value">{stats.get('tariff_entries', 'N/A')}</div>
                <div class="label">Tariff Entries</div>
            </div>
            <div class="stat-box">
                <div class="value">{f"{stats['high_tariff_hours']:,}" if 'high_tariff_hours' in stats else 'N/A'}</div>
                <div class="label">High Tariff Hours</div>
            </div>
            <div class="stat-box">
                <div class="value">{f"{stats['low_tariff_hours']:,}" if 'low_tariff_hours' in stats else 'N/A'}</div>
                <div class="label">Low Tariff Hours</div>
            </div>
        </div>

        <h3>Tariff Time Windows</h3>
        <div class="card">
            <table>
                <thead>
                    <tr><th>Tariff</th><th>Time Windows</th></tr>
                </thead>
                <tbody>
                    <tr>
                        <td>High Tariff (Hochtarif)</td>
                        <td>Mon-Fri 06:00-21:00, Sat 06:00-12:00</td>
                    </tr>
                    <tr>
                        <td>Low Tariff (Niedertarif)</td>
                        <td>Mon-Fri 21:00-06:00, Sat 12:00 - Mon 06:00, Federal holidays</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <h3>Tariff Schedule</h3>
        <div class="card">
            {tariff_schedule_html}
        </div>

        <details>
            <summary>Tariff Details and Data Sources</summary>
            <div class="card">
                {tariff_report_section}
            </div>
        </details>

        <h2 id="outputs">6. Output Files</h2>

        <div class="card">
            <table>
                <thead>
                    <tr><th>File</th><th>Description</th><th>Format</th></tr>
                </thead>
                <tbody>
                    <tr><td>energy_balance_15min.parquet</td><td>15-minute interval energy data (kWh)</td><td>Parquet</td></tr>
                    <tr><td>energy_balance_daily.parquet</td><td>Daily totals from monthly files</td><td>Parquet</td></tr>
                    <tr><td>energy_balance_monthly.parquet</td><td>Monthly totals from yearly files</td><td>Parquet</td></tr>
                    <tr><td>sensors_heating.parquet</td><td>Heat pump and heating sensors</td><td>Parquet</td></tr>
                    <tr><td>sensors_weather.parquet</td><td>Davis weather station data</td><td>Parquet</td></tr>
                    <tr><td>sensors_rooms.parquet</td><td>Room temperature sensors</td><td>Parquet</td></tr>
                    <tr><td>sensors_energy.parquet</td><td>Smart plug consumption data</td><td>Parquet</td></tr>
                    <tr><td>integrated_dataset.parquet</td><td>All data merged (full period)</td><td>Parquet</td></tr>
                    <tr><td>integrated_overlap_only.parquet</td><td>Merged data (overlap period only)</td><td>Parquet</td></tr>
                    <tr><td>corrections_log.csv</td><td>Data cleaning corrections log</td><td>CSV</td></tr>
                    <tr><td>validation_results.csv</td><td>Daily vs monthly validation</td><td>CSV</td></tr>
                    <tr><td>sensor_summary.csv</td><td>Per-sensor statistics</td><td>CSV</td></tr>
                    <tr><td>data_overlap_summary.csv</td><td>Data source overlap info</td><td>CSV</td></tr>
                    <tr><td>tariff_schedule.csv</td><td>Electricity tariff rates by period</td><td>CSV</td></tr>
                    <tr><td>tariff_flags_hourly.parquet</td><td>Hourly high/low tariff flags</td><td>Parquet</td></tr>
                    <tr><td>tariff_series_hourly.parquet</td><td>Time-indexed tariff rates</td><td>Parquet</td></tr>
                </tbody>
            </table>
        </div>

        <h2 id="logs">7. Detailed Logs</h2>

        <details>
            <summary>Step 1: Energy Balance Preprocessing Log</summary>
            <pre>{format_log(step_logs.get('energy_balance', (False, 'Not executed'))[1])}</pre>
        </details>

        <details>
            <summary>Step 2: Sensor Data Preprocessing Log</summary>
            <pre>{format_log(step_logs.get('sensors', (False, 'Not executed'))[1])}</pre>
        </details>

        <details>
            <summary>Step 3: Data Integration Log</summary>
            <pre>{format_log(step_logs.get('integration', (False, 'Not executed'))[1])}</pre>
        </details>

        <details>
            <summary>Step 4: Tariff Preprocessing Log</summary>
            <pre>{format_log(step_logs.get('tariffs', (False, 'Not executed'))[1])}</pre>
        </details>

        <div class="footer">
            <p>ESTAT - Energy System Analysis | Phase 1: Preprocessing</p>
        </div>
    </div>
</body>
</html>"""

    return html


def regenerate_report_only():
    """Regenerate HTML report from existing data without re-running preprocessing."""
    print("="*60)
    print("REGENERATING HTML REPORT (using existing data)")
    print("="*60)

    # Create mock step logs (mark all as success since data exists)
    step_logs['energy_balance'] = (True, "Using existing preprocessed data")
    step_logs['sensors'] = (True, "Using existing preprocessed data")
    step_logs['integration'] = (True, "Using existing preprocessed data")
    step_logs['tariffs'] = (True, "Using existing preprocessed data")

    # Extract stats from existing files
    all_stats = {}

    # Try to read some basic stats from parquet files
    try:
        import pyarrow.parquet as pq
        energy_15min = PROCESSED_DIR / "energy_balance_15min.parquet"
        if energy_15min.exists():
            table = pq.read_table(energy_15min)
            all_stats['daily_records'] = table.num_rows

        integrated = PROCESSED_DIR / "integrated_dataset.parquet"
        if integrated.exists():
            table = pq.read_table(integrated)
            all_stats['merged_records'] = table.num_rows
            all_stats['merged_columns'] = len(table.column_names)

        overlap = PROCESSED_DIR / "integrated_overlap_only.parquet"
        if overlap.exists():
            table = pq.read_table(overlap)
            # Estimate days from 15-min records
            all_stats['overlap_days'] = table.num_rows // 96

        # Sensor stats from summary
        sensor_summary = PROCESSED_DIR / "sensor_summary.csv"
        if sensor_summary.exists():
            df = pd.read_csv(sensor_summary)
            for cat in ['heating', 'weather', 'rooms', 'energy']:
                count = len(df[df['category'] == cat]) if 'category' in df.columns else 0
                all_stats[f'{cat}_sensors'] = count

        # Tariff stats
        tariff_flags = PROCESSED_DIR / "tariff_flags_hourly.parquet"
        if tariff_flags.exists():
            table = pq.read_table(tariff_flags)
            df = table.to_pandas()
            all_stats['high_tariff_hours'] = df['is_high_tariff'].sum()
            all_stats['low_tariff_hours'] = df['is_low_tariff'].sum()

        tariff_schedule = PROCESSED_DIR / "tariff_schedule.csv"
        if tariff_schedule.exists():
            df = pd.read_csv(tariff_schedule)
            all_stats['tariff_entries'] = len(df)

    except Exception as e:
        print(f"Warning: Could not extract all stats: {e}")

    # Generate HTML report
    html_report = generate_html_report(step_logs, all_stats)
    report_path = PROCESSED_DIR / "phase1_report.html"
    report_path.write_text(html_report)
    print(f"Report saved to: {report_path}")

    return 0


def main():
    """Run complete preprocessing pipeline and generate report."""
    # Check for --report-only flag
    if len(sys.argv) > 1 and sys.argv[1] == "--report-only":
        return regenerate_report_only()

    print("="*60)
    print("ESTAT PHASE 1: DATA PREPROCESSING PIPELINE")
    print("="*60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    all_stats = {}

    # Step 1: Energy Balance
    success, log = run_step(
        "Step 1: Energy Balance Preprocessing",
        SRC_DIR / "01_preprocess_energy_balance.py"
    )
    step_logs['energy_balance'] = (success, log)
    all_stats.update(extract_stats_from_log(log))

    if not success:
        print("\nWARNING: Energy balance preprocessing failed. Continuing...")

    # Step 2: Sensors
    success, log = run_step(
        "Step 2: Sensor Data Preprocessing",
        SRC_DIR / "02_preprocess_sensors.py"
    )
    step_logs['sensors'] = (success, log)
    all_stats.update(extract_stats_from_log(log))

    if not success:
        print("\nWARNING: Sensor preprocessing failed. Continuing...")

    # Step 3: Integration
    success, log = run_step(
        "Step 3: Data Integration",
        SRC_DIR / "03_integrate_data.py"
    )
    step_logs['integration'] = (success, log)
    all_stats.update(extract_stats_from_log(log))

    # Step 4: Tariffs
    success, log = run_step(
        "Step 4: Tariff Preprocessing",
        SRC_DIR / "04_preprocess_tariffs.py"
    )
    step_logs['tariffs'] = (success, log)
    all_stats.update(extract_stats_from_log(log))

    # Generate HTML report
    print("\n" + "="*60)
    print("Generating HTML Report")
    print("="*60)

    html_report = generate_html_report(step_logs, all_stats)
    report_path = PROCESSED_DIR / "phase1_report.html"
    report_path.write_text(html_report)
    print(f"Report saved to: {report_path}")

    # Summary
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nOutput directory: {PROCESSED_DIR}")
    print(f"Report: {report_path}")

    # Return success status
    all_success = all(step_logs[k][0] for k in step_logs)
    return 0 if all_success else 1


if __name__ == "__main__":
    sys.exit(main())
