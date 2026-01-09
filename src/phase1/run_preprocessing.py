#!/usr/bin/env python3
"""
Phase 1: Data Preprocessing Pipeline

This script runs all preprocessing steps and generates an HTML report
documenting the data cleaning, transformations, and quality checks.

Steps:
1. Energy balance preprocessing (daily, monthly, yearly CSV files)
2. Sensor data preprocessing (InfluxDB export from Home Assistant)
3. Data integration (merge energy and sensor data)

Output:
- Processed parquet files in output/phase1/
- HTML report: output/phase1/preprocessing_report.html
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

    return stats


def generate_html_report(step_logs: dict, stats: dict) -> str:
    """Generate comprehensive HTML preprocessing report."""

    # Load CSV summaries
    corrections_html = load_csv_as_html_table(PROCESSED_DIR / "corrections_log.csv")
    validation_html = load_csv_as_html_table(PROCESSED_DIR / "validation_results_fuzzy.csv")
    sensor_summary_html = load_csv_as_html_table(PROCESSED_DIR / "sensor_summary.csv", max_rows=None)
    overlap_html = load_csv_as_html_table(PROCESSED_DIR / "data_overlap_summary.csv")
    energy_summary = load_text_file(PROCESSED_DIR / "energy_balance_summary.txt")

    # Format logs for HTML
    def format_log(log: str) -> str:
        return log.replace('<', '&lt;').replace('>', '&gt;')

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ESTAT Phase 1: Preprocessing Report</title>
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
            --code-bg: #1e293b;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
            padding: 2rem;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{ color: var(--primary); margin-bottom: 0.5rem; }}
        h2 {{ color: var(--text); margin: 2rem 0 1rem; padding-bottom: 0.5rem; border-bottom: 2px solid var(--primary); }}
        h3 {{ color: var(--text-muted); margin: 1.5rem 0 0.75rem; }}
        h4 {{ margin: 1rem 0 0.5rem; }}
        .meta {{ color: var(--text-muted); margin-bottom: 2rem; }}
        .card {{
            background: var(--card-bg);
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; }}
        .stat-box {{
            background: var(--card-bg);
            border-radius: 8px;
            padding: 1rem 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .stat-box .value {{ font-size: 1.75rem; font-weight: bold; color: var(--primary); }}
        .stat-box .label {{ color: var(--text-muted); font-size: 0.875rem; }}
        .success {{ color: var(--success); }}
        .warning {{ color: var(--warning); }}
        .error {{ color: var(--danger); }}
        .note {{ color: var(--text-muted); font-style: italic; margin-bottom: 0.5rem; }}
        table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
        th, td {{ padding: 0.5rem 0.75rem; text-align: left; border-bottom: 1px solid var(--border); font-size: 0.875rem; }}
        th {{ background: var(--bg); font-weight: 600; }}
        .data-table {{ font-size: 0.8rem; }}
        .data-table th, .data-table td {{ padding: 0.35rem 0.5rem; }}
        pre {{
            background: var(--code-bg);
            color: #e2e8f0;
            padding: 1rem;
            border-radius: 8px;
            overflow-x: auto;
            font-size: 0.8rem;
            line-height: 1.4;
            max-height: 400px;
            overflow-y: auto;
        }}
        .toc {{ background: var(--card-bg); padding: 1.5rem; border-radius: 8px; margin-bottom: 2rem; }}
        .toc ul {{ list-style: none; }}
        .toc li {{ margin: 0.5rem 0; }}
        .toc a {{ color: var(--primary); text-decoration: none; }}
        .toc a:hover {{ text-decoration: underline; }}
        .step-header {{
            display: flex;
            align-items: center;
            gap: 1rem;
        }}
        .step-status {{
            padding: 0.25rem 0.75rem;
            border-radius: 4px;
            font-size: 0.875rem;
            font-weight: 500;
        }}
        .step-status.success {{ background: #dcfce7; color: var(--success); }}
        .step-status.failed {{ background: #fee2e2; color: var(--danger); }}
        details {{ margin: 1rem 0; }}
        summary {{
            cursor: pointer;
            padding: 0.5rem;
            background: var(--bg);
            border-radius: 4px;
            font-weight: 500;
        }}
        summary:hover {{ background: var(--border); }}
        .methodology {{
            background: #eff6ff;
            border-left: 4px solid var(--primary);
            padding: 1rem 1.5rem;
            margin: 1rem 0;
        }}
        .methodology h4 {{ margin-top: 0; }}
        .methodology ul {{ margin-left: 1.5rem; }}
        .methodology li {{ margin: 0.25rem 0; }}
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
                <li><a href="#outputs">5. Output Files</a></li>
                <li><a href="#logs">6. Detailed Logs</a></li>
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
                <div class="value">{stats.get('validation_match_rate', 'N/A'):.1f}%</div>
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

        <h2 id="outputs">5. Output Files</h2>

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
                </tbody>
            </table>
        </div>

        <h2 id="logs">6. Detailed Logs</h2>

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

    </div>
</body>
</html>"""

    return html


def main():
    """Run complete preprocessing pipeline and generate report."""
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

    # Generate HTML report
    print("\n" + "="*60)
    print("Generating HTML Report")
    print("="*60)

    html_report = generate_html_report(step_logs, all_stats)
    report_path = PROCESSED_DIR / "preprocessing_report.html"
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
