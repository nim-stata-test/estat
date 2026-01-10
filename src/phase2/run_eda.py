#!/usr/bin/env python3
"""
Phase 2: Exploratory Data Analysis Pipeline

This script runs all EDA steps and generates an HTML report
documenting the analysis results.

Steps:
1. Load and filter sensor data (exclude non-room sensors, apply mappings)
2. Run EDA analysis (energy patterns, heating system, solar-heating correlation)
3. Generate HTML report

Note: Battery degradation analysis is in separate script 02_battery_degradation.py

Configuration:
- EXCLUDED_ROOM_SENSORS: Sensors to exclude from room analysis
- SENSOR_MAPPINGS: Rename sensors (e.g., office2 → atelier)
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "output" / "phase1"
OUTPUT_DIR = PROJECT_ROOT / "output" / "phase2"
SRC_DIR = PROJECT_ROOT / "src" / "phase2"

# =============================================================================
# EDA Configuration
# =============================================================================

# Room sensors to exclude from analysis (not actual room temperatures)
EXCLUDED_ROOM_SENSORS = [
    "temphum_woz_temperature",  # Not a main room
    "motion_gang_2_temperature",  # Motion sensor, not room temp
    "motion_halle_n_temperature",  # Motion sensor, not room temp
    "outfeeler_temperature",  # Outdoor sensor
]

# Sensor name mappings (old_name → new_name)
# office2 was renamed to atelier at some point
SENSOR_MAPPINGS = {
    "office2_temperature": "atelier_temperature",
    "office2_humidity": "atelier_humidity",
}

# Primary room sensors for analysis (after filtering and mapping)
PRIMARY_ROOM_SENSORS = [
    "atelier_temperature",
    "bric_temperature",
    "dorme_temperature",
    "halle_temperature",
    "office1_temperature",
    "simlab_temperature",
    "studio_temperature",
    "guest_temperature",
    "temp_cave",
    "temphum_plant_temperature",
    "temphum_bano_temperature",
    "davis_inside_temperature",
]


def prepare_room_data() -> pd.DataFrame:
    """Load and prepare room sensor data with filtering and mappings."""
    print("Preparing room sensor data...")

    rooms_path = PROCESSED_DIR / "sensors_rooms.parquet"
    if not rooms_path.exists():
        print(f"  ERROR: {rooms_path} not found")
        return pd.DataFrame()

    df = pd.read_parquet(rooms_path)
    initial_count = len(df)
    initial_sensors = df['entity_id'].nunique()

    # Exclude sensors
    exclude_mask = df['entity_id'].isin(EXCLUDED_ROOM_SENSORS)
    excluded_count = exclude_mask.sum()
    df = df[~exclude_mask]

    print(f"  Excluded {excluded_count} records from {len(EXCLUDED_ROOM_SENSORS)} sensors")

    # Apply mappings
    mappings_applied = 0
    for old_name, new_name in SENSOR_MAPPINGS.items():
        mask = df['entity_id'] == old_name
        if mask.any():
            df.loc[mask, 'entity_id'] = new_name
            mappings_applied += mask.sum()
            print(f"  Mapped {old_name} → {new_name} ({mask.sum()} records)")

    # Summary
    final_sensors = df['entity_id'].nunique()
    print(f"  Final: {len(df)} records, {final_sensors} sensors (was {initial_sensors})")

    return df


def save_prepared_data(rooms_df: pd.DataFrame):
    """Save prepared room data for EDA consumption."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save filtered/mapped room data
    output_path = OUTPUT_DIR / "rooms_prepared.parquet"
    rooms_df.to_parquet(output_path, index=False)
    print(f"  Saved prepared room data to: {output_path}")

    return output_path


def run_eda_script() -> tuple[bool, str]:
    """Run the main EDA analysis script."""
    print("\n" + "="*60)
    print("Running EDA Analysis")
    print("="*60)

    script_path = SRC_DIR / "01_eda.py"

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=600
        )

        output = result.stdout
        if result.stderr:
            output += "\n\nSTDERR:\n" + result.stderr

        print(output)

        return result.returncode == 0, output

    except subprocess.TimeoutExpired:
        return False, "ERROR: EDA script timed out"
    except Exception as e:
        return False, f"ERROR: {e}"


def run_heating_curve_analysis() -> tuple[bool, str]:
    """Run the heating curve analysis script."""
    print("\n" + "="*60)
    print("Running Heating Curve Analysis")
    print("="*60)

    script_path = SRC_DIR / "03_heating_curve_analysis.py"

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=300
        )

        output = result.stdout
        if result.stderr:
            output += "\n\nSTDERR:\n" + result.stderr

        print(output)

        return result.returncode == 0, output

    except subprocess.TimeoutExpired:
        return False, "ERROR: Heating curve script timed out"
    except Exception as e:
        return False, f"ERROR: {e}"


def run_weighted_temp_analysis() -> tuple[bool, str]:
    """Run the weighted temperature analysis script."""
    print("\n" + "="*60)
    print("Running Weighted Temperature Analysis")
    print("="*60)

    script_path = SRC_DIR / "05_weighted_temperature_analysis.py"

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=300
        )

        output = result.stdout
        if result.stderr:
            output += "\n\nSTDERR:\n" + result.stderr

        print(output)

        return result.returncode == 0, output

    except subprocess.TimeoutExpired:
        return False, "ERROR: Weighted temperature script timed out"
    except Exception as e:
        return False, f"ERROR: {e}"


def run_tariff_analysis() -> tuple[bool, str]:
    """Run the tariff analysis script."""
    print("\n" + "="*60)
    print("Running Tariff Analysis")
    print("="*60)

    script_path = SRC_DIR / "04_tariff_analysis.py"

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=300
        )

        output = result.stdout
        if result.stderr:
            output += "\n\nSTDERR:\n" + result.stderr

        print(output)

        return result.returncode == 0, output

    except subprocess.TimeoutExpired:
        return False, "ERROR: Tariff analysis script timed out"
    except Exception as e:
        return False, f"ERROR: {e}"


def collect_figure_info() -> list[dict]:
    """Collect information about generated figures."""
    figures = []

    figure_descriptions = {
        "fig01_daily_energy_timeseries.png": "Daily PV generation, consumption, grid interaction, and battery activity over time",
        "fig02_monthly_energy_patterns.png": "Monthly aggregated energy patterns, grid balance, and self-sufficiency",
        "fig03_hourly_heatmaps.png": "Average consumption, PV generation, and grid import by hour and day of week",
        "fig04_seasonal_patterns.png": "Seasonal comparison of PV generation, consumption, and self-sufficiency",
        "fig05_heat_pump_cop.png": "Heat pump COP analysis: time series, vs outdoor temperature, consumption patterns",
        "fig06_temperature_differentials.png": "Flow/return temperatures, buffer tank, and heating circuits",
        "fig07_indoor_outdoor_temp.png": "Outdoor vs indoor room temperatures with comfort bounds",
        "fig08_solar_heating_hourly.png": "Hourly PV generation, heating activity, and grid import profiles",
        "fig09_battery_evening_heating.png": "Battery charging/discharging patterns and evening energy sources",
        "fig10_forced_grid_heating.png": "Periods of grid-dependent heating when no PV is available",
        "fig11_summary_statistics.png": "Monthly breakdown, HDD analysis, consumption distribution, yearly totals",
        "fig12_heating_curve_schedule.png": "Heating curve analysis: setpoint regimes, target vs outdoor temperature, model residuals",
        "fig12a_heating_curve_censored.png": "Heating curve analysis (censored): excluding anomalous eco >= comfort periods",
        "fig13_weighted_temp_parameters.png": "Weighted temperature analysis: parameter response with 48h washout exclusion",
        "fig14_tariff_timeline.png": "Electricity tariff timeline: purchase and feed-in rates over time (2023-2025)",
        "fig15_tariff_windows.png": "Tariff time windows: high/low tariff distribution by hour and day",
        "fig16_tariff_costs.png": "Tariff cost implications: rate trends and comparison scenarios",
    }

    for fig_file, description in figure_descriptions.items():
        fig_path = OUTPUT_DIR / fig_file
        if fig_path.exists():
            figures.append({
                "filename": fig_file,
                "description": description,
                "exists": True,
            })
        else:
            figures.append({
                "filename": fig_file,
                "description": description,
                "exists": False,
            })

    return figures


def extract_stats_from_log(log: str) -> dict:
    """Extract key statistics from EDA log output."""
    import re
    stats = {}

    # Energy stats
    if match := re.search(r'PV Generation:\s+([\d.]+) kWh/day', log):
        stats['pv_daily'] = float(match.group(1))
    if match := re.search(r'Total Consumption:\s+([\d.]+) kWh/day', log):
        stats['consumption_daily'] = float(match.group(1))
    if match := re.search(r'Grid Import:\s+([\d.]+) kWh/day', log):
        stats['grid_import_daily'] = float(match.group(1))
    if match := re.search(r'Self-Sufficiency:\s+([\d.]+)%', log):
        stats['self_sufficiency'] = float(match.group(1))

    # COP stats
    if match := re.search(r'Mean COP:\s+([\d.]+)', log):
        stats['cop_mean'] = float(match.group(1))
    if match := re.search(r'Min COP:\s+([\d.]+)', log):
        stats['cop_min'] = float(match.group(1))
    if match := re.search(r'Max COP:\s+([\d.]+)', log):
        stats['cop_max'] = float(match.group(1))

    # Heating stats
    if match := re.search(r'Daily Heating Electricity:.*?Mean:\s+([\d.]+) kWh/day', log, re.DOTALL):
        stats['heating_mean'] = float(match.group(1))
    if match := re.search(r'Daily Heating Electricity:.*?Max:\s+([\d.]+) kWh/day', log, re.DOTALL):
        stats['heating_max'] = float(match.group(1))

    # Solar-heating correlation
    if match := re.search(r'Solar hours.*?Heating active: ([\d.]+)% of time', log, re.DOTALL):
        stats['heating_solar_hours_pct'] = float(match.group(1))
    if match := re.search(r'Non-solar hours.*?Heating active: ([\d.]+)% of time', log, re.DOTALL):
        stats['heating_nonsolar_hours_pct'] = float(match.group(1))

    return stats


def load_heating_curve_section() -> str:
    """Load the heating curve HTML section if available."""
    section_path = OUTPUT_DIR / "heating_curve_report_section.html"
    if section_path.exists():
        return section_path.read_text()
    return ""


def load_weighted_temp_section() -> str:
    """Load the weighted temperature analysis HTML section if available."""
    section_path = OUTPUT_DIR / "weighted_temp_report_section.html"
    if section_path.exists():
        return section_path.read_text()
    return ""


def load_tariff_section() -> str:
    """Load the tariff analysis HTML section if available."""
    section_path = OUTPUT_DIR / "tariff_report_section.html"
    if section_path.exists():
        return section_path.read_text()
    return ""


def generate_html_report(figures: list[dict], stats: dict, eda_log: str,
                         heating_curve_log: str = "", weighted_temp_log: str = "",
                         tariff_log: str = "") -> str:
    """Generate comprehensive HTML EDA report."""

    # Helper to format stats with fallback for missing values
    def fmt(key: str, format_spec: str = '.2f', suffix: str = '') -> str:
        val = stats.get(key)
        if val is None:
            return 'N/A'
        try:
            return f"{val:{format_spec}}{suffix}"
        except (ValueError, TypeError):
            return str(val)

    # Build figure gallery HTML
    figures_html = ""
    for fig in figures:
        if fig['exists']:
            figures_html += f"""
            <div class="figure">
                <img src="{fig['filename']}" alt="{fig['description']}">
                <div class="figure-caption">{fig['description']}</div>
            </div>
            """
        else:
            figures_html += f"""
            <div class="figure missing">
                <p class="error">Figure not generated: {fig['filename']}</p>
                <div class="figure-caption">{fig['description']}</div>
            </div>
            """

    # Format log for HTML
    def format_log(log: str) -> str:
        return log.replace('<', '&lt;').replace('>', '&gt;')

    # Room sensor configuration
    excluded_html = "<ul>" + "".join(f"<li>{s}</li>" for s in EXCLUDED_ROOM_SENSORS) + "</ul>"
    mappings_html = "<ul>" + "".join(f"<li>{k} → {v}</li>" for k, v in SENSOR_MAPPINGS.items()) + "</ul>"
    primary_html = "<ul>" + "".join(f"<li>{s}</li>" for s in PRIMARY_ROOM_SENSORS) + "</ul>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ESTAT Phase 2: EDA Report</title>
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
        ul {{ margin-left: 1.5rem; }}
        li {{ margin: 0.25rem 0; }}
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
        .toc ul {{ list-style: none; margin: 0; }}
        .toc li {{ margin: 0.5rem 0; }}
        .toc a {{ color: var(--primary); text-decoration: none; }}
        .toc a:hover {{ text-decoration: underline; }}
        .figure {{
            background: var(--card-bg);
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .figure img {{ width: 100%; height: auto; border-radius: 4px; }}
        .figure-caption {{ color: var(--text-muted); font-size: 0.875rem; margin-top: 0.5rem; text-align: center; }}
        .figure.missing {{ border: 2px dashed var(--danger); }}
        details {{ margin: 1rem 0; }}
        summary {{
            cursor: pointer;
            padding: 0.5rem;
            background: var(--bg);
            border-radius: 4px;
            font-weight: 500;
        }}
        summary:hover {{ background: var(--border); }}
        .config-section {{
            background: #eff6ff;
            border-left: 4px solid var(--primary);
            padding: 1rem 1.5rem;
            margin: 1rem 0;
        }}
        table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
        th, td {{ padding: 0.5rem 0.75rem; text-align: left; border-bottom: 1px solid var(--border); }}
        th {{ background: var(--bg); font-weight: 600; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>ESTAT Phase 2: Exploratory Data Analysis Report</h1>
        <p class="meta">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <div class="toc">
            <strong>Contents</strong>
            <ul>
                <li><a href="#overview">1. Overview</a></li>
                <li><a href="#config">2. Configuration</a></li>
                <li><a href="#quality">3. Data Quality Notes</a></li>
                <li><a href="#energy">4. Energy Patterns</a></li>
                <li><a href="#heating">5. Heating System</a></li>
                <li><a href="#solar">6. Solar-Heating Correlation</a></li>
                <li><a href="#summary">7. Summary Statistics</a></li>
                <li><a href="#heating-curve">8. Heating Curve Analysis</a></li>
                <li><a href="#weighted-temp">9. Weighted Temperature Analysis</a></li>
                <li><a href="#tariffs">10. Electricity Tariffs</a></li>
                <li><a href="#log">11. Detailed Log</a></li>
            </ul>
        </div>

        <h2 id="overview">1. Overview</h2>

        <div class="grid">
            <div class="stat-box">
                <div class="value">{fmt('pv_daily', '.1f')}</div>
                <div class="label">kWh/day PV Generation</div>
            </div>
            <div class="stat-box">
                <div class="value">{fmt('consumption_daily', '.1f')}</div>
                <div class="label">kWh/day Consumption</div>
            </div>
            <div class="stat-box">
                <div class="value">{fmt('self_sufficiency', '.0f', '%')}</div>
                <div class="label">Self-Sufficiency</div>
            </div>
            <div class="stat-box">
                <div class="value">{fmt('cop_mean')}</div>
                <div class="label">Mean Heat Pump COP</div>
            </div>
        </div>

        <div class="card">
            <h4>Heat Pump Performance</h4>
            <table>
                <tr><td>Mean COP</td><td><strong>{fmt('cop_mean')}</strong></td></tr>
                <tr><td>COP Range</td><td>{fmt('cop_min')} - {fmt('cop_max')}</td></tr>
                <tr><td>Mean Heating Electricity</td><td>{fmt('heating_mean', '.1f')} kWh/day</td></tr>
                <tr><td>Max Heating Electricity</td><td>{fmt('heating_max', '.1f')} kWh/day</td></tr>
            </table>
        </div>

        <h2 id="config">2. Configuration</h2>

        <div class="config-section">
            <h4>Room Sensor Configuration</h4>
            <p>The following configuration was applied to room sensors for this analysis:</p>

            <h4>Excluded Sensors</h4>
            <p>These sensors were excluded (not actual room temperatures):</p>
            {excluded_html}

            <h4>Sensor Mappings</h4>
            <p>These sensors were renamed (same physical sensor, renamed over time):</p>
            {mappings_html}

            <details>
                <summary>Primary Room Sensors Used</summary>
                {primary_html}
            </details>
        </div>

        <h2 id="quality">3. Data Quality Notes</h2>

        <div class="card">
            <h4>COP Calculation Methodology</h4>
            <p>Daily Coefficient of Performance (COP) is calculated as:</p>
            <pre style="background: var(--card-bg); color: var(--text); padding: 0.5rem;">COP = produced_heating / consumed_heating</pre>
            <p>Where both values are daily deltas from cumulative counter sensors.</p>

            <h4>Overlapping Time Window Filter</h4>
            <p>To ensure accurate COP calculations, daily values are computed only from <strong>overlapping time periods</strong>
            where both <code>consumed_heating</code> and <code>produced_heating</code> sensors have data.</p>
            <ul>
                <li>Days with less than 20 hours of sensor overlap are excluded</li>
                <li>This prevents artificially low COP values when one sensor has data gaps (e.g., after spike removal in preprocessing)</li>
                <li>Example: If spike removal creates a gap in <code>produced_heating</code> until noon, but <code>consumed_heating</code>
                has data from midnight, only the noon-to-midnight overlap is used for that day's calculation</li>
            </ul>

            <h4>Additional Filters</h4>
            <ul>
                <li>Negative delta values are excluded (sensor reset/error)</li>
                <li>Daily consumed &gt; 100 kWh excluded (unrealistic)</li>
                <li>Daily produced &gt; 500 kWh excluded (unrealistic)</li>
                <li>COP &lt; 1 or &gt; 10 excluded (physically impossible for heat pump)</li>
            </ul>
        </div>

        <div class="card">
            <h4>Glossary of Terms</h4>
            <table>
                <tr>
                    <td><strong>COP (Coefficient of Performance)</strong></td>
                    <td>
                        <p>The efficiency metric for heat pumps, defined as the ratio of useful heat energy delivered to electrical energy consumed:</p>
                        <pre style="background: var(--card-bg); color: var(--text); padding: 0.5rem; margin: 0.5rem 0;">COP = Heat Output (kWh) / Electrical Input (kWh)</pre>
                        <p>A COP of 3.5 means for every 1 kWh of electricity consumed, 3.5 kWh of heat is delivered to the building.
                        The "extra" energy comes from the outdoor air (or ground), which the heat pump extracts and concentrates.</p>
                        <p><strong>Factors affecting COP:</strong></p>
                        <ul>
                            <li><em>Outdoor temperature:</em> COP decreases as outdoor temp drops (less heat available to extract).
                            At 10°C outdoor, COP might be 4.5; at -5°C, it might drop to 2.5.</li>
                            <li><em>Flow temperature:</em> Higher heating circuit temperatures (e.g., for radiators vs underfloor) reduce COP.</li>
                            <li><em>Defrost cycles:</em> In cold/humid conditions, the outdoor unit frosts up and needs periodic defrosting, reducing effective COP.</li>
                        </ul>
                        <p><strong>Interpretation:</strong> COP &lt; 1 is impossible (would violate thermodynamics). COP of 1 equals resistive heating.
                        Modern air-source heat pumps typically achieve seasonal average COP of 2.5–4.0. Values above 5 are excellent but rare.</p>
                    </td>
                </tr>
                <tr>
                    <td><strong>Heating Degree Days (HDD)</strong></td>
                    <td>
                        <p>A weather-based metric that quantifies heating demand. For each day, HDD is calculated as:</p>
                        <pre style="background: var(--card-bg); color: var(--text); padding: 0.5rem; margin: 0.5rem 0;">HDD = max(0, T_base - T_outdoor_avg)</pre>
                        <p>Where <code>T_base</code> is the balance point temperature (15°C in this analysis) below which heating is typically needed,
                        and <code>T_outdoor_avg</code> is the average outdoor temperature for that day.</p>
                        <p><strong>Examples:</strong></p>
                        <ul>
                            <li>Outdoor avg = 5°C → HDD = 15 - 5 = <strong>10</strong> (cold day, significant heating needed)</li>
                            <li>Outdoor avg = 12°C → HDD = 15 - 12 = <strong>3</strong> (mild day, some heating needed)</li>
                            <li>Outdoor avg = 18°C → HDD = max(0, 15 - 18) = <strong>0</strong> (warm day, no heating needed)</li>
                        </ul>
                        <p><strong>Usage:</strong> HDD allows comparison of heating consumption across different weather conditions.
                        A well-insulated building uses fewer kWh per HDD. The slope of the "Heating vs HDD" plot indicates building thermal performance:
                        steeper slope = more energy needed per degree of temperature difference = poorer insulation or higher heat loss.</p>
                    </td>
                </tr>
                <tr>
                    <td><strong>Forced Grid Consumption</strong></td>
                    <td>Periods when the heat pump is running but must draw electricity from the grid because:
                    (1) no solar PV is available (night/cloudy), and (2) battery is empty or insufficient.
                    This represents unavoidable grid dependency. Minimizing forced grid consumption is a key optimization goal.</td>
                </tr>
                <tr>
                    <td><strong>Self-Sufficiency</strong></td>
                    <td>The percentage of total consumption that comes directly from solar PV (not from grid or battery).
                    Calculated as: <code>direct_consumption / total_consumption</code>.
                    Higher is better. 100% means all energy used came directly from PV at time of generation.</td>
                </tr>
                <tr>
                    <td><strong>Solar Hours</strong></td>
                    <td>Defined as 8:00–17:00 (8 AM to 5 PM) when meaningful solar generation is typically available.
                    Used to analyze heating patterns relative to PV availability.</td>
                </tr>
            </table>
        </div>

        <h2 id="energy">4. Energy Patterns</h2>

        {figures_html.split('fig05')[0] if 'fig05' in figures_html else figures_html[:len(figures_html)//3]}

        <h2 id="heating">5. Heating System</h2>

        {''.join([f for f in figures_html.split('</div>') if 'fig05' in f or 'fig06' in f or 'fig07' in f])}

        <h2 id="solar">6. Solar-Heating Correlation</h2>

        <div class="card">
            <h4>Heating Activity by Time of Day</h4>
            <table>
                <tr><td>Solar hours (8AM-5PM)</td><td>{fmt('heating_solar_hours_pct', '.1f')}% heating active</td></tr>
                <tr><td>Non-solar hours (5PM-8AM)</td><td>{fmt('heating_nonsolar_hours_pct', '.1f')}% heating active</td></tr>
            </table>
        </div>

        {''.join([f for f in figures_html.split('</div>') if 'fig08' in f or 'fig09' in f or 'fig10' in f])}

        <h2 id="summary">7. Summary Statistics</h2>

        {''.join([f for f in figures_html.split('</div>') if 'fig11' in f])}

        {load_heating_curve_section()}

        {load_weighted_temp_section()}

        {load_tariff_section()}

        <h2 id="log">11. Detailed Log</h2>

        <details>
            <summary>Full EDA Output Log</summary>
            <pre>{format_log(eda_log)}</pre>
        </details>

        <details>
            <summary>Heating Curve Analysis Log</summary>
            <pre>{format_log(heating_curve_log)}</pre>
        </details>

        <details>
            <summary>Weighted Temperature Analysis Log</summary>
            <pre>{format_log(weighted_temp_log)}</pre>
        </details>

        <details>
            <summary>Tariff Analysis Log</summary>
            <pre>{format_log(tariff_log)}</pre>
        </details>

    </div>
</body>
</html>"""

    return html


def main():
    """Run complete EDA pipeline and generate report."""
    print("="*60)
    print("ESTAT PHASE 2: EXPLORATORY DATA ANALYSIS PIPELINE")
    print("="*60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Prepare room sensor data
    print("\n" + "="*60)
    print("Step 1: Preparing Room Sensor Data")
    print("="*60)

    rooms_df = prepare_room_data()
    if not rooms_df.empty:
        save_prepared_data(rooms_df)

    # Step 2: Run EDA analysis
    success, eda_log = run_eda_script()

    if not success:
        print("\nWARNING: EDA script encountered errors")

    # Step 3: Run heating curve analysis
    hc_success, heating_curve_log = run_heating_curve_analysis()

    if not hc_success:
        print("\nWARNING: Heating curve analysis encountered errors")

    # Step 4: Run weighted temperature analysis
    wt_success, weighted_temp_log = run_weighted_temp_analysis()

    if not wt_success:
        print("\nWARNING: Weighted temperature analysis encountered errors")

    # Step 5: Run tariff analysis
    tariff_success, tariff_log = run_tariff_analysis()

    if not tariff_success:
        print("\nWARNING: Tariff analysis encountered errors")

    # Step 6: Generate HTML report
    print("\n" + "="*60)
    print("Generating HTML Report")
    print("="*60)

    figures = collect_figure_info()
    stats = extract_stats_from_log(eda_log)

    html_report = generate_html_report(figures, stats, eda_log, heating_curve_log, weighted_temp_log, tariff_log)
    report_path = OUTPUT_DIR / "eda_report.html"
    report_path.write_text(html_report)
    print(f"Report saved to: {report_path}")

    # Summary
    print("\n" + "="*60)
    print("EDA PIPELINE COMPLETE")
    print("="*60)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Report: {report_path}")
    print(f"Figures: {sum(1 for f in figures if f['exists'])}/{len(figures)} generated")

    return 0 if (success and hc_success and wt_success and tariff_success) else 1


if __name__ == "__main__":
    sys.exit(main())
