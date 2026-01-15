#!/usr/bin/env python3
"""
Phase 3: System Modeling - Master Script

Runs all Phase 3 modeling steps and generates a combined HTML report:
1. Thermal Model - Building heat loss and response characteristics
2. Heat Pump Model - COP relationships, capacity, buffer tank
3. Energy System Model - PV patterns, battery dynamics, self-sufficiency
4. Tariff Cost Model - Electricity cost analysis and forecasting

Usage:
    python src/phase3/run_phase3.py
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'output' / 'phase3'
OUTPUT_DIR.mkdir(exist_ok=True)


def run_script(script_name: str) -> bool:
    """Run a Phase 3 script and return success status."""
    script_path = Path(__file__).parent / script_name

    print(f"\n{'='*60}")
    print(f"Running {script_name}")
    print('='*60)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_ROOT
    )

    return result.returncode == 0


def generate_html_report():
    """Combine individual report sections into full HTML report."""
    print(f"\n{'='*60}")
    print("Generating combined HTML report")
    print('='*60)

    # Read individual sections
    sections = []

    thermal_path = OUTPUT_DIR / 'thermal_model_report_section.html'
    if thermal_path.exists():
        sections.append(thermal_path.read_text())

    greybox_path = OUTPUT_DIR / 'greybox_report_section.html'
    if greybox_path.exists():
        sections.append(greybox_path.read_text())

    heat_pump_path = OUTPUT_DIR / 'heat_pump_model_report_section.html'
    if heat_pump_path.exists():
        sections.append(heat_pump_path.read_text())

    energy_path = OUTPUT_DIR / 'energy_system_model_report_section.html'
    if energy_path.exists():
        sections.append(energy_path.read_text())

    cost_path = OUTPUT_DIR / 'tariff_cost_model_report_section.html'
    if cost_path.exists():
        sections.append(cost_path.read_text())

    # Generate full report
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Phase 3: System Modeling Report</title>
    <meta charset="UTF-8">
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
            color: #333;
        }}
        h1 {{ color: #1a5f7a; border-bottom: 2px solid #1a5f7a; padding-bottom: 10px; }}
        h2 {{ color: #2c3e50; margin-top: 40px; }}
        h3 {{ color: #34495e; }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{ background-color: #f8f9fa; font-weight: 600; }}
        tr:nth-child(even) {{ background-color: #f8f9fa; }}
        figure {{
            margin: 30px 0;
            text-align: center;
        }}
        figure img {{
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
        figcaption {{
            font-style: italic;
            color: #666;
            margin-top: 10px;
        }}
        pre {{
            background-color: #f4f4f4;
            padding: 15px;
            border-radius: 4px;
            overflow-x: auto;
            font-size: 14px;
        }}
        code {{
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: "SF Mono", Monaco, "Courier New", monospace;
        }}
        ul {{ padding-left: 20px; }}
        li {{ margin: 8px 0; }}
        .summary-box {{
            background-color: #e8f4f8;
            border-left: 4px solid #1a5f7a;
            padding: 15px;
            margin: 20px 0;
        }}
        .warning {{
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 20px 0;
        }}
        .table-of-contents {{
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            padding: 20px 30px;
            margin: 30px 0;
        }}
        .table-of-contents h2 {{
            margin-top: 0;
            color: #1a5f7a;
            font-size: 1.3em;
        }}
        .table-of-contents ol {{
            margin: 0;
            padding-left: 20px;
        }}
        .table-of-contents li {{
            margin: 8px 0;
        }}
        .table-of-contents a {{
            color: #2c3e50;
            text-decoration: none;
        }}
        .table-of-contents a:hover {{
            color: #1a5f7a;
            text-decoration: underline;
        }}
        /* MathJax equation styling */
        .MathJax {{ font-size: 1.1em !important; }}
        .equation-box {{
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 4px;
            padding: 15px;
            margin: 15px 0;
            overflow-x: auto;
        }}
    </style>
    <!-- MathJax for LaTeX rendering -->
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <script>
        window.MathJax = {{
            tex: {{
                inlineMath: [['$', '$'], ['\\(', '\\)']],
                displayMath: [['$$', '$$'], ['\\[', '\\]']]
            }}
        }};
    </script>
</head>
<body>
    <h1>Phase 3: System Modeling Report</h1>
    <p><em>Generated: {timestamp}</em></p>

    <div class="summary-box">
        <h3>Executive Summary</h3>
        <p>This report presents the system models developed for heating strategy optimization:</p>
        <ul>
            <li><strong>Thermal Model</strong>: Building time constant ~54-60 hours, indicating good thermal mass</li>
            <li><strong>Heat Pump Model</strong>: COP = 6.52 + 0.13×T_outdoor - 0.10×T_flow (R²=0.95)</li>
            <li><strong>Energy System</strong>: Current self-sufficiency 58%, potential 85% with optimization</li>
            <li><strong>Tariff Cost Model</strong>: High-tariff periods account for ~60% of grid costs; potential 15-20% cost reduction through load shifting</li>
        </ul>
    </div>

    <nav class="table-of-contents">
        <h2>Table of Contents</h2>
        <ol>
            <li><a href="#thermal-model">Thermal Model (Transfer Function)</a></li>
            <li><a href="#greybox-thermal-model">Thermal Model (Grey-Box)</a></li>
            <li><a href="#heat-pump-model">Heat Pump Model</a></li>
            <li><a href="#energy-system-model">Energy System Model</a></li>
            <li><a href="#tariff-cost-model">Tariff Cost Model</a></li>
            <li><a href="#model-assumptions">Model Assumptions and Limitations</a></li>
            <li><a href="#equations">Key Equations</a></li>
            <li><a href="#conclusions">Conclusions and Next Steps</a></li>
        </ol>
    </nav>

    {''.join(sections)}

    <section id="model-assumptions">
    <h2>Model Assumptions and Limitations</h2>

    <h3>Thermal Model Assumptions</h3>
    <table>
        <tr><th>Assumption</th><th>Justification</th><th>Impact if Violated</th></tr>
        <tr><td>Single thermal zone</td><td>Simplifies to one representative temperature</td><td>Under-predicts variation between rooms</td></tr>
        <tr><td>First-order dynamics</td><td>Building has dominant single time constant</td><td>Misses fast/slow thermal responses</td></tr>
        <tr><td>Linear heat loss</td><td>Convective losses dominate</td><td>Underestimates losses at high ΔT</td></tr>
        <tr><td>PV as solar proxy</td><td>PV generation correlates with irradiance</td><td>Imperfect due to panel orientation</td></tr>
    </table>

    <h3>Heat Pump Model Assumptions</h3>
    <table>
        <tr><th>Assumption</th><th>Justification</th><th>Impact if Violated</th></tr>
        <tr><td>Linear COP relationship</td><td>Empirically validated (R²=0.95)</td><td>May miss nonlinearities at extremes</td></tr>
        <tr><td>Daily averaging</td><td>Smooths out cycling effects</td><td>Loses detail on part-load efficiency</td></tr>
        <tr><td>No defrost modeling</td><td>Not explicitly captured</td><td>Underestimates cold-weather losses</td></tr>
    </table>

    <h3>Energy System Model Assumptions</h3>
    <table>
        <tr><th>Assumption</th><th>Justification</th><th>Impact if Violated</th></tr>
        <tr><td>Linear load shifting</td><td>Simple approximation</td><td>May over/underestimate flexibility</td></tr>
        <tr><td>70% grid avoidance</td><td>Conservative estimate for shifted load</td><td>Actual depends on timing</td></tr>
        <tr><td>Battery efficiency constant</td><td>Short-term approximation</td><td>Efficiency varies with SOC, rate</td></tr>
    </table>

    <div class="warning">
        <strong>Data Limitations:</strong> Sensor data covers only 64 days of overlap with energy data.
        Seasonal coverage is limited to autumn/early winter. Models should be validated with
        additional data as it becomes available.
    </div>
    </section>

    <section id="equations">
    <h2>Key Equations</h2>

    <h3>Thermal Model</h3>
    <pre>
Time constant: τ = C/UA ≈ 54-60 hours

Discrete temperature change (per 15 min):
ΔT = a × (T_flow - T_room) - b × (T_room - T_outdoor) + c × PV

Where:
  a = heating coefficient (~0.005 K/15min/K)
  b = loss coefficient (~0.005 K/15min/K)
  c = solar gain coefficient
    </pre>

    <h3>Heat Pump COP Model</h3>
    <pre>
COP = 6.52 + 0.1319 × T_outdoor - 0.1007 × T_flow

Example predictions:
  T_outdoor = 0°C, T_flow = 35°C  →  COP = 3.00
  T_outdoor = 8°C, T_flow = 35°C  →  COP = 4.06
  T_outdoor = 8°C, T_flow = 30°C  →  COP = 4.56
    </pre>

    <h3>Self-Sufficiency</h3>
    <pre>
Self-sufficiency = 1 - (Grid_Import / Total_Consumption)
                 = (Direct_PV + Battery_Discharge) / Total_Consumption

Current: 58.1%
Potential with optimization: 85.3%
    </pre>
    </section>

    <section id="conclusions">
    <h2>Conclusions and Next Steps</h2>

    <h3>Key Model Parameters for Optimization</h3>
    <table>
        <tr><th>Parameter</th><th>Value</th><th>Source</th><th>Optimization Use</th></tr>
        <tr><td>Building time constant</td><td>~54-60 hours</td><td>Thermal model</td><td>Pre-heating timing</td></tr>
        <tr><td>COP sensitivity to outdoor temp</td><td>+0.13 per °C</td><td>Heat pump model</td><td>Schedule heating for warm hours</td></tr>
        <tr><td>COP sensitivity to flow temp</td><td>-0.10 per °C</td><td>Heat pump model</td><td>Reduce curve rise when possible</td></tr>
        <tr><td>Battery round-trip efficiency</td><td>83.7%</td><td>Energy system model</td><td>Prefer direct use over storage</td></tr>
        <tr><td>Peak PV hours</td><td>10:00-16:00</td><td>Energy system model</td><td>Target heating to these hours</td></tr>
    </table>

    <h3>Optimization Recommendations</h3>
    <ol>
        <li><strong>Timing Strategy</strong>: Shift heating to solar hours (10:00-16:00) to maximize COP
            (warmer outdoor temps) and direct PV consumption. Running at midday vs midnight
            could improve COP by 0.5-1.0 (15-25% more efficient).</li>
        <li><strong>Flow Temperature</strong>: Reduce flow temps where possible to improve COP.
            Each -1°C improves COP by ~0.10. Reducing curve rise by 0.1 saves ~0.8°C flow temp.</li>
        <li><strong>Pre-heating</strong>: With ~54h time constant, building responds slowly.
            Start comfort mode 2-3 hours before needed for gradual warm-up during solar hours.</li>
        <li><strong>Buffer Tank</strong>: Charge during peak PV (12:00-15:00) to store thermal energy
            for evening heating. Buffer can bridge 2-4 hours of heating demand.</li>
        <li><strong>Evening Setback</strong>: Can start eco mode 1-2 hours before bedtime.
            Room temp drops only ~0.5-1°C before sleep due to high thermal mass.</li>
    </ol>

    <h3>Phase 4: Optimization Strategy Development</h3>
    <p>The models developed here feed into Phase 4, where we will:</p>
    <ul>
        <li>Develop rule-based heuristics using the model parameters</li>
        <li>Quantify expected energy savings from each strategy</li>
        <li>Prepare parameter sets for the randomized intervention study (Phase 5)</li>
    </ul>

    <p><strong>For detailed model documentation, see:</strong> <code>docs/phase3_models.md</code></p>
    </section>

</body>
</html>
"""

    report_path = OUTPUT_DIR / 'phase3_report.html'
    report_path.write_text(html)
    print(f"  Saved: {report_path.name}")


def main():
    """Run all Phase 3 modeling steps."""
    print("="*60)
    print("Phase 3: System Modeling")
    print("="*60)

    success_count = 0

    # Run each modeling script
    scripts = [
        '01_thermal_model.py',
        '01b_greybox_thermal_model.py',
        '02_heat_pump_model.py',
        '03_energy_system_model.py',
        '04_tariff_cost_model.py'
    ]

    for script in scripts:
        if run_script(script):
            success_count += 1
        else:
            print(f"  WARNING: {script} failed")

    # Generate combined report
    generate_html_report()

    # Final summary
    print("\n" + "="*60)
    print("PHASE 3 COMPLETE")
    print("="*60)
    print(f"\nScripts completed: {success_count}/{len(scripts)}")
    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print("\nKey outputs:")
    print("  - fig18_thermal_model.png")
    print("  - fig18b_greybox_model.png")
    print("  - fig19_heat_pump_model.png")
    print("  - fig20_energy_system_model.png")
    print("  - fig21_tariff_cost_model.png")
    print("  - phase3_report.html")


if __name__ == '__main__':
    main()
