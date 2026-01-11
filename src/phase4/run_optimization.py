#!/usr/bin/env python3
"""
Phase 4: Optimization Strategy Development

Runs all Phase 4 steps and generates a combined HTML report.

Steps:
1. Rule-based strategy definitions
2. Strategy simulation on historical data
3. Parameter set generation for Phase 5

Usage:
    python src/phase4/run_optimization.py
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'output' / 'phase4'
OUTPUT_DIR.mkdir(exist_ok=True)


def run_script(script_name: str) -> bool:
    """Run a Phase 4 script and return success status."""
    script_path = Path(__file__).parent / script_name
    print(f"\n{'='*60}")
    print(f"Running {script_name}")
    print('='*60)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(PROJECT_ROOT)
    )

    return result.returncode == 0


def generate_html_report():
    """Combine individual report sections into full HTML report."""
    print(f"\n{'='*60}")
    print("Generating combined HTML report")
    print('='*60)

    # Read individual sections
    sections = []

    section_files = [
        'strategies_report_section.html',
        'simulation_report_section.html',
        'parameter_sets_report_section.html',
        'pareto_report_section.html',  # Multi-objective Pareto optimization
        'strategy_evaluation_report.html',  # Comfort violation analysis
    ]

    for section_file in section_files:
        section_path = OUTPUT_DIR / section_file
        if section_path.exists():
            sections.append(section_path.read_text())
            print(f"  Added: {section_file}")
        else:
            print(f"  Warning: {section_file} not found")

    # Build HTML structure
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Phase 4: Optimization Strategy Development Report</title>
    <meta charset="UTF-8">
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
        }}
        h1 {{
            color: #1a5f7a;
            border-bottom: 3px solid #1a5f7a;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #2c3e50;
            border-bottom: 1px solid #ccc;
            padding-bottom: 5px;
            margin-top: 40px;
        }}
        h3 {{
            color: #34495e;
        }}
        h4 {{
            color: #1a5f7a;
            margin-top: 20px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 15px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 10px;
            text-align: left;
        }}
        th {{
            background-color: #f5f5f5;
        }}
        tr:nth-child(even) {{
            background-color: #fafafa;
        }}
        figure {{
            margin: 20px 0;
            text-align: center;
        }}
        figcaption {{
            font-style: italic;
            color: #666;
            margin-top: 10px;
        }}
        img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .summary-box {{
            background-color: #e8f4f8;
            border-left: 4px solid #1a5f7a;
            padding: 15px;
            margin: 20px 0;
        }}
        .toc {{
            background-color: #f8f9fa;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 20px 25px;
            margin: 25px 0;
        }}
        .toc h3 {{
            margin-top: 0;
            margin-bottom: 15px;
            color: #1a5f7a;
        }}
        .toc ul {{
            list-style: none;
            padding-left: 0;
            margin: 0;
        }}
        .toc > ul > li {{
            margin: 8px 0;
        }}
        .toc a {{
            color: #2c3e50;
            text-decoration: none;
            border-bottom: 1px dotted #ccc;
        }}
        .toc a:hover {{
            color: #1a5f7a;
            border-bottom-color: #1a5f7a;
        }}
        .strategy-card, .parameter-card {{
            background-color: #fff;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            margin: 15px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        pre {{
            background-color: #f5f5f5;
            padding: 15px;
            border-radius: 4px;
            overflow-x: auto;
        }}
        ul, ol {{
            margin: 10px 0;
            padding-left: 25px;
        }}
        li {{
            margin: 5px 0;
        }}
        em {{
            color: #666;
        }}
        strong {{
            color: #1a5f7a;
        }}
    </style>
</head>
<body>
    <h1>Phase 4: Optimization Strategy Development Report</h1>
    <p><em>Generated: {timestamp}</em></p>

    <div class="summary-box">
        <h3>Executive Summary</h3>
        <p>This report presents heating optimization strategies developed using Phase 3 model parameters
        and refined through multi-objective Pareto optimization (NSGA-II). Strategies were validated
        through simulation on 64 days of historical data and parameter sets are ready for
        Phase 5 randomized intervention study (Winter 2027-2028).</p>

        <p><strong>Key Findings from Pareto Optimization:</strong></p>
        <ul>
            <li><strong>Eco setpoint has minimal daytime impact</strong>: -0.09°C per 1°C change allows aggressive setback (12-14°C)</li>
            <li><strong>Grid-Minimal strategy</strong>: eco=13.6°C, curve_rise=0.81 → 1069 kWh grid (-11% vs baseline)</li>
            <li><strong>Balanced strategy</strong>: eco=12.5°C, curve_rise=0.98 → 1104 kWh grid (-8% vs baseline)</li>
            <li><strong>Optimal comfort window</strong>: All Pareto solutions converge to ~4h solar-aligned window (11:00-16:00)</li>
            <li><strong>21 Pareto-optimal solutions</strong> available for strategy selection</li>
        </ul>
    </div>

    <nav class="toc">
        <h3>Table of Contents</h3>
        <ul>
            <li><a href="#rule-based-strategies">4.1 Rule-Based Optimization Strategies</a></li>
            <li><a href="#strategy-simulation">4.2 Strategy Simulation</a></li>
            <li><a href="#parameter-sets">4.3 Parameter Sets for Phase 5</a></li>
            <li><a href="#pareto-optimization">4.4 Pareto Multi-Objective Optimization</a></li>
            <li><a href="#strategy-evaluation">4.5 Strategy Evaluation and Comfort Analysis</a></li>
            <li><a href="#next-steps">4.6 Next Steps: Phase 5 Preparation</a></li>
        </ul>
    </nav>

    {''.join(sections)}

    <section id="next-steps">
    <h2>4.6 Next Steps: Phase 5 Preparation</h2>

    <h3>Timeline</h3>
    <ul>
        <li><strong>2026-2027</strong>: Continue sensor data collection, refine models with additional winter data</li>
        <li><strong>Fall 2027</strong>: Finalize randomization schedule, prepare equipment</li>
        <li><strong>Nov 2027 - Mar 2028</strong>: Execute randomized intervention study</li>
    </ul>

    <h3>Required Actions</h3>
    <ol>
        <li>Verify heat pump interface access for curve_rise adjustment</li>
        <li>Set up automated parameter switching (if possible)</li>
        <li>Create monitoring dashboard for real-time comfort tracking</li>
        <li>Generate randomized block schedule</li>
        <li>Prepare data collection protocol</li>
    </ol>

    <h3>Risk Mitigation</h3>
    <ul>
        <li><strong>Comfort violations</strong>: Set hard limits (16°C min) with automatic override</li>
        <li><strong>Equipment issues</strong>: Document fallback to baseline settings</li>
        <li><strong>Weather variability</strong>: Use heating degree days for normalization</li>
        <li><strong>Manual overrides</strong>: Log all interventions for analysis adjustment</li>
    </ul>
    </section>

    <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ccc; color: #666;">
        <p>ESTAT Project - Phase 4 Optimization Report</p>
        <p>Generated by Phase 4 analysis pipeline</p>
    </footer>
</body>
</html>"""

    report_path = OUTPUT_DIR / 'phase4_report.html'
    report_path.write_text(html)
    print(f"  Saved: {report_path.name}")


def main():
    """Run all Phase 4 steps."""
    print("="*60)
    print("Phase 4: Optimization Strategy Development")
    print("="*60)

    scripts = [
        '01_rule_based_strategies.py',
        '02_strategy_simulation.py',
        '03_parameter_sets.py',
    ]

    success_count = 0

    for script in scripts:
        if run_script(script):
            success_count += 1
        else:
            print(f"  WARNING: {script} failed")

    # Generate combined report
    generate_html_report()

    # Summary
    print("\n" + "="*60)
    print("PHASE 4 COMPLETE")
    print("="*60)
    print(f"Scripts completed: {success_count}/{len(scripts)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("\nKey outputs:")
    print("  - phase4_report.html (combined report)")
    print("  - selected_strategies.json (Pareto-optimized strategies)")
    print("  - pareto_archive.json (full Pareto front for warm-starting)")
    print("  - phase5_parameter_sets.json (intervention parameters)")
    print("  - phase5_predictions.json (testable predictions)")
    print("\nFigures:")
    print("  - fig21_strategy_comparison.png")
    print("  - fig22_simulation_results.png")
    print("  - fig23_parameter_space.png")
    print("  - fig24_pareto_front.png (Pareto optimization)")
    print("  - fig25_pareto_strategy_comparison.png (Pareto strategies)")
    print("  - fig27_strategy_temperature_predictions.png (Comfort evaluation)")


if __name__ == '__main__':
    main()
