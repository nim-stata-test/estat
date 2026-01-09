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
        <p>This report presents three heating optimization strategies developed using Phase 3 model parameters.
        Strategies were validated through simulation on 64 days of historical data and parameter sets are
        ready for Phase 5 randomized intervention study (Winter 2027-2028).</p>

        <p><strong>Key Findings:</strong></p>
        <ul>
            <li>Energy-Optimized strategy expected to improve self-sufficiency by +10pp and COP by +0.5</li>
            <li>Aggressive Solar strategy targets 85% self-sufficiency (+27pp) with 17-23°C comfort band</li>
            <li>Schedule shifting from 06:30-20:00 to 10:00-17:00/18:00 aligns heating with PV availability</li>
            <li>Reducing curve_rise from 1.08 to 0.95-0.98 improves COP through lower flow temperatures</li>
        </ul>
    </div>

    {''.join(sections)}

    <section id="next-steps">
    <h2>Next Steps: Phase 5 Preparation</h2>

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

    report_path = OUTPUT_DIR / 'phase4_optimization_report.html'
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
    print("  - phase4_optimization_report.html (combined report)")
    print("  - phase5_parameter_sets.json (intervention parameters)")
    print("  - phase5_predictions.json (testable predictions)")
    print("  - phase5_implementation_checklist.md (protocol)")
    print("\nFigures:")
    print("  - fig16_strategy_comparison.png")
    print("  - fig17_simulation_results.png")
    print("  - fig18_parameter_space.png")


if __name__ == '__main__':
    main()
