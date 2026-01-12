#!/usr/bin/env python3
"""
Phase 5 Pilot: Main Runner

Generates the T_HK2-targeted design and pilot schedule in one go.

The design maximizes T_HK2 (flow temperature) spread to learn the thermal
response function: T_indoor = f(T_HK2 history, T_outdoor history).

Usage:
    python src/phase5_pilot/run_pilot.py
    python src/phase5_pilot/run_pilot.py --start 2026-01-13 --seed 42

Outputs:
    output/phase5_pilot/thk2_design.csv
    output/phase5_pilot/thk2_design.json
    output/phase5_pilot/pilot_schedule.csv
    output/phase5_pilot/pilot_schedule.json
    output/phase5_pilot/pilot_protocol.html
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
SRC_DIR = Path(__file__).parent


def run_step(script_name: str, args: list) -> int:
    """Run a pilot script with arguments."""
    script_path = SRC_DIR / script_name
    cmd = [sys.executable, str(script_path)] + args

    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print(f"{'='*60}")

    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description='Run Phase 5 Pilot: Generate T_HK2-targeted design and schedule',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python src/phase5_pilot/run_pilot.py
    python src/phase5_pilot/run_pilot.py --start 2026-01-13 --seed 123
    python src/phase5_pilot/run_pilot.py --ref-outdoor 3  # Colder reference temp
        """,
    )
    parser.add_argument(
        '--start',
        type=str,
        default='2026-01-13',
        help='Start date (YYYY-MM-DD), default: 2026-01-13',
    )
    parser.add_argument(
        '--ref-outdoor',
        type=float,
        default=5.0,
        help='Reference outdoor temp for T_HK2 calculation (default: 5°C)',
    )
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=42,
        help='Random seed (default: 42)',
    )
    parser.add_argument(
        '--design-only',
        action='store_true',
        help='Only generate design, skip schedule',
    )
    parser.add_argument(
        '--schedule-only',
        action='store_true',
        help='Only generate schedule (assumes design exists)',
    )

    args = parser.parse_args()

    print(f"Phase 5 Pilot: T_HK2-Targeted Parameter Exploration")
    print(f"Start date: {args.start}")
    print(f"Reference outdoor temp: {args.ref_outdoor}°C")
    print(f"Seed: {args.seed}")

    # Step 1: Generate T_HK2-targeted design
    if not args.schedule_only:
        rc = run_step(
            '01_generate_thk2_design.py',
            ['--ref-outdoor', str(args.ref_outdoor)]
        )
        if rc != 0:
            print(f"\nError: Design generation failed with code {rc}")
            return rc

    # Step 2: Generate schedule
    if not args.design_only:
        rc = run_step(
            '02_generate_pilot_schedule.py',
            ['--start', args.start, '--seed', str(args.seed)]
        )
        if rc != 0:
            print(f"\nError: Schedule generation failed with code {rc}")
            return rc

    # Print summary
    print(f"\n{'='*60}")
    print("PILOT GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"\nOutputs in output/phase5_pilot/:")
    print(f"  - thk2_design.csv         (T_HK2-targeted design matrix)")
    print(f"  - pilot_schedule.csv      (dated schedule)")
    print(f"  - pilot_protocol.html     (human-readable protocol)")
    print(f"\nNext steps:")
    print(f"  1. Review pilot_protocol.html")
    print(f"  2. Start Block 1 on {args.start}")
    print(f"  3. Update parameters weekly per schedule")
    print(f"  4. Run analysis after each block with 03_pilot_analysis.py")

    return 0


if __name__ == '__main__':
    exit(main())
