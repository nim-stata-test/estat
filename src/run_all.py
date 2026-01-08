#!/usr/bin/env python3
"""
Master script for running all ESTAT preprocessing and analysis phases.

Usage:
    python src/run_all.py              # Run all phases
    python src/run_all.py --phase 1    # Run only Phase 1 (Preprocessing)
    python src/run_all.py --phase 2    # Run only Phase 2 (EDA)
    python src/run_all.py --phase 3    # Run only Phase 3 (System Modeling)
    python src/run_all.py --step 1.2   # Run specific step (Phase 1, Step 2)
"""

import argparse
import sys
import time
from pathlib import Path
import importlib.util

PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"


def load_module(phase: int, step: int):
    """Dynamically load a phase module by its numbered filename."""
    phase_dir = SRC_DIR / f"phase{phase}"
    # Find the script with matching step number
    pattern = f"{step:02d}_*.py"
    matches = list(phase_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No script found matching {phase_dir}/{pattern}")

    script_path = matches[0]
    spec = importlib.util.spec_from_file_location(f"phase{phase}_step{step}", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_phase1_step1():
    """Phase 1, Step 1: Preprocess energy balance data."""
    print("\n" + "=" * 70)
    print("RUNNING: Phase 1, Step 1 - Preprocess Energy Balance Data")
    print("=" * 70)
    module = load_module(1, 1)
    module.main()


def run_phase1_step2():
    """Phase 1, Step 2: Preprocess sensor data."""
    print("\n" + "=" * 70)
    print("RUNNING: Phase 1, Step 2 - Preprocess Sensor Data")
    print("=" * 70)
    module = load_module(1, 2)
    module.main()


def run_phase1_step3():
    """Phase 1, Step 3: Integrate datasets."""
    print("\n" + "=" * 70)
    print("RUNNING: Phase 1, Step 3 - Integrate Data")
    print("=" * 70)
    module = load_module(1, 3)
    module.main()


def run_phase2_step1():
    """Phase 2, Step 1: Exploratory Data Analysis."""
    print("\n" + "=" * 70)
    print("RUNNING: Phase 2, Step 1 - Exploratory Data Analysis")
    print("=" * 70)
    module = load_module(2, 1)
    module.main()


def run_phase3_step1():
    """Phase 3, Step 1: Thermal Model."""
    print("\n" + "=" * 70)
    print("RUNNING: Phase 3, Step 1 - Building Thermal Model")
    print("=" * 70)
    module = load_module(3, 1)
    module.main()


def run_phase3_step2():
    """Phase 3, Step 2: Heat Pump Model."""
    print("\n" + "=" * 70)
    print("RUNNING: Phase 3, Step 2 - Heat Pump Model")
    print("=" * 70)
    module = load_module(3, 2)
    module.main()


def run_phase3_step3():
    """Phase 3, Step 3: Energy System Model."""
    print("\n" + "=" * 70)
    print("RUNNING: Phase 3, Step 3 - Energy System Model")
    print("=" * 70)
    module = load_module(3, 3)
    module.main()


# Define all phases and steps
PHASES = {
    1: {
        1: ("Preprocess Energy Balance", run_phase1_step1),
        2: ("Preprocess Sensors", run_phase1_step2),
        3: ("Integrate Data", run_phase1_step3),
    },
    2: {
        1: ("Exploratory Data Analysis", run_phase2_step1),
    },
    3: {
        1: ("Thermal Model", run_phase3_step1),
        2: ("Heat Pump Model", run_phase3_step2),
        3: ("Energy System Model", run_phase3_step3),
    },
}


def run_step(phase: int, step: int):
    """Run a specific step."""
    if phase not in PHASES:
        print(f"Error: Phase {phase} not found")
        return False
    if step not in PHASES[phase]:
        print(f"Error: Step {step} not found in Phase {phase}")
        return False

    name, func = PHASES[phase][step]
    start = time.time()
    try:
        func()
        elapsed = time.time() - start
        print(f"\n[OK] Phase {phase}, Step {step} ({name}) completed in {elapsed:.1f}s")
        return True
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n[ERROR] Phase {phase}, Step {step} ({name}) failed after {elapsed:.1f}s: {e}")
        return False


def run_phase(phase: int):
    """Run all steps in a phase."""
    if phase not in PHASES:
        print(f"Error: Phase {phase} not found")
        return False

    print(f"\n{'#' * 70}")
    print(f"# PHASE {phase}")
    print(f"{'#' * 70}")

    success = True
    for step in sorted(PHASES[phase].keys()):
        if not run_step(phase, step):
            success = False
            break  # Stop on first failure

    return success


def run_all():
    """Run all phases and steps."""
    print("=" * 70)
    print("ESTAT - Full Pipeline Execution")
    print("=" * 70)

    start = time.time()

    for phase in sorted(PHASES.keys()):
        if not run_phase(phase):
            print(f"\nPipeline stopped due to error in Phase {phase}")
            return False

    elapsed = time.time() - start
    print(f"\n{'=' * 70}")
    print(f"COMPLETE - Total execution time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'=' * 70}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run ESTAT preprocessing and analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/run_all.py              # Run all phases
  python src/run_all.py --phase 1    # Run Phase 1 only (Preprocessing)
  python src/run_all.py --phase 2    # Run Phase 2 only (EDA)
  python src/run_all.py --phase 3    # Run Phase 3 only (System Modeling)
  python src/run_all.py --step 1.2   # Run Phase 1, Step 2 only
  python src/run_all.py --step 3.2   # Run Phase 3, Step 2 only (Heat Pump Model)
  python src/run_all.py --list       # List all available phases and steps
        """
    )

    parser.add_argument("--phase", type=int, help="Run specific phase (1, 2, ...)")
    parser.add_argument("--step", type=str, help="Run specific step (e.g., '1.2' for Phase 1 Step 2)")
    parser.add_argument("--list", action="store_true", help="List all available phases and steps")

    args = parser.parse_args()

    if args.list:
        print("\nAvailable phases and steps:")
        print("-" * 50)
        for phase in sorted(PHASES.keys()):
            print(f"\nPhase {phase}:")
            for step, (name, _) in sorted(PHASES[phase].items()):
                print(f"  {phase}.{step}: {name}")
        return

    if args.step:
        try:
            phase, step = map(int, args.step.split("."))
            run_step(phase, step)
        except ValueError:
            print(f"Error: Invalid step format '{args.step}'. Use 'phase.step' (e.g., '1.2')")
            sys.exit(1)
    elif args.phase:
        run_phase(args.phase)
    else:
        run_all()


if __name__ == "__main__":
    main()
