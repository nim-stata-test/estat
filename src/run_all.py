#!/usr/bin/env python3
"""
Master script for running all ESTAT preprocessing and analysis phases.

Usage:
    python src/run_all.py              # Run all phases
    python src/run_all.py --phase 1    # Run only Phase 1 (Preprocessing)
    python src/run_all.py --phase 2    # Run only Phase 2 (EDA)
    python src/run_all.py --phase 3    # Run only Phase 3 (System Modeling)
    python src/run_all.py --phase 4    # Run only Phase 4 (Optimization)
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

    # Save and clear sys.argv to prevent argument conflicts with modules
    # that have their own argparse (e.g., Pareto optimization)
    saved_argv = sys.argv.copy()
    sys.argv = [str(script_path)]

    try:
        spec = importlib.util.spec_from_file_location(f"phase{phase}_step{step}", script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.argv = saved_argv


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


def run_phase1_step4():
    """Phase 1, Step 4: Preprocess tariff data."""
    print("\n" + "=" * 70)
    print("RUNNING: Phase 1, Step 4 - Preprocess Electricity Tariffs")
    print("=" * 70)
    module = load_module(1, 4)
    module.main()


def run_phase1_step5():
    """Phase 1, Step 5: Generate HTML report."""
    print("\n" + "=" * 70)
    print("RUNNING: Phase 1, Step 5 - Generate HTML Report")
    print("=" * 70)
    # Import and call the report generation function
    phase1_dir = SRC_DIR / "phase1"
    spec = importlib.util.spec_from_file_location(
        "run_preprocessing", phase1_dir / "run_preprocessing.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.regenerate_report_only()


def run_phase2_step1():
    """Phase 2, Step 1: Exploratory Data Analysis."""
    print("\n" + "=" * 70)
    print("RUNNING: Phase 2, Step 1 - Exploratory Data Analysis")
    print("=" * 70)
    module = load_module(2, 1)
    module.main()


def run_phase2_step2():
    """Phase 2, Step 2: Battery Degradation Analysis."""
    print("\n" + "=" * 70)
    print("RUNNING: Phase 2, Step 2 - Battery Degradation Analysis")
    print("=" * 70)
    module = load_module(2, 2)
    module.main()


def run_phase2_step3():
    """Phase 2, Step 3: Heating Curve Analysis."""
    print("\n" + "=" * 70)
    print("RUNNING: Phase 2, Step 3 - Heating Curve Analysis")
    print("=" * 70)
    module = load_module(2, 3)
    module.main()


def run_phase2_step4():
    """Phase 2, Step 4: Tariff Analysis."""
    print("\n" + "=" * 70)
    print("RUNNING: Phase 2, Step 4 - Tariff Analysis")
    print("=" * 70)
    module = load_module(2, 4)
    module.main()


def run_phase2_step5():
    """Phase 2, Step 5: Weighted Temperature Analysis."""
    print("\n" + "=" * 70)
    print("RUNNING: Phase 2, Step 5 - Weighted Temperature Analysis")
    print("=" * 70)
    module = load_module(2, 5)
    module.main()


def run_phase3_step1():
    """Phase 3, Step 1: Thermal Model."""
    print("\n" + "=" * 70)
    print("RUNNING: Phase 3, Step 1 - Building Thermal Model")
    print("=" * 70)
    module = load_module(3, 1)
    module.main()


def run_phase3_step1b():
    """Phase 3, Step 1b: Grey-Box Thermal Model."""
    print("\n" + "=" * 70)
    print("RUNNING: Phase 3, Step 1b - Grey-Box Thermal Model")
    print("=" * 70)
    script_path = SRC_DIR / "phase3" / "01b_greybox_thermal_model.py"
    spec = importlib.util.spec_from_file_location("phase3_step1b", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
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


def run_phase3_step4():
    """Phase 3, Step 4: Tariff Cost Model."""
    print("\n" + "=" * 70)
    print("RUNNING: Phase 3, Step 4 - Tariff Cost Model")
    print("=" * 70)
    module = load_module(3, 4)
    module.main()


def run_phase3_step5():
    """Phase 3, Step 5: Generate HTML report."""
    print("\n" + "=" * 70)
    print("RUNNING: Phase 3, Step 5 - Generate HTML Report")
    print("=" * 70)
    # Import and call the report generation function
    phase3_dir = SRC_DIR / "phase3"
    spec = importlib.util.spec_from_file_location(
        "run_phase3", phase3_dir / "run_phase3.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.generate_html_report()


def run_phase4_step1():
    """Phase 4, Step 1: Rule-Based Strategies."""
    print("\n" + "=" * 70)
    print("RUNNING: Phase 4, Step 1 - Rule-Based Optimization Strategies")
    print("=" * 70)
    module = load_module(4, 1)
    module.main()


def run_phase4_step2():
    """Phase 4, Step 2: Strategy Simulation."""
    print("\n" + "=" * 70)
    print("RUNNING: Phase 4, Step 2 - Strategy Simulation")
    print("=" * 70)
    module = load_module(4, 2)
    module.main()


def run_phase4_step3():
    """Phase 4, Step 3: Parameter Set Generation."""
    print("\n" + "=" * 70)
    print("RUNNING: Phase 4, Step 3 - Parameter Set Generation")
    print("=" * 70)
    module = load_module(4, 3)
    module.main()


def run_phase4_step4():
    """Phase 4, Step 4: Pareto Optimization."""
    print("\n" + "=" * 70)
    print("RUNNING: Phase 4, Step 4 - Pareto Optimization")
    print("=" * 70)
    module = load_module(4, 4)
    # Clear sys.argv before calling main() to avoid argparse conflicts
    saved_argv = sys.argv.copy()
    sys.argv = [sys.argv[0]]  # Keep only script name
    try:
        # Run with default settings (10 generations, warm-start from archive)
        module.main()
    finally:
        sys.argv = saved_argv


def run_phase4_step5():
    """Phase 4, Step 5: Strategy Evaluation."""
    print("\n" + "=" * 70)
    print("RUNNING: Phase 4, Step 5 - Strategy Evaluation")
    print("=" * 70)
    module = load_module(4, 5)
    module.main()


def run_phase4_step6():
    """Phase 4, Step 6: Strategy Detailed Analysis."""
    print("\n" + "=" * 70)
    print("RUNNING: Phase 4, Step 6 - Strategy Detailed Analysis")
    print("=" * 70)
    module = load_module(4, 6)
    module.main()


def run_phase4_step7():
    """Phase 4, Step 7: Pareto Animation."""
    print("\n" + "=" * 70)
    print("RUNNING: Phase 4, Step 7 - Pareto Animation")
    print("=" * 70)
    module = load_module(4, 7)
    module.main()


# Define all phases and steps
PHASES = {
    1: {
        1: ("Preprocess Energy Balance", run_phase1_step1),
        2: ("Preprocess Sensors", run_phase1_step2),
        3: ("Integrate Data", run_phase1_step3),
        4: ("Preprocess Tariffs", run_phase1_step4),
        5: ("Generate HTML Report", run_phase1_step5),
    },
    2: {
        1: ("Exploratory Data Analysis", run_phase2_step1),
        2: ("Battery Degradation", run_phase2_step2),
        3: ("Heating Curve Analysis", run_phase2_step3),
        4: ("Tariff Analysis", run_phase2_step4),
        5: ("Weighted Temperature", run_phase2_step5),
    },
    3: {
        1: ("Thermal Model (Transfer Function)", run_phase3_step1),
        "1b": ("Thermal Model (Grey-Box)", run_phase3_step1b),
        2: ("Heat Pump Model", run_phase3_step2),
        3: ("Energy System Model", run_phase3_step3),
        4: ("Tariff Cost Model", run_phase3_step4),
        5: ("Generate HTML Report", run_phase3_step5),
    },
    4: {
        1: ("Rule-Based Strategies", run_phase4_step1),
        2: ("Strategy Simulation", run_phase4_step2),
        3: ("Parameter Sets", run_phase4_step3),
        4: ("Pareto Optimization", run_phase4_step4),
        5: ("Strategy Evaluation", run_phase4_step5),
        6: ("Strategy Detailed Analysis", run_phase4_step6),
        7: ("Pareto Animation", run_phase4_step7),
    },
}


def run_step(phase: int, step):
    """Run a specific step.

    Args:
        phase: Phase number (int)
        step: Step identifier (int or str like "1b")
    """
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


def step_sort_key(step):
    """Sort key for steps that handles mixed int/str like 1, '1b', 2."""
    if isinstance(step, int):
        return (step, '')
    # String step like '1b' -> (1, 'b')
    import re
    match = re.match(r'(\d+)(\w*)', str(step))
    if match:
        return (int(match.group(1)), match.group(2))
    return (999, str(step))


def run_phase(phase: int, rerun_optimization: bool = False):
    """Run all steps in a phase."""
    if phase not in PHASES:
        print(f"Error: Phase {phase} not found")
        return False

    print(f"\n{'#' * 70}")
    print(f"# PHASE {phase}")
    print(f"{'#' * 70}")

    # Steps to skip unless --rerun_optimization is set
    skip_steps = {4: [4, 5, 6, 7]} if not rerun_optimization else {}

    success = True
    for step in sorted(PHASES[phase].keys(), key=step_sort_key):
        if phase in skip_steps and step in skip_steps[phase]:
            name, _ = PHASES[phase][step]
            print(f"\n[SKIP] Phase {phase}, Step {step} ({name}) - use --rerun_optimization to include")
            continue
        if not run_step(phase, step):
            success = False
            break  # Stop on first failure

    return success


def run_all(rerun_optimization: bool = False):
    """Run all phases and steps."""
    print("=" * 70)
    print("ESTAT - Full Pipeline Execution")
    print("=" * 70)

    start = time.time()

    for phase in sorted(PHASES.keys()):
        if not run_phase(phase, rerun_optimization=rerun_optimization):
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
  python src/run_all.py              # Run all phases (skips Pareto optimization)
  python src/run_all.py --phase 1    # Run Phase 1 only (Preprocessing)
  python src/run_all.py --phase 2    # Run Phase 2 only (EDA)
  python src/run_all.py --phase 3    # Run Phase 3 only (System Modeling)
  python src/run_all.py --phase 4    # Run Phase 4 only (skips Pareto)
  python src/run_all.py --phase 4 --rerun_optimization  # Phase 4 with Pareto
  python src/run_all.py --step 1.2   # Run Phase 1, Step 2 only
  python src/run_all.py --step 3.1b  # Run Grey-Box Thermal Model only
  python src/run_all.py --step 4.4   # Run Pareto optimization only
  python src/run_all.py --list       # List all available phases and steps
        """
    )

    parser.add_argument("--phase", type=int, help="Run specific phase (1, 2, ...)")
    parser.add_argument("--step", type=str, help="Run specific step (e.g., '1.2' for Phase 1 Step 2)")
    parser.add_argument("--list", action="store_true", help="List all available phases and steps")
    parser.add_argument("--rerun_optimization", action="store_true",
                        help="Include Pareto optimization (steps 4.4, 4.5) which are skipped by default")

    args = parser.parse_args()

    if args.list:
        print("\nAvailable phases and steps:")
        print("-" * 50)
        for phase in sorted(PHASES.keys()):
            print(f"\nPhase {phase}:")
            for step in sorted(PHASES[phase].keys(), key=step_sort_key):
                name, _ = PHASES[phase][step]
                print(f"  {phase}.{step}: {name}")
        return

    if args.step:
        try:
            parts = args.step.split(".")
            phase = int(parts[0])
            # Step can be int (e.g., "2") or string (e.g., "1b")
            step_str = parts[1]
            step = int(step_str) if step_str.isdigit() else step_str
            run_step(phase, step)
        except (ValueError, IndexError):
            print(f"Error: Invalid step format '{args.step}'. Use 'phase.step' (e.g., '1.2' or '3.1b')")
            sys.exit(1)
    elif args.phase:
        run_phase(args.phase, rerun_optimization=args.rerun_optimization)
    else:
        run_all(rerun_optimization=args.rerun_optimization)


if __name__ == "__main__":
    main()
