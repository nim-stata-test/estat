#!/usr/bin/env python3
"""
Run Pareto Optimization for Heating Strategies

Wrapper script for multi-objective optimization using NSGA-II.
See 04_pareto_optimization.py for implementation details.

Usage:
    # Standard run (200 generations, 100 population)
    python src/phase4/run_pareto.py

    # Quick test run
    python src/phase4/run_pareto.py -g 50 -p 50

    # Warm start from previous archive
    python src/phase4/run_pareto.py --warm-start output/phase4/pareto_archive.json

    # Custom settings
    python src/phase4/run_pareto.py -g 300 -p 150 -n 15 --seed 123
"""

import sys
import importlib.util
from pathlib import Path

# Load the optimization module (name starts with number, can't use regular import)
SCRIPT_PATH = Path(__file__).parent / '04_pareto_optimization.py'

spec = importlib.util.spec_from_file_location("pareto_optimization", SCRIPT_PATH)
pareto_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pareto_module)

if __name__ == '__main__':
    sys.exit(pareto_module.main())
