#!/usr/bin/env python3
"""
Phase 5 Pilot: Generate T_HK2-Targeted Design

Generates a design that maximizes T_HK2 (flow temperature) spread to learn
the thermal response function: T_indoor = f(T_HK2 history, T_outdoor history).

Key insight: The heating curve model is deterministic and well-understood:
    T_HK2 = T_setpoint + curve_rise × (T_ref - T_outdoor)

What we DON'T understand is the thermal response - how T_indoor depends on
T_HK2 history. Therefore, we should maximize T_HK2 spread, not raw parameter spread.

Usage:
    python src/phase5_pilot/01_generate_thk2_design.py
    python src/phase5_pilot/01_generate_thk2_design.py --ref-outdoor 5

Outputs:
    output/phase5_pilot/thk2_design.csv - Design matrix with T_HK2 values
    output/phase5_pilot/thk2_design.json - Machine-readable format
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'output' / 'phase5_pilot'
PHASE2_DIR = PROJECT_ROOT / 'output' / 'phase2'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Default reference outdoor temperature for T_HK2 calculation
DEFAULT_REF_OUTDOOR = 5.0  # °C, typical winter temperature

# Parameter bounds (for validation)
PARAM_BOUNDS = {
    'comfort_setpoint': (19.0, 22.0),
    'eco_setpoint': (14.0, 19.0),
    'curve_rise': (0.80, 1.20),
    'comfort_hours': (8.0, 16.0),
}

# Safety constraints
SAFETY = {
    'min_temp_floor': 17.0,
}


def load_heating_curve_params() -> dict:
    """Load T_ref values from Phase 2 heating curve analysis."""
    params_path = PHASE2_DIR / 'heating_curve_params.json'
    if not params_path.exists():
        print(f"Warning: {params_path} not found, using defaults")
        return {
            't_ref_comfort': 21.32,
            't_ref_eco': 19.16,
        }

    with open(params_path) as f:
        params = json.load(f)

    print(f"Loaded heating curve params:")
    print(f"  T_ref_comfort: {params['t_ref_comfort']:.2f}°C")
    print(f"  T_ref_eco: {params['t_ref_eco']:.2f}°C")

    return params


def calc_thk2(setpoint: float, curve_rise: float, t_ref: float, t_outdoor: float) -> float:
    """Calculate target flow temperature T_HK2."""
    return setpoint + curve_rise * (t_ref - t_outdoor)


def hours_to_schedule(comfort_hours: float, center_hour: float = 13.0) -> tuple:
    """Convert comfort duration to schedule start/end times."""
    half_duration = comfort_hours / 2
    start_decimal = max(6.0, center_hour - half_duration)
    end_decimal = min(22.0, center_hour + half_duration)

    # Adjust if we hit bounds
    actual_duration = end_decimal - start_decimal
    if actual_duration < comfort_hours:
        if start_decimal == 6.0:
            end_decimal = min(22.0, start_decimal + comfort_hours)
        elif end_decimal == 22.0:
            start_decimal = max(6.0, end_decimal - comfort_hours)

    def decimal_to_time(h):
        hours = int(h)
        minutes = int((h - hours) * 60)
        return f"{hours:02d}:{minutes:02d}"

    return decimal_to_time(start_decimal), decimal_to_time(end_decimal)


def generate_thk2_design(t_outdoor_ref: float = DEFAULT_REF_OUTDOOR) -> pd.DataFrame:
    """
    Generate T_HK2-targeted design.

    Design philosophy:
    1. Define target T_HK2 levels for comfort and eco modes
    2. Select parameter combinations that achieve target T_HK2 with good spread
    3. Vary schedule hours independently (orthogonal to T_HK2)
    """
    hc_params = load_heating_curve_params()
    t_ref_comfort = hc_params['t_ref_comfort']
    t_ref_eco = hc_params['t_ref_eco']

    print(f"\nGenerating T_HK2-targeted design:")
    print(f"  Reference outdoor temp: {t_outdoor_ref}°C")
    print(f"  Comfort delta (T_ref - T_out): {t_ref_comfort - t_outdoor_ref:.2f}°C")
    print(f"  Eco delta (T_ref - T_out): {t_ref_eco - t_outdoor_ref:.2f}°C")

    # Design points optimized for T_HK2 spread
    # Each row: (comfort_sp, eco_sp, curve_rise, comfort_hours, notes)
    design_points = [
        # Block 1: Minimum T_HK2 corner
        (19.0, 14.0, 0.80, 12, "Minimum T_HK2 corner"),
        # Block 2: Low-mid comfort, high eco
        (19.5, 18.0, 0.95, 10, "Low-mid comfort, high eco"),
        # Block 3: Mid comfort, mid eco
        (20.0, 15.0, 1.10, 8, "Mid comfort, mid eco"),
        # Block 4: High T_HK2 both modes
        (20.5, 19.0, 1.20, 14, "High T_HK2 both modes"),
        # Block 5: Maximum comfort T_HK2
        (22.0, 14.0, 1.20, 10, "Maximum comfort T_HK2"),
        # Block 6: High setpoint, low curve
        (21.0, 17.0, 0.85, 16, "High setpoint, low curve"),
        # Block 7: Low setpoint, high curve
        (19.0, 16.0, 1.15, 12, "Low setpoint, high curve"),
        # Block 8: Mid comfort, low eco T_HK2
        (20.5, 14.5, 0.90, 8, "Mid comfort, low eco T_HK2"),
        # Block 9: Near baseline + high eco
        (21.5, 18.5, 1.00, 14, "Near baseline + high eco"),
        # Block 10: Center of space
        (19.5, 15.5, 1.05, 16, "Center of space"),
    ]

    records = []
    for i, (comfort_sp, eco_sp, curve, hours, notes) in enumerate(design_points, 1):
        # Calculate T_HK2 values
        thk2_comfort = calc_thk2(comfort_sp, curve, t_ref_comfort, t_outdoor_ref)
        thk2_eco = calc_thk2(eco_sp, curve, t_ref_eco, t_outdoor_ref)

        # Convert hours to schedule times
        start_time, end_time = hours_to_schedule(hours)

        records.append({
            'design_point': i,
            'comfort_setpoint': comfort_sp,
            'eco_setpoint': eco_sp,
            'curve_rise': curve,
            'comfort_hours': hours,
            'comfort_start': start_time,
            'comfort_end': end_time,
            'T_HK2_comfort': round(thk2_comfort, 1),
            'T_HK2_eco': round(thk2_eco, 1),
            'notes': notes,
        })

    df = pd.DataFrame(records)

    # Print design with T_HK2 values
    print(f"\nT_HK2-Targeted Design Matrix:")
    print(df[['design_point', 'comfort_setpoint', 'eco_setpoint', 'curve_rise',
              'comfort_hours', 'T_HK2_comfort', 'T_HK2_eco']].to_string(index=False))

    return df


def validate_design(df: pd.DataFrame) -> dict:
    """Validate T_HK2 spread and parameter bounds."""
    results = {
        'n_points': len(df),
        'thk2_comfort_range': (df['T_HK2_comfort'].min(), df['T_HK2_comfort'].max()),
        'thk2_eco_range': (df['T_HK2_eco'].min(), df['T_HK2_eco'].max()),
        'thk2_comfort_spread': df['T_HK2_comfort'].max() - df['T_HK2_comfort'].min(),
        'thk2_eco_spread': df['T_HK2_eco'].max() - df['T_HK2_eco'].min(),
        'warnings': [],
    }

    # Check T_HK2 spread
    if results['thk2_comfort_spread'] < 8.0:
        results['warnings'].append(
            f"Comfort T_HK2 spread only {results['thk2_comfort_spread']:.1f}°C (target: >8°C)"
        )

    if results['thk2_eco_spread'] < 8.0:
        results['warnings'].append(
            f"Eco T_HK2 spread only {results['thk2_eco_spread']:.1f}°C (target: >8°C)"
        )

    # Check for near-duplicate T_HK2 values
    thk2_comfort = df['T_HK2_comfort'].values
    for i in range(len(thk2_comfort)):
        for j in range(i + 1, len(thk2_comfort)):
            if abs(thk2_comfort[i] - thk2_comfort[j]) < 0.5:
                results['warnings'].append(
                    f"Design points {i+1} and {j+1} have similar comfort T_HK2 "
                    f"({thk2_comfort[i]:.1f} vs {thk2_comfort[j]:.1f})"
                )

    # Check parameter bounds
    for param, (low, high) in PARAM_BOUNDS.items():
        if param in df.columns:
            if df[param].min() < low or df[param].max() > high:
                results['warnings'].append(
                    f"{param} outside bounds [{low}, {high}]"
                )

    return results


def save_design(df: pd.DataFrame, t_outdoor_ref: float) -> None:
    """Save design to CSV and JSON formats."""
    hc_params = load_heating_curve_params()

    # CSV format
    csv_path = OUTPUT_DIR / 'thk2_design.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # JSON format
    json_path = OUTPUT_DIR / 'thk2_design.json'
    design_dict = {
        'generated': datetime.now().isoformat(),
        'design_type': 'T_HK2-targeted',
        'n_points': len(df),
        'reference_outdoor_temp': t_outdoor_ref,
        'heating_curve_params': hc_params,
        'param_bounds': PARAM_BOUNDS,
        'safety': SAFETY,
        'design_points': df.to_dict(orient='records'),
    }
    with open(json_path, 'w') as f:
        json.dump(design_dict, f, indent=2)
    print(f"Saved: {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate T_HK2-targeted design for pilot experiment',
    )
    parser.add_argument(
        '--ref-outdoor',
        type=float,
        default=DEFAULT_REF_OUTDOOR,
        help=f'Reference outdoor temperature for T_HK2 calculation (default: {DEFAULT_REF_OUTDOOR}°C)',
    )

    args = parser.parse_args()

    # Generate design
    df = generate_thk2_design(t_outdoor_ref=args.ref_outdoor)

    # Validate
    validation = validate_design(df)

    print(f"\nValidation:")
    print(f"  Comfort T_HK2 range: {validation['thk2_comfort_range'][0]:.1f} - "
          f"{validation['thk2_comfort_range'][1]:.1f}°C "
          f"(spread: {validation['thk2_comfort_spread']:.1f}°C)")
    print(f"  Eco T_HK2 range: {validation['thk2_eco_range'][0]:.1f} - "
          f"{validation['thk2_eco_range'][1]:.1f}°C "
          f"(spread: {validation['thk2_eco_spread']:.1f}°C)")

    if validation['warnings']:
        print(f"  Warnings:")
        for w in validation['warnings']:
            print(f"    - {w}")
    else:
        print(f"  No warnings - design looks good!")

    # Save
    save_design(df, args.ref_outdoor)

    print("\nDone! Review thk2_design.csv")
    return 0


if __name__ == '__main__':
    exit(main())
