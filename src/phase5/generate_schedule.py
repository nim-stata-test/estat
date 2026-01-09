#!/usr/bin/env python3
"""
Phase 5: Generate Randomization Schedule

Generates a balanced randomized block schedule for the intervention study.
Uses a constrained randomization approach to ensure:
- Each strategy appears ~equally across the study
- Strategies are balanced across early/mid/late winter
- No strategy follows itself (carryover balance)

Usage:
    python src/phase5/generate_schedule.py --start 2027-11-01 --weeks 20 --seed 42
    python src/phase5/generate_schedule.py --help

Outputs:
    output/phase5/block_schedule.csv - Complete block schedule
    output/phase5/block_schedule.json - Machine-readable format
"""

import argparse
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'output' / 'phase5'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Strategy definitions
STRATEGIES = {
    'A': {
        'name': 'Baseline',
        'comfort_start': '06:30',
        'comfort_end': '20:00',
        'setpoint_comfort': 20.2,
        'setpoint_eco': 18.0,
        'curve_rise': 1.08,
        'buffer_target': 36,
    },
    'B': {
        'name': 'Energy-Optimized',
        'comfort_start': '10:00',
        'comfort_end': '18:00',
        'setpoint_comfort': 20.0,
        'setpoint_eco': 17.5,
        'curve_rise': 0.98,
        'buffer_target': 40,
    },
    'C': {
        'name': 'Aggressive Solar',
        'comfort_start': '10:00',
        'comfort_end': '17:00',
        'setpoint_comfort': 21.0,
        'setpoint_eco': 17.0,
        'curve_rise': 0.95,
        'buffer_target': 45,
    },
    'D': {
        'name': 'Cost-Optimized',
        'comfort_start': '11:00',
        'comfort_end': '21:00',
        'setpoint_comfort': 20.0,
        'setpoint_eco': 17.0,
        'curve_rise': 0.95,
        'buffer_target': 38,
    },
}

# Block duration in days (from power analysis: 3-day washout + 2-day measurement)
BLOCK_DAYS = 5
WASHOUT_DAYS = 3
MEASUREMENT_DAYS = 2


def get_season(date: datetime, study_start: datetime) -> str:
    """Determine season tercile based on study progress."""
    days_elapsed = (date - study_start).days
    total_days = 140  # ~20 weeks

    if days_elapsed < total_days / 3:
        return 'early'
    elif days_elapsed < 2 * total_days / 3:
        return 'mid'
    else:
        return 'late'


def generate_balanced_sequence(n_blocks: int, seed: int) -> list:
    """
    Generate a balanced sequence of strategies.

    Ensures:
    - Approximately equal representation of each strategy
    - No consecutive repeats
    - Reasonably balanced across study phases
    """
    random.seed(seed)
    strategies = list(STRATEGIES.keys())
    n_strategies = len(strategies)

    # Calculate how many complete sets we need
    complete_sets = n_blocks // n_strategies
    remainder = n_blocks % n_strategies

    # Build base sequence with complete balanced sets
    sequence = []
    for _ in range(complete_sets):
        shuffled = strategies.copy()
        random.shuffle(shuffled)

        # Ensure no consecutive repeat with previous block
        if sequence and shuffled[0] == sequence[-1]:
            # Swap first element with another
            swap_idx = random.randint(1, n_strategies - 1)
            shuffled[0], shuffled[swap_idx] = shuffled[swap_idx], shuffled[0]

        sequence.extend(shuffled)

    # Add remainder blocks
    if remainder > 0:
        extra = strategies.copy()
        random.shuffle(extra)

        # Avoid consecutive repeat
        if sequence and extra[0] == sequence[-1]:
            swap_idx = random.randint(1, n_strategies - 1)
            extra[0], extra[swap_idx] = extra[swap_idx], extra[0]

        sequence.extend(extra[:remainder])

    return sequence


def generate_schedule(
    start_date: datetime,
    n_weeks: int,
    seed: int,
) -> pd.DataFrame:
    """Generate the complete block schedule."""

    # Calculate number of blocks
    total_days = n_weeks * 7
    n_blocks = total_days // BLOCK_DAYS

    print(f"Generating schedule:")
    print(f"  Start date: {start_date.strftime('%Y-%m-%d')}")
    print(f"  Duration: {n_weeks} weeks ({total_days} days)")
    print(f"  Block length: {BLOCK_DAYS} days")
    print(f"  Total blocks: {n_blocks}")
    print(f"  Random seed: {seed}")

    # Generate balanced sequence
    strategy_sequence = generate_balanced_sequence(n_blocks, seed)

    # Build schedule DataFrame
    records = []
    current_date = start_date

    for block_num, strategy_code in enumerate(strategy_sequence, 1):
        block_start = current_date
        block_end = current_date + timedelta(days=BLOCK_DAYS - 1)
        washout_end = current_date + timedelta(days=WASHOUT_DAYS - 1)
        measurement_start = current_date + timedelta(days=WASHOUT_DAYS)
        season = get_season(block_start, start_date)
        strategy = STRATEGIES[strategy_code]

        records.append({
            'block': block_num,
            'start_date': block_start.strftime('%Y-%m-%d'),
            'end_date': block_end.strftime('%Y-%m-%d'),
            'washout_end': washout_end.strftime('%Y-%m-%d'),
            'measurement_start': measurement_start.strftime('%Y-%m-%d'),
            'strategy_code': strategy_code,
            'strategy_name': strategy['name'],
            'season': season,
            'comfort_start': strategy['comfort_start'],
            'comfort_end': strategy['comfort_end'],
            'setpoint_comfort': strategy['setpoint_comfort'],
            'setpoint_eco': strategy['setpoint_eco'],
            'curve_rise': strategy['curve_rise'],
            'buffer_target': strategy['buffer_target'],
            'notes': '',
        })

        current_date = block_end + timedelta(days=1)

    df = pd.DataFrame(records)

    # Print summary statistics
    print(f"\nStrategy distribution:")
    for code, name in [(k, v['name']) for k, v in STRATEGIES.items()]:
        count = (df['strategy_code'] == code).sum()
        print(f"  {code} ({name}): {count} blocks")

    print(f"\nSeason distribution:")
    for season in ['early', 'mid', 'late']:
        subset = df[df['season'] == season]
        print(f"  {season.capitalize()}: {len(subset)} blocks")
        for code in STRATEGIES.keys():
            count = (subset['strategy_code'] == code).sum()
            print(f"    {code}: {count}")

    return df


def save_schedule(df: pd.DataFrame) -> None:
    """Save schedule to CSV and JSON formats."""

    # CSV format
    csv_path = OUTPUT_DIR / 'block_schedule.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # JSON format (for programmatic access)
    json_path = OUTPUT_DIR / 'block_schedule.json'
    schedule_dict = {
        'generated': datetime.now().isoformat(),
        'n_blocks': len(df),
        'block_days': BLOCK_DAYS,
        'strategies': STRATEGIES,
        'blocks': df.to_dict(orient='records'),
    }
    with open(json_path, 'w') as f:
        json.dump(schedule_dict, f, indent=2)
    print(f"Saved: {json_path}")

    # Print first few blocks as preview
    print(f"\nSchedule preview (first 8 blocks):")
    preview = df[['block', 'start_date', 'end_date', 'strategy_code', 'strategy_name', 'season']].head(8)
    print(preview.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description='Generate randomized block schedule for Phase 5 intervention study',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python src/phase5/generate_schedule.py --start 2027-11-01 --weeks 20 --seed 42
    python src/phase5/generate_schedule.py --start 2027-11-01 --weeks 16
        """,
    )
    parser.add_argument(
        '--start',
        type=str,
        default='2027-11-01',
        help='Study start date (YYYY-MM-DD), default: 2027-11-01',
    )
    parser.add_argument(
        '--weeks',
        type=int,
        default=20,
        help='Study duration in weeks, default: 20',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility (default: random)',
    )

    args = parser.parse_args()

    # Parse start date
    try:
        start_date = datetime.strptime(args.start, '%Y-%m-%d')
    except ValueError:
        print(f"Error: Invalid date format '{args.start}'. Use YYYY-MM-DD.")
        return 1

    # Use random seed if not specified
    seed = args.seed if args.seed is not None else random.randint(1, 99999)

    # Generate and save schedule
    df = generate_schedule(start_date, args.weeks, seed)
    save_schedule(df)

    print("\nDone! Review the schedule and adjust manually if needed.")
    return 0


if __name__ == '__main__':
    exit(main())
