# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ESTAT is an energy balance data repository for solar/battery system monitoring. Currently contains historical energy data organized by temporal granularity.

## Data Structure

```
data/
├── daily/      # 15-minute interval readings: Energy_Balance_YYYY_MM_DD.csv
├── monthly/    # Aggregated daily data: Energy_Balance_YYYY_MM.csv
└── yearly/     # Aggregated monthly data: Energy_Balance_YYYY.csv

mainic*.csv     # InfluxDB export from Home Assistant (annotated CSV format)
```

## Data Format

- **Delimiter**: Semicolon (`;`)
- **Decimal separator**: Comma (European format, e.g., `1234,56`)
- **Encoding**: UTF-8 with BOM
- **Daily metrics** (15-min intervals, values in Watts):
  - Direct consumption / Mean values [W]
  - Battery discharging / Mean values [W]
  - External energy supply / Mean values [W]
  - Total consumption / Mean values [W]
  - Grid feed-in / Mean values [W]
  - Battery charging / Mean values [W]
  - PV power generation / Mean values [W]
  - Limiting of active power feed-in / Mean values [W]
- **Monthly/Yearly metrics**: Meter change values in kWh

## Development Environment

- **Language**: Python 3.14
- **IDE**: PyCharm configured
- **Virtual env**: `.venv/`

## Commands

```bash
# Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Preprocessing (Phase 1)
python src/preprocess_energy_balance.py  # Process energy balance CSVs
python src/preprocess_sensors.py         # Process mainic sensor data (~10 min)
python src/integrate_data.py             # Merge datasets
```

## Processed Data

After preprocessing, data is saved to `processed/`:
- `energy_balance_15min.parquet` - 15-min interval energy data (kWh)
- `sensors_heating.parquet` - Heat pump sensors (96 sensors)
- `sensors_weather.parquet` - Davis weather station (27 sensors)
- `sensors_rooms.parquet` - Room temperature sensors (3 sensors)
- `sensors_energy.parquet` - Smart plug consumption (43 sensors)
- `integrated_dataset.parquet` - All data merged (185 columns)
- `integrated_overlap_only.parquet` - Overlap period only (64 days)

## Research Design

See `RDP.md` for the heating strategy optimization research plan.
