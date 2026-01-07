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

## Current State

This is a data repository without source code yet. No build system, tests, or dependencies are configured.
