# Data Format and Structure

## Data Structure

```
data/
├── daily/       # 15-minute interval readings: Energy_Balance_YYYY_MM_DD.csv
├── monthly/     # Aggregated daily data: Energy_Balance_YYYY_MM.csv
├── yearly/      # Aggregated monthly data: Energy_Balance_YYYY.csv
├── mainic*.csv  # InfluxDB export from Home Assistant (annotated CSV format)
└── tariffs/     # Electricity tariff data (Primeo Energie)
    └── primeo_tariffs_source.json  # Manually collected tariff rates
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

## Processed Data

After preprocessing, data is saved to `output/phase1/`:

**Parquet datasets:**
- `energy_balance_15min.parquet` - 15-min interval energy data (kWh)
- `energy_balance_daily.parquet` - Daily totals from monthly files
- `energy_balance_monthly.parquet` - Monthly totals from yearly files
- `sensors_heating.parquet` - Heat pump sensors (96 sensors)
- `sensors_weather.parquet` - Davis weather station (27 sensors)
- `sensors_rooms.parquet` - Room temperature sensors (3 sensors)
- `sensors_energy.parquet` - Smart plug consumption (43 sensors)
- `integrated_dataset.parquet` - All data merged (185 columns)
- `integrated_overlap_only.parquet` - Overlap period only (64 days)

**Reports and logs:**
- `phase1_report.html` - Comprehensive preprocessing documentation
- `corrections_log.csv` - Data cleaning corrections applied
- `validation_results.csv` - Daily vs monthly validation results
- `sensor_summary.csv` - Per-sensor statistics
- `data_overlap_summary.csv` - Data source overlap info

## Electricity Tariffs

Tariff data from Primeo Energie (provider) for cost modeling.

**Important:** Only feed-in tariffs WITH HKN (Herkunftsnachweis) are used in analysis.
Base-only feed-in rates are excluded because this installation participates in the
HKN program, receiving the additional HKN bonus on top of the base rate.

Run:
```bash
python src/phase1/04_preprocess_tariffs.py
```

**Data sources:**
- Primeo Energie official announcements and price sheets
- ElCom LINDAS SPARQL endpoint (official Swiss tariff database)

**Tariff time windows:**
| Tariff | Time Windows |
|--------|--------------|
| High (Hochtarif) | Mon-Fri 06:00-21:00, Sat 06:00-12:00 |
| Low (Niedertarif) | Mon-Fri 21:00-06:00, Sat 12:00 - Mon 06:00, Federal holidays |

**Swiss holidays (low tariff all day):**
- January 1, Good Friday, Easter Saturday/Monday, Ascension, Whit Monday, August 1, December 25

**Purchase tariffs (Rp/kWh):**
| Period | All-in Average | Notes |
|--------|----------------|-------|
| 2023 | 31.3 | Peak energy crisis year |
| 2024 | 32.8 | +4.8% vs 2023 |
| 2025 | 32.6 | -1% vs 2024 |

**Feed-in tariffs (Ruckliefervergutung, Rp/kWh):**
| Period | Base Rate | With HKN | Notes |
|--------|-----------|----------|-------|
| Jan-Jun 2023 | 20.0 | 21.5 | Peak rate after energy crisis |
| Jul-Dec 2023 | 16.0 | 17.5 | Market price reduction |
| Jan-Jun 2024 | 16.0 | 17.5 | Stable |
| Jul-Dec 2024 | 13.0 | 14.5 | -20% announced cut |
| Jan-Mar 2025 | 13.0 | 15.5 | HKN increased to 2.5 Rp |
| Apr 2025+ | 10.5 | 13.0 | Further reduction |
| Minimum guarantee | 9.0 | - | Guaranteed through 2028 |

**Outputs saved to `output/phase1/`:**
- `tariff_schedule.csv` - All tariff rates with validity periods
- `tariff_flags_hourly.parquet` - Hourly high/low tariff flags
- `tariff_series_hourly.parquet` - Time-indexed tariff rates
- `tariff_report_section.html` - HTML report section
