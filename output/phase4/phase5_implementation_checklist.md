
# Phase 5 Implementation Checklist

## Pre-Study Setup (Fall 2027)

### Equipment Preparation
- [ ] Verify heat pump interface access for curve_rise adjustment
- [ ] Verify Home Assistant integration for setpoint control
- [ ] Set up automated logging for all sensor data
- [ ] Create backup of current settings

### Data Collection Setup
- [ ] Ensure sensor data logging rate: 15-minute intervals minimum
- [ ] Set up weather forecast integration (for MPC future work)
- [ ] Create daily summary dashboard
- [ ] Test alert system for comfort violations

### Randomization
- [ ] Generate randomized block schedule (3-5 days per block)
- [ ] Balance strategies across early/mid/late winter
- [ ] Document exact switch times in log

## Parameter Change Protocol

### Baseline Settings
```
Comfort Start:    06:30
Comfort End:      20:00
Setpoint Comfort: 20.2°C
Setpoint Eco:     18.0°C
Curve Rise:       1.08
Buffer Target:    36°C
```

### Energy-Optimized Settings
```
Comfort Start:    10:00
Comfort End:      18:00
Setpoint Comfort: 20.0°C
Setpoint Eco:     17.5°C
Curve Rise:       0.98
Buffer Target:    40°C
```

### Aggressive Solar Settings
```
Comfort Start:    10:00
Comfort End:      17:00
Setpoint Comfort: 21.0°C
Setpoint Eco:     17.0°C
Curve Rise:       0.95
Buffer Target:    45°C
```

### Cost-Optimized Settings
```
Comfort Start:    11:00
Comfort End:      21:00
Setpoint Comfort: 20.0°C
Setpoint Eco:     17.0°C
Curve Rise:       0.95
Buffer Target:    40°C
Curve Rise (Grid): 0.85 (when grid-dependent)
```

## Daily Monitoring

- [ ] Check comfort compliance (target: ≥95%)
- [ ] Log any manual overrides
- [ ] Note occupancy deviations
- [ ] Record weather conditions
- [ ] Check sensor data quality

## Block Transition Protocol

1. Record end time of current block
2. Wait for system to reach steady state (minimum 2 hours after schedule change)
3. Apply new parameter set
4. Record start time of new block
5. Verify all parameters changed successfully

## Success Metrics

| Strategy | Self-Sufficiency Target | COP Target | Comfort Min | Cost Change |
|----------|------------------------|------------|-------------|-------------|
| Baseline | 58% | 3.5 | 95% | — |
| Energy-Optimized | 68% | 4.0 | 95% | +5-10% savings |
| Aggressive Solar | 85% | 4.2 | 95% | +10-15% savings |
| Cost-Optimized | 61% | 3.4 | 90% | +15-25% savings |

## Safety Limits

- Minimum room temperature: 16°C (override if violated)
- Maximum flow temperature: 45°C
- Alert if comfort compliance < 90% for 24 hours
