# Sink and Source Errors

## Keywords
sink, source, mdot, mdot_kg_per_s, mass, flow, demand, load, consumer, supply, feed, balance, scaling, sign, negative, withdraw, inject, ext_grid, mass_storage, overload, kg/h

## What it means
Sinks and sources set the mass-flow boundary conditions. pandapipes convention:
a **sink** withdraws fluid (consumer) and a **source** feeds fluid in (producer), both
declared with a positive `mdot_kg_per_s`. Getting the element or sign wrong unbalances the
network and can prevent convergence.

## Source basis
This pack follows pandapipes sink/source component documentation. A sink has a connected
`junction`, a drawn mass flow `mdot_kg_per_s`, and a non-negative `scaling` factor.
Positive sink mass flow means flow leaves the network system and is used for hydraulic
loads. The kg/h conversion examples are unit-sanity guidance: pandapipes expects kg/s,
so external data in kg/h must be converted before assignment.

## Cause 1: Mass imbalance — demand without supply
The total fluid withdrawn by sinks must be supplied by sources, pumps, and/or the ext_grid.
If sinks demand more than anything supplies, the solver cannot close the mass balance.
```python
print("Total sink  :", net.sink.mdot_kg_per_s.sum(), "kg/s")
print("Total source:", net.source.mdot_kg_per_s.sum(), "kg/s")
# The ext_grid acts as the slack feed-in, absorbing the remaining imbalance.
# With no ext_grid, source feed-in must cover sink demand exactly.
```

## Cause 2: Using a sink where a source is needed (sign/element mix-up)
A negative `mdot_kg_per_s` on a sink represents feed-in — use a `source` instead of a
negative sink (and vice versa).
```python
# Wrong: negative sink to represent a feed-in
pp.create_sink(net, junction=j, mdot_kg_per_s=-2.0)

# Right: a source feeds fluid in
pp.create_source(net, junction=j, mdot_kg_per_s=2.0)
```

## Cause 3: NaN or missing mdot_kg_per_s
An unset mass flow injects NaN into the balance and the solver diverges immediately.
```python
print(net.sink[net.sink.mdot_kg_per_s.isna()])
net.sink['mdot_kg_per_s'] = net.sink['mdot_kg_per_s'].fillna(0.0)
```

## Cause 4: `scaling` factor set to zero (or unrealistic)
`mdot_kg_per_s` is multiplied by `scaling`. A scaling of 0 zeroes the demand; a very large
scaling can overload the network.
```python
print(net.sink[['mdot_kg_per_s', 'scaling']])
net.sink['scaling'] = 1.0   # reset to nominal
```

## Cause 5: Wrong magnitude / units
Mass flow is in **kg/s**, not kg/h or m³/s. A value entered in kg/h without conversion
is too large by a factor of 3600.
```python
# Wrong: 3600 kg/h entered directly as mdot_kg_per_s
# Right: 3600 kg/h ÷ 3600 = 1.0 kg/s
pp.create_sink(net, junction=j, mdot_kg_per_s=3600/3600)
```

## Cause 6: Unrealistic sink demand relative to the network
A sink demand can be numeric and positive but still physically impossible for the pipe
diameters, lengths, pressure level, and fluid. If the diagnostic says pipeflow converges
after scaling sinks down, identify the corrupted sink row and reduce/convert that demand
instead of changing generic solver settings.
```python
print(net.sink[["junction", "mdot_kg_per_s", "scaling"]].sort_values("mdot_kg_per_s"))
```

## Quick checklist
- [ ] `net.sink.mdot_kg_per_s.isna().any()` → False (and same for `net.source`)
- [ ] `(net.sink.mdot_kg_per_s >= 0).all()` — use `source` for feed-in, not negative sinks
- [ ] Total supply (sources + ext_grid + pumps) can cover total sink demand
- [ ] `net.sink['scaling'].min() > 0` — scaling not accidentally zeroed
- [ ] Mass-flow magnitudes are in kg/s (not kg/h)
