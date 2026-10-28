# Fluid Definition Errors

## Keywords
fluid, gas, liquid, water, hgas, lgas, hydrogen, methane, air, density, viscosity, compressible, incompressible, create_fluid_from_lib, properties, no fluid, missing fluid, is_gas, temperature

## What it means
`pipeflow` needs a fluid to provide physical properties (density, viscosity,
compressibility) for the hydraulic and thermal equations. A missing or wrong fluid is a
frequent cause of non-convergence or unphysical results. Gases are compressible and
pressure-dependent; liquids (water) are treated as incompressible — choosing the wrong one
changes the whole solution.

## Source basis
This pack follows pandapipes fluid-property documentation. A pandapipes network is run
with a selected fluid from the fluid library or a user-created fluid, and fluid properties
provide density, viscosity, heat capacity, compressibility, and related values used by
components during pipeflow. Standard library fluids include water, air, hydrogen, methane,
high/low calorific gas, and biomethane variants.

## Cause 1: No fluid defined on the net
Without a fluid, pipeflow cannot evaluate the friction and continuity terms.
```python
# Check:
print(net.fluid)   # None or missing means no fluid set

# Fix — set a standard fluid (either at creation or after):
net = pp.create_empty_network(fluid="water")
# or
pp.create_fluid_from_lib(net, "water")
```

## Cause 2: Wrong or misspelled fluid name
`create_fluid_from_lib` only accepts names that exist in the standard library. A typo
raises an error.
```python
# Standard library fluids (verified):
#   Liquid: "water"
#   Gases : "hgas", "lgas", "air", "hydrogen", "methane",
#           "biomethane_pure", "biomethane_treated"
pp.create_fluid_from_lib(net, "lgas")   # low-calorific natural gas
```

## Cause 3: Gas vs. liquid mismatch
A model built for water (incompressible, small pressure drops) will behave very differently
if the fluid is a gas, and vice versa. Confirm the fluid type matches the physical system.
```python
print(net.fluid.name, "| is_gas =", net.fluid.is_gas)
```

## Cause 4: Unrealistic temperature / pressure for the fluid
Fluid properties are evaluated at the junction temperature `tfluid_k` (Kelvin) and the
local pressure (bar, absolute). Values far outside the fluid's valid range (e.g. tfluid_k
near 0, or negative pressure) produce NaN properties.
```python
print(net.junction[['pn_bar', 'tfluid_k']])   # pressure > 0 bar, temperature in sensible K
```

## Cause 5: Heat-transfer run without temperature boundary conditions
For `mode="heat"` / `"sequential"` / `"bidirectional"`, a temperature reference is needed
(an ext_grid with `type="t"` or `"pt"`, or `t_k` set).
```python
pp.create_ext_grid(net, junction=j_supply, p_bar=5.0, t_k=363.15, type="pt")
pp.pipeflow(net, mode="sequential")
```

## Quick checklist
- [ ] `net.fluid is not None` — a fluid is defined
- [ ] Fluid name is one of the standard library fluids (or a valid custom fluid)
- [ ] `net.fluid.is_gas` matches the physical system (gas vs. water)
- [ ] `net.junction.tfluid_k.min() > 0` and `net.junction.pn_bar.min() > 0`
- [ ] For heat-transfer modes, a temperature reference (ext_grid type "t"/"pt") exists
