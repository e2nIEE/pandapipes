# Convergence Failures

## Keywords
converged, convergence, pipeflow, hydraulic, hydraulics, not converge, did not converge, iterations, PipeflowNotConverged, diverge, singular, matrix, jacobian, newton, raphson, heat, transfer, bidirectional, slack, pressure, reference

## What it means
`PipeflowNotConverged` means the Newton-Raphson solver could not balance mass flow and
pressure (hydraulics) or temperature (heat transfer) within the allowed iterations. The
message is one of:
- "The hydraulic calculation did not converge to a solution."
- "The heat transfer calculation did not converge to a solution."
- "The bidrectional calculation did not converge to a solution."
This is almost always a network-modelling problem — pandapipes' solver is correct. The
network is either physically infeasible, missing a boundary condition, or poorly initialised.

## Source basis
This pack is grounded in pandapipes' documented pipeflow behavior: pipeflow calculates
junction pressures, pipe velocities, temperatures, and heat transfer, and requires
boundary conditions. The external grid documentation states that hydraulic calculations
need at least one fixed pressure value in each separate grid area. The pipe model includes
pipe length, diameter, roughness/friction, and height difference in pressure-loss
relationships. The diagnostic thresholds in Explain are preflight red flags, not official
validity limits.

## Use pandapipes diagnostics before guessing
`pandapipes.diagnostic.diagnostic(net, report=False)` runs structured checks for invalid
values, missing junction references, topology/boundary issues, and selected
non-convergence hypotheses. The Explain feature injects a compact subset of these
diagnostic results into the LLM prompt; treat those results as stronger evidence than the
generic `PipeflowNotConverged` message.
```python
from pandapipes.diagnostic import diagnostic
print(diagnostic(net, report=False))

import pandapipes.topology as top
top.unsupplied_junctions(net)          # junctions with no path to any ext_grid (slack)
pp.pipeflow(net, check_connectivity=True)  # default ON: auto-sets unsupplied areas out of service
```

## Cause 1: No ext_grid — no pressure reference (slack node) — most common
Every network needs at least one `ext_grid` that fixes a pressure. Without it the solver
has no anchor and the pressure equations are underdetermined.
```python
# Fix — fix pressure at the supply/feed-in junction:
pp.create_ext_grid(net, junction=j_supply, p_bar=5.0, t_k=293.15, type="pt")
```

## Cause 2: Disconnected junctions / network islands
A junction with no path to an ext_grid cannot be solved.
```python
import pandapipes.topology as top
print(top.unsupplied_junctions(net))   # set of junctions not connected to a slack
# Fix: connect with a pipe, or remove the isolated element.
```

## Cause 3: Missing or wrong fluid
`pipeflow` needs a fluid (densities, viscosities). Gas networks (compressible) and water
networks (incompressible) behave very differently.
```python
pp.create_fluid_from_lib(net, "water")     # or set fluid= in create_empty_network
```

## Cause 4: All pumps / sources out of service, or no feed-in
If every pump or feed-in element is `in_service=False`, or sinks demand fluid that nothing
supplies, the mass balance cannot close.
```python
print(net.pump[~net.pump.in_service])      # check for out-of-service pumps
print(net.sink.mdot_kg_per_s.sum(), net.source.mdot_kg_per_s.sum())  # demand vs supply
```

## Cause 5: Colebrook friction does not converge
The Colebrook-White friction model iterates internally and can fail to converge.
```python
pp.pipeflow(net, friction_model="nikuradse")   # default, robust
pp.pipeflow(net, friction_model="colebrook", max_iter_colebrook=100)
```

## Cause 6: Too few outer iterations / needs damping
```python
pp.pipeflow(net, max_iter_hyd=100)                 # default is 10
pp.pipeflow(net, nonlinear_method="automatic")     # adaptive damping (alpha) for hard nets
```

## Cause 7: Implausible geometry or units
Very small/zero pipe diameter, zero length, or mixing units (e.g. metres entered as km)
makes the hydraulic matrix ill-conditioned.
```python
print(net.pipe[["length_km", "inner_diameter_mm"]])  # length in km, diameter in mm
```

## Cause 8: Extreme elevations dominate hydrostatic pressure
`junction.height_m` is a real elevation in metres. A corrupted value such as `1e6` m
creates an enormous hydrostatic pressure term and can make an otherwise valid network
non-convergent. If flattening heights makes the pipeflow converge, inspect the corrupted
junction height rather than changing solver settings.
```python
print(net.junction[["height_m"]].sort_values("height_m"))
```

## Quick checklist
- [ ] `len(net.ext_grid) >= 1` — at least one pressure reference (slack node)
- [ ] `net.fluid is not None` — a fluid is defined
- [ ] `pandapipes.topology.unsupplied_junctions(net)` is empty
- [ ] No NaN: `net.sink.mdot_kg_per_s.isna().any()` should be False
- [ ] `(net.pipe.inner_diameter_mm > 0).all()` and `(net.pipe.length_km > 0).all()`
- [ ] At least one feed-in (ext_grid / source / pump) supplies the sinks' demand
