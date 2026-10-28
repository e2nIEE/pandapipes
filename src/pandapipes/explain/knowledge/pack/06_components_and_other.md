# Pumps, Valves, Modes, and Algorithm Errors

## Keywords
pump, valve, compressor, pressure_control, press_control, flow_control, heat_exchanger, mode, hydraulics, heat, sequential, bidirectional, friction_model, nikuradse, colebrook, swamee-jain, std_type, UserWarning, calculation mode, max_iter

## 1. Wrong or missing calculation mode

### What it means
`pipeflow` raises `UserWarning("No proper calculation mode chosen.")` if `mode` is not one
of the valid options.

### Valid modes
```python
# Valid options for pp.pipeflow(net, mode=...):
# "hydraulics"    — pressure & mass flow only (default)
# "heat"          — temperature only, reusing existing hydraulic results
# "sequential"    — hydraulics first, then heat transfer
# "bidirectional" — solve hydraulics and heat coupled
pp.pipeflow(net, mode="sequential")
```
`mode="heat"` requires hydraulic results to already exist, otherwise it raises
`UserWarning("Converged flag not set. Make sure that hydraulic calculation results are available.")`.

## 2. Pump / compressor issues

### What it means
Pumps and compressors raise pressure between `from_junction` and `to_junction`. Common
failures: all pumps out of service, an unknown pump `std_type`, or parallel pumps that the
solver cannot balance.
```python
# Check pump service state and types:
print(net.pump[['from_junction', 'to_junction', 'std_type', 'in_service']])
# List available pump std_types:
print(net.std_types['pump'].keys())
```

## 3. Valve state

### What it means
A `valve` with `opened=False` blocks flow and can isolate part of the network.
```python
print(net.valve[['junction', 'element', 'opened', 'loss_coefficient']])
net.valve['opened'] = True   # open all to test
```

## 4. Pressure / flow control infeasibility

### What it means
`pressure_control` (press_control) and `flow_control` enforce a setpoint. If the setpoint
is physically unreachable (e.g. controlled pressure higher than the supply), the solver
diverges.
```python
print(net.press_control[['controlled_junction', 'controlled_p_bar', 'control_active']])
# Relax or disable the control to test feasibility:
net.press_control['control_active'] = False
pp.pipeflow(net)
```

## 5. Friction model

### What it means
The friction model sets how pipe pressure loss is computed. Colebrook iterates internally
and can fail to converge on difficult nets.
```python
# Valid options for pp.pipeflow(net, friction_model=...):
# "nikuradse"   — default, robust, non-iterative
# "colebrook"   — Colebrook-White (iterative; tune max_iter_colebrook)
# "swamee-jain" — explicit approximation of Colebrook
pp.pipeflow(net, friction_model="nikuradse")
```

## 6. Solver robustness knobs

```python
pp.pipeflow(net, max_iter_hyd=100)              # default 10 (hydraulics)
pp.pipeflow(net, max_iter_therm=100)            # default 10 (heat transfer)
pp.pipeflow(net, nonlinear_method="automatic")  # adaptive damping factor alpha
pp.pipeflow(net, tol_m=1e-5, tol_p=1e-5)        # mass-flow / pressure tolerances
```

## Quick checklist
- [ ] `mode` is one of "hydraulics" / "heat" / "sequential" / "bidirectional"
- [ ] For `mode="heat"`, hydraulic results already exist (run hydraulics first)
- [ ] Not all pumps / sources are `in_service=False`
- [ ] Pump / pipe `std_type` names exist in `net.std_types`
- [ ] Pressure/flow-control setpoints are physically reachable
- [ ] `friction_model` is "nikuradse", "colebrook", or "swamee-jain"
