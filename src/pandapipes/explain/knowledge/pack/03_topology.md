# Topology Errors

## Keywords
topology, disconnected, isolated, island, junction, valve, opened, closed, connected, ext_grid, path, network, subnet, missing, unsupplied, connectivity, slack, reachable

## What it means
Topology errors mean the network graph has junctions with no path to an external grid.
Every junction needs a path to a slack node (`ext_grid`) via in-service pipes, open valves,
and pumps. An unsupplied junction has no pressure reference and cannot be solved, which
surfaces as `PipeflowNotConverged`.

## Source basis
This pack follows pandapipes connectivity behavior and component definitions. External
grids fix pressure and/or temperature at a junction; the documentation states that each
separate hydraulic grid area needs a fixed pressure value. Pipes connect two junctions,
and valves can block flow when closed. Missing junction references and unsupplied islands
are therefore structural model errors, not generic Python bugs.

## Diagnostic
```python
from pandapipes.diagnostic import diagnostic
print(diagnostic(net, report=False))

import pandapipes.topology as top
unsupplied = top.unsupplied_junctions(net)   # set of junctions not connected to any slack
print("Unsupplied junctions:", unsupplied)

# Build the network graph directly for custom checks:
mg = top.create_nxgraph(net)
```
By default `pp.pipeflow(net, check_connectivity=True)` auto-sets unreachable in-service
areas out of service so the rest can still solve — but the disconnected part is then unsolved.

## Cause 1: Junction created but not connected
The most common mistake: calling `create_junction()` without adding a pipe to it.
```python
# Wrong: orphan junction
j = pp.create_junction(net, pn_bar=1.0, tfluid_k=293.15, name="MV Junction")
# (no pipe connecting it to anything)

# Right: connect it immediately
j = pp.create_junction(net, pn_bar=1.0, tfluid_k=293.15, name="MV Junction")
pp.create_pipe_from_parameters(net, from_junction=existing_j, to_junction=j,
                               length_km=0.5, inner_diameter_mm=200)
```

## Cause 2: Closed valve isolating a section
A valve with `opened=False` cuts a section off from the supply.
```python
# Check for closed valves:
print(net.valve[~net.valve.opened])

# Open all valves temporarily to test connectivity:
net.valve['opened'] = True
pp.pipeflow(net)   # if this converges, a valve state was the problem
```

## Cause 3: Out-of-service pipes
A pipe with `in_service=False` is removed from the graph, possibly stranding downstream
junctions.
```python
print(net.pipe[~net.pipe.in_service])
net.pipe['in_service'] = True   # re-enable to test
```

## Cause 4: Pipe connected to a non-existent junction
If `from_junction` or `to_junction` references a junction index that doesn't exist, the
pipe cannot bridge to the downstream junction.
```python
all_j = set(net.junction.index)
bad = net.pipe[~net.pipe.from_junction.isin(all_j) | ~net.pipe.to_junction.isin(all_j)]
print("Pipes with missing junction references:", bad)
```
If a diagnostic finding gives a specific reference such as
`net.pipe.loc[0, "to_junction"] = 999` while junction indices are `0..5`, report that exact
pipe row as the root cause.

## Cause 5: No external grid at all
```python
# Fix — fix pressure at the feed-in junction:
pp.create_ext_grid(net, junction=j_supply, p_bar=5.0, t_k=293.15, type="pt")
```

## Quick checklist
- [ ] `len(net.ext_grid) >= 1` — at least one slack node exists
- [ ] `pandapipes.topology.unsupplied_junctions(net)` returns an empty set
- [ ] Every junction with a sink is reachable from an ext_grid via in-service pipes
- [ ] All valves that should be open have `opened=True`
- [ ] No required pipe has `in_service=False`
