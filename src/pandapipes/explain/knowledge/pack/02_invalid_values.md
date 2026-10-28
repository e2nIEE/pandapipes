# Invalid Values and NaN

## Keywords
invalid, nan, none, negative, index, dtype, type, value, zero, infinite, inf, missing, undefined, diameter, length, mdot, pressure, height, elevation, junction, reference, KeyError, ValueError, MatrixRankWarning, singular

## What it means
One or more network elements have parameters that violate pandapipes' type or range
constraints. NaN values propagate through the Jacobian and the solver diverges immediately,
often surfacing as `PipeflowNotConverged` or a NumPy "invalid value" warning.

## Source basis
This pack follows pandapipes component tables and physical models: pipes connect
`from_junction` to `to_junction`, `length_km` is pipe length in km, pipe diameter must be
positive, roughness is given in mm, junction pressure/temperature/elevation are physical
inputs, and pipe pressure loss depends on length, diameter, friction, and height
difference. Values such as `inner_diameter_mm < 20`, `length_km > 50`, or
`abs(height_m) > 1000` are Explain heuristics for suspicious corrupted input; they are
not pandapipes' official validity ranges.

## Cause 1: NaN in mdot_kg_per_s (sink / source)
A sink or source with an unset mass flow injects NaN into the mass balance.
```python
# Find NaN sinks:
print(net.sink[net.sink['mdot_kg_per_s'].isna()])

# Fix: fill with a numeric value or remove the element
net.sink['mdot_kg_per_s'] = net.sink['mdot_kg_per_s'].fillna(0.0)
```

## Cause 2: Non-positive pipe diameter or length
`inner_diameter_mm` and `length_km` must be strictly positive. A zero diameter is a
closed pipe (infinite resistance); a zero length makes the friction term singular.
```python
print(net.pipe[(net.pipe.inner_diameter_mm <= 0) | (net.pipe.length_km <= 0)])
# Fix:
net.pipe.at[bad_idx, 'inner_diameter_mm'] = 200.0   # mm
net.pipe.at[bad_idx, 'length_km'] = 0.5             # km
```

## Cause 3: Unrealistically small but positive pipe diameter
A diameter can pass the `> 0` type/range check and still be physically unusable. A pipe
diameter of `0.001` mm is effectively blocked; pressure loss scales very strongly with
diameter, so the Newton matrix can become singular. A preceding `MatrixRankWarning:
Matrix is exactly singular` is consistent with this failure mode.
```python
print(net.pipe[net.pipe.inner_diameter_mm < 20][["from_junction", "to_junction", "inner_diameter_mm"]])
```

## Cause 4: Unrealistically large pipe length
`length_km` is kilometres. Accidentally entering metres as kilometres (for example `500`
instead of `0.5`) multiplies the length-dependent pressure-loss contribution and can directly cause
non-convergence.
```python
print(net.pipe[net.pipe.length_km > 50][["from_junction", "to_junction", "length_km"]])
```

## Cause 5: Non-existent junction index
Connecting an element to a junction index that does not exist raises an error or silently
leaves the element disconnected.
```python
all_j = set(net.junction.index)
for table in ['pipe', 'sink', 'source', 'ext_grid', 'valve', 'pump']:
    df = net[table]
    if df.empty:
        continue
    jcols = [c for c in df.columns if 'junction' in c]   # junction, from_junction, to_junction
    for col in jcols:
        bad = df[~df[col].isin(all_j)]
        if not bad.empty:
            print(f"net.{table}['{col}'] references missing junctions:", bad.index.tolist())
```

If the evidence says `net.pipe.loc[0, "to_junction"] = 999` and the network has only
6 junctions, the root cause is the corrupted pipe reference, not a generic Python indexing
bug.

## Cause 6: Unphysical pressure, temperature, or elevation values
Pressures in pandapipes are absolute [bar] and temperatures are in Kelvin, so physically
they should be > 0. Note `pn_bar` (junction) and `tfluid_k` (junction) are *initial values*
for the iteration, while `p_bar`/`t_k` on the `ext_grid` are the enforced boundary
conditions — but a non-positive value in any of them is still unphysical and a likely bug.
`height_m` is an elevation in metres; extreme corrupted values can dominate hydrostatic
pressure even if all pressures are positive.
```python
print(net.junction[net.junction.pn_bar <= 0])     # initial pressure guess
print(net.ext_grid[net.ext_grid.p_bar <= 0])       # enforced pressure boundary
print(net.junction[net.junction.tfluid_k <= 0])    # initial / fluid temperature
print(net.junction[net.junction.height_m.abs() > 1000])
```

## Cause 7: Wrong units (m vs km, m vs mm)
pandapipes is strict about units: pipe length in **km**, diameter in **mm**, pressure in
**bar**, mass flow in **kg/s**, temperature in **K**, height in **m**.
```python
# Wrong: a 500 m pipe entered as length_km=500 (means 500 km!)
# Right: length_km=0.5
print(net.pipe[['length_km', 'inner_diameter_mm']])  # sanity-check magnitudes
```

## Quick checklist
- [ ] `net.sink.mdot_kg_per_s.isna().any()` → False (and same for `net.source`)
- [ ] `(net.pipe.inner_diameter_mm > 0).all()` → True
- [ ] `(net.pipe.length_km > 0).all()` → True
- [ ] All junction indices referenced by elements exist in `net.junction.index`
- [ ] `net.junction.pn_bar.min() > 0` and `net.junction.tfluid_k.min() > 0`
