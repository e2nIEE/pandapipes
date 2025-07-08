import pandapipes as pp
import pandas as pd
from pandapipes.plotting.simple_plot import simple_plot


# ── 1) COORDINATES ────────────────────────────────────────────────────────────
coords_main = [
    (14.369806, 51.760483),  # j0: Plant
    (14.371217, 51.761439),  # j1
    (14.371767, 51.761465),  # j2: branch to 3 houses
    (14.372488, 51.761515),  # j3: branch to 2 houses
    (14.372856, 51.761525),  # j4
    (14.372706, 51.762125),  # j5: branch to 2 houses
    (14.372628, 51.762463),  # j6
    (14.371670, 51.762480),  # j7: branch to 1 house
    (14.370804, 51.762546),  # j8
    (14.371021, 51.761573),  # j9
    (14.369806, 51.760483),  # j10 back to Plant
]

coords_houses = {
    "h21": (14.371512, 51.761698),
    "h22": (14.371753, 51.761690),
    "h23": (14.372008, 51.761675),
    "h31": (14.372327, 51.761675),
    "h32": (14.372571, 51.761661),
    "h51": (14.372456, 51.762126),
    "h52": (14.372907, 51.762056),
    "h71": (14.371611, 51.762403),
}

# ── 2) BUILD NETWORK ──────────────────────────────────────────────────────────
net = pp.create_empty_network(fluid="water")

# 2a) Main junctions
main_junc = [pp.create_junction(net, pn_bar=1, tfluid_k=293.15) for _ in coords_main]

# 2b) House junctions
house_junc = {
    name: pp.create_junction(net, pn_bar=1, tfluid_k=293.15)
    for name in coords_houses
}

# 2c) Geodata for plotting
all_coords = coords_main + list(coords_houses.values())
all_juncs  = main_junc + list(house_junc.values())
net.junction_geodata = pd.DataFrame({
    "lon": [c[0] for c in all_coords],
    "lat": [c[1] for c in all_coords],
}, index=all_juncs)

# ── 3) PIPES ─────────────────────────────────────────────────────────────────
# 3a) Main closed loop
for i in range(len(main_junc) - 1):
    pp.create_pipe_from_parameters(
        net,
        from_junction=main_junc[i],
        to_junction  =main_junc[i+1],
        length_km    =0.1,    # placeholder length
        diameter_m   =0.1,
        name         =f"loop_{i}"
    )

# 3b) Branch stubs for each house
branch_map = {
    "h21": main_junc[2], "h22": main_junc[2], "h23": main_junc[2],
    "h31": main_junc[3], "h32": main_junc[3],
    "h51": main_junc[5], "h52": main_junc[5],
    "h71": main_junc[7],
}
for hn, jh in house_junc.items():
    pp.create_pipe_from_parameters(
        net,
        from_junction=branch_map[hn],
        to_junction  =jh,
        length_km    =0.02,   # ~20 m stub
        diameter_m   =0.05,
        name         =f"branch_{hn}"
    )

# ── 4) SUPPLY & PUMP ─────────────────────────────────────────────────────────
pp.create_ext_grid(
    net,
    junction=main_junc[0],
    p_bar=1.0,
    t_k=353.15,
    name="Plant"
)
pp.create_circ_pump_const_pressure(
    net,
    return_junction=main_junc[-2],  # j9
    flow_junction =main_junc[0],     # back to plant
    p_flow_bar    =2.0,
    plift_bar     =0.5,
    t_flow_k      =353.15,
    name="Loop Pump"
)

# ── 5) HEAT CONSUMERS & SINKS ────────────────────────────────────────────────
for hn, jh in house_junc.items():
    # heat consumer at the house
    pp.create_heat_consumer(
        net,
        from_junction=jh,
        to_junction  =jh,
        diameter_m   =0.05,
        controlled_mdot_kg_per_s=0.2,
        qext_w       =2e4,
        name         =f"hc_{hn}"
    )
    # sink to remove mass
    pp.create_sink(
        net,
        junction    =jh,
        mdot_kg_per_s=0.2,
        name         =f"sink_{hn}"
    )

# ── 6) SOLVE & DEBUG ─────────────────────────────────────────────────────────
pp.pipeflow(net)

# Print tables to verify element placement
print("\n--- Pipes ---")
print(net.pipe)
print("\n--- Heat Consumers ---")
print(net.heat_consumer)
print("\n--- Sinks ---")
print(net.sink)

# ── 7) PLOT ───────────────────────────────────────────────────────────────────
simple_plot(
    net,
    use_mapbox=True,
    mapbox_access_token="pk.eyJ1IjoiYWZhcWphbXNoYWlkIiwiYSI6ImNtNnFwdTVlajFxajUya3M0Nm9jZDRuZHUifQ.jberIPfticU3j0aiR2obQg",
    map_style="light",
    zoom=16,
    pipe_temperature_coloring=False,
    pipe_color="red",
    show_colorbar=False,
    line_width=2,  # Make pipes more visible
    junction_size=20,  # Make junctions more visible
    renderer='browser'  # Specify the renderer (default is 'browser')
)


