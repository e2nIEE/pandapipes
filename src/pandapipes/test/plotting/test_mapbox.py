# test_mapbox_plot.py

import copy
import pandas as pd
import pytest
import pandapipes as pp
import plotly.graph_objects as go

from pandapipes.plotting.mapbox_plot import (
    set_mapbox_token,
    _get_mapbox_token,
    _on_map_test,
    create_mapbox_figure
)
from pandapipes.plotting.simple_plot import simple_plot

def create_test_network():
    """
    Build the "2-Consumer Loop" network with geodata, pipes, HX, ext_grid, and a pump.
    """
    # Coordinates
    coords_main = [
        (13.377704, 52.509669),  # j0: Plant
        (13.380000, 52.509700),  # j1
        (13.382000, 52.509720),  # j2
        (13.384000, 52.509740),  # j3
    ]
    coords_h = {
        'h1_feed':   (13.380500, 52.510500),
        'h1_return': (13.380300, 52.510300),
        'h2_feed':   (13.382500, 52.510500),
        'h2_return': (13.382300, 52.510300),
    }

    # Build network
    net = pp.create_empty_network(fluid='water', name='2-Consumer Loop')
    main_j = [pp.create_junction(net, pn_bar=3, tfluid_k=353.15) for _ in coords_main]
    j_feed1 = pp.create_junction(net, pn_bar=3, tfluid_k=353.15)
    j_ret1  = pp.create_junction(net, pn_bar=3, tfluid_k=353.15)
    j_feed2 = pp.create_junction(net, pn_bar=3, tfluid_k=353.15)
    j_ret2  = pp.create_junction(net, pn_bar=3, tfluid_k=353.15)

    # Assign geodata
    all_coords = coords_main + list(coords_h.values())
    all_juncs  = main_j + [j_feed1, j_ret1, j_feed2, j_ret2]
    net.junction_geodata = pd.DataFrame({
        'lon': [lon for lon, lat in all_coords],
        'lat': [lat for lon, lat in all_coords],
    }, index=all_juncs)

    # Pipes
    for i in range(len(main_j)-1):
        pp.create_pipe_from_parameters(net,
            main_j[i], main_j[i+1], length_km=0.1, diameter_m=0.1,
            k_mm=0.02, alpha_w_per_m2k=10, text_k=293.15,
            name=f'loop_{i}'
        )
    pp.create_pipe_from_parameters(net, main_j[1], j_feed1,   0.02, 0.05,
        k_mm=0.02, alpha_w_per_m2k=10, text_k=293.15, name='h1_supp'
    )
    pp.create_pipe_from_parameters(net, j_ret1,   main_j[2], 0.02, 0.05,
        k_mm=0.02, alpha_w_per_m2k=10, text_k=293.15, name='h1_ret'
    )
    pp.create_pipe_from_parameters(net, main_j[2], j_feed2,   0.02, 0.05,
        k_mm=0.02, alpha_w_per_m2k=10, text_k=293.15, name='h2_supp'
    )
    pp.create_pipe_from_parameters(net, j_ret2,   main_j[3], 0.02, 0.05,
        k_mm=0.02, alpha_w_per_m2k=10, text_k=293.15, name='h2_ret'
    )

    # Heat exchangers
    pp.create_heat_exchanger(net, from_junction=j_feed1, to_junction=j_ret1,
                             qext_w=15e3, diameter_m=0.05,
                             length_km=0.005, sections=2, name='HX1')
    pp.create_heat_exchanger(net, from_junction=j_feed2, to_junction=j_ret2,
                             qext_w=25e3, diameter_m=0.05,
                             length_km=0.005, sections=2, name='HX2')

    # External grid and pump
    pp.create_ext_grid(net, junction=main_j[0], p_bar=3.0, t_k=353.15, name='Plant')
    pp.create_circ_pump_const_mass_flow(net,
        return_junction=main_j[-1], flow_junction=main_j[0],
        mdot_flow_kg_per_s=0.1, p_flow_bar=3.5, t_flow_k=353.15,
        name='LoopPump'
    )

    # Run a pipeflow to generate res_pipe
    pp.pipeflow(net, mode='sequential', friction_model='swamee-jain')
    return net

def test_set_and_get_token():
    token = "test-token-123"
    set_mapbox_token(token)
    assert _get_mapbox_token() == token

def test_on_map_test_behavior():
    # valid bounds
    assert _on_map_test(0, 0)
    assert _on_map_test(180,  90)
    assert _on_map_test(-180, -90)
    # out of bounds
    assert not _on_map_test(200, 0)
    assert not _on_map_test(0, -100)

def test_create_mapbox_figure_basic():
    net = create_test_network()
    fig = create_mapbox_figure(net, mapbox_access_token="dummy-token")
    assert isinstance(fig, go.Figure)
    # must contain at least the junctions layer
    names = [trace.name for trace in fig.data]
    assert "Junctions" in names

def test_pipe_color_override():
    net = create_test_network()
    fig = create_mapbox_figure(net, mapbox_access_token="dummy", pipe_color="purple")
    # all pipe traces should use the overridden 'purple' color
    pipe_traces = [t for t in fig.data if t.name.startswith("Pipe")]
    assert pipe_traces, "No pipe traces found"
    for t in pipe_traces:
        assert t.line.color == "purple"

def test_temperature_coloring_and_colorbar():
    net = create_test_network()
    # override res_pipe to guarantee a t_to_k field
    net.res_pipe = net.pipe.copy()
    net.res_pipe["t_to_k"] = 300 + net.res_pipe.index
    fig = create_mapbox_figure(
        net,
        mapbox_access_token="dummy",
        pipe_temperature_coloring=True,
        show_colorbar=True
    )
    # exactly one trace should carry the colorbar
    cb_traces = [
        t for t in fig.data
        if hasattr(t.marker, "showscale") and t.marker.showscale
    ]
    assert len(cb_traces) == 1

def test_simple_plot_returns_plotly_figure(monkeypatch):
    net = create_test_network()
    # ensure token comes from env if not passed explicitly
    monkeypatch.setenv("MAPBOX_ACCESS_TOKEN", "env-token")
    fig = simple_plot(
        net,
        use_mapbox=True,
        mapbox_access_token=None,
        show_plot=False
    )
    assert isinstance(fig, go.Figure)

if __name__ == '__main__':
    pytest.main(["test_mapbox.py"])
