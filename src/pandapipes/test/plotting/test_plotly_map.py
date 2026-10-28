# Copyright (c) 2020-2026 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import pytest

import pandapipes as pp
from pandapipes.plotting import create_map_figure

go = pytest.importorskip("plotly.graph_objects")
if not hasattr(go, "Scattermap"):
    pytest.skip("plotly >= 5.24 is required", allow_module_level=True)


@pytest.fixture
def lonlat_net():
    """Small water network with valid WGS84 coordinates and results."""
    net = pp.create_empty_network(fluid="water")
    coords = [(14.30, 51.75), (14.31, 51.76), (14.32, 51.76), (14.33, 51.75)]
    for x, y in coords:
        j = pp.create_junction(net, pn_bar=5, tfluid_k=330)
        net.junction_geodata.loc[j, ["x", "y"]] = (x, y)
    pp.create_ext_grid(net, junction=0, p_bar=5, t_k=330)
    pp.create_pipe_from_parameters(net, 0, 1, length_km=1, inner_diameter_mm=100, u_w_per_m2k=1)
    pp.create_pipe_from_parameters(net, 1, 2, length_km=1, inner_diameter_mm=100, u_w_per_m2k=1)
    pp.create_pipe_from_parameters(net, 2, 3, length_km=1, inner_diameter_mm=100, u_w_per_m2k=1)
    pp.create_sink(net, junction=3, mdot_kg_per_s=0.1)
    pp.pipeflow(net, mode="all")
    return net


def _names(fig):
    return {trace.name for trace in fig.data}


def test_basic_figure_needs_no_token(lonlat_net):
    fig = create_map_figure(lonlat_net)
    assert isinstance(fig, go.Figure)
    assert all(isinstance(t, go.Scattermap) for t in fig.data)
    assert {"Pipes", "Junctions", "ext_grid", "sink"} <= _names(fig)
    assert fig.layout.map.style == "carto-positron"
    assert "accesstoken" not in fig.layout.map.to_plotly_json()


def test_pipes_are_one_disconnected_trace(lonlat_net):
    pipes = next(t for t in create_map_figure(lonlat_net).data if t.name == "Pipes")
    # 3 pipes with 2 points each + a None separator after each pipe
    assert len(pipes.lon) == 9
    assert sum(v is None for v in pipes.lon) == 3


def test_result_coloring_adds_colorbar(lonlat_net):
    fig = create_map_figure(lonlat_net, pipe_result="t_to_k", colorbar_title="T [K]")
    bars = [t for t in fig.data if t.marker.showscale]
    assert len(bars) == 1
    assert bars[0].marker.colorbar.title.text == "T [K]"
    res = lonlat_net.res_pipe["t_to_k"]
    assert bars[0].marker.cmin == pytest.approx(res.min())
    assert bars[0].marker.cmax == pytest.approx(res.max())


def test_result_coloring_requires_pipeflow(lonlat_net):
    lonlat_net.res_pipe = lonlat_net.res_pipe.iloc[0:0]
    with pytest.raises(ValueError, match="pipeflow"):
        create_map_figure(lonlat_net, pipe_result="t_to_k")


def test_unknown_result_column(lonlat_net):
    with pytest.raises(ValueError, match="not a column"):
        create_map_figure(lonlat_net, pipe_result="does_not_exist")


def test_projected_coordinates_give_helpful_error(lonlat_net):
    lonlat_net.junction_geodata["x"] += 3_400_000
    with pytest.raises(ValueError, match="convert_epsg_junction_geodata"):
        create_map_figure(lonlat_net)


def test_projected_pipe_geodata_gives_helpful_error(lonlat_net):
    lonlat_net.pipe_geodata.loc[0, "coords"] = [[3_416_947.6, 5_369_956.6], [3_417_001.2, 5_369_931.0]]
    with pytest.raises(ValueError, match="pipe route"):
        create_map_figure(lonlat_net)


def test_convert_epsg_pipe_geodata_roundtrip():
    from pandapipes.plotting import convert_epsg_pipe_geodata
    net = pp.create_empty_network(fluid="water")
    for _ in range(2):
        pp.create_junction(net, pn_bar=5, tfluid_k=330)
    pp.create_pipe_from_parameters(net, 0, 1, length_km=1, inner_diameter_mm=100,
                                   geodata=[[7.88, 48.46], [7.89, 48.47]])
    original = net.pipe_geodata.at[0, "coords"]
    convert_epsg_pipe_geodata(net, epsg_in=4326, epsg_out=31467)
    assert net.pipe_geodata.at[0, "coords"][0][0] > 1_000_000
    convert_epsg_pipe_geodata(net, epsg_in=31467, epsg_out=4326)
    flat = [v for point in net.pipe_geodata.at[0, "coords"] for v in point]
    assert flat == pytest.approx([v for point in original for v in point], abs=1e-6)


def test_fixed_zoom_and_hidden_components(lonlat_net):
    fig = create_map_figure(lonlat_net, zoom=12, show_components=False, show_junctions=False)
    assert fig.layout.map.zoom == 12
    assert _names(fig) == {"Pipes"}


def test_fits_map_to_network_by_default(lonlat_net):
    layout_map = create_map_figure(lonlat_net).layout.map
    assert layout_map.center.lon == pytest.approx(14.315)
    assert 51.75 < layout_map.center.lat < 51.76
    # the network spans ~0.03 deg lon; a fitted view shows it at a city-scale zoom
    assert 12 < layout_map.zoom < 16



def test_valves_and_pumps_are_drawn():
    net = pp.create_empty_network(fluid="water")
    for x, y in [(14.30, 51.75), (14.31, 51.76), (14.32, 51.76), (14.33, 51.75)]:
        j = pp.create_junction(net, pn_bar=5, tfluid_k=330)
        net.junction_geodata.loc[j, ["x", "y"]] = (x, y)
    pp.create_pump_from_parameters(net, 0, 1, "P1")
    pp.create_valve(net, junction=1, element=2, et="ju", inner_diameter_mm=100)
    pp.create_pipe_from_parameters(net, 2, 3, length_km=1, inner_diameter_mm=100)
    assert {"pump", "Valves", "Pipes"} <= _names(create_map_figure(net))


def test_pipes_without_flow_are_grey_and_excluded_from_color_range(lonlat_net):
    lonlat_net.res_pipe.loc[2, "mdot_from_kg_per_s"] = 0.0
    lonlat_net.res_pipe.loc[2, "t_to_k"] = 250.0  # ambient temperature of an idle pipe
    fig = create_map_figure(lonlat_net, pipe_result="t_to_k")
    assert "Pipes without flow" in _names(fig)
    bar = next(t for t in fig.data if t.marker.showscale)
    assert bar.marker.cmin > 300
    fig = create_map_figure(lonlat_net, pipe_result="t_to_k", no_flow_color=None)
    assert "Pipes without flow" not in _names(fig)


def test_junctions_without_coordinates_are_skipped_with_warning(lonlat_net):
    lonlat_net.junction_geodata.loc[3, ["x", "y"]] = float("nan")
    with pytest.warns(UserWarning, match="no coordinates"):
        fig = create_map_figure(lonlat_net)
    pipes = next(t for t in fig.data if t.name == "Pipes")
    assert sum(v is None for v in pipes.lon) == 2  # the pipe to junction 3 is not drawn
