# Copyright (c) 2020-2026 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

"""Interactive web-map plot of a pandapipes network using plotly (MapLibre).

The map background uses plotly's built-in token-free styles (e.g. "carto-positron",
"open-street-map"), so no account or API key is needed. Plotly >= 5.24 is required.
"""

# plotly is an optional dependency, so it is imported inside the functions that draw (lazy import).
# pylint: disable=import-outside-toplevel
import warnings

import numpy as np
import pandas as pd

# Branch-like components drawn as a line between two junctions: table -> (from column, to column).
_LINE_COMPONENTS = {
    "pump": ("from_junction", "to_junction"),
    "compressor": ("from_junction", "to_junction"),
    "flow_control": ("from_junction", "to_junction"),
    "press_control": ("from_junction", "to_junction"),
    "heat_exchanger": ("from_junction", "to_junction"),
    "heat_consumer": ("from_junction", "to_junction"),
    "circ_pump_mass": ("flow_junction", "return_junction"),
    "circ_pump_pressure": ("flow_junction", "return_junction"),
}

# Components drawn as a marker on the junction they are connected to: table -> color.
_NODE_COMPONENTS = {
    "ext_grid": "#d62728",
    "source": "#2ca02c",
    "sink": "#ff7f0e",
    "mass_storage": "#9467bd",
}

_N_COLOR_BINS = 32

_CRS_HINT = ("The {what} coordinates are not valid longitude/latitude values (WGS84). Convert them "
             "first, e.g. with pandapipes.plotting.convert_epsg_junction_geodata(net, epsg_in=<your "
             "EPSG>, epsg_out=4326) and, if the network has pipe_geodata, "
             "pandapipes.plotting.convert_epsg_pipe_geodata(...).")


def create_map_figure(net, map_style="carto-positron", zoom=None, pipe_color="#1f77b4",
                      pipe_result=None, colorscale="Bluered", colorbar_title=None,
                      no_flow_color="#b0b0b0", show_components=True, show_junctions=True, height=700):
    """Create an interactive plotly figure of a pandapipes network on a web map.

    The junction geodata (``net.junction_geodata``) must hold WGS84 coordinates with
    ``x`` = longitude and ``y`` = latitude. If your coordinates use a projected CRS, convert them
    first, e.g. with :func:`pandapipes.plotting.convert_epsg_junction_geodata`. No access token
    is needed for the built-in map styles. The map tiles are loaded by the browser from public
    tile servers, so an internet connection is needed to see the background.

    :param net: The pandapipes network
    :type net: pandapipesNet
    :param map_style: Token-free plotly map style, e.g. "carto-positron" (default),
        "carto-darkmatter", "open-street-map" or "white-bg" (no background)
    :type map_style: str
    :param zoom: Fixed zoom level. If None (default), the map is fitted to the network
    :type zoom: float, optional
    :param pipe_color: Color of the pipes if ``pipe_result`` is None
    :type pipe_color: str
    :param pipe_result: Column of ``net.res_pipe`` used to color the pipes (e.g. "t_to_k" or
        "v_mean_m_per_s"). Requires a prior pipeflow calculation
    :type pipe_result: str, optional
    :param colorscale: Named plotly colorscale used for ``pipe_result``
    :type colorscale: str
    :param colorbar_title: Title of the colorbar, defaults to the ``pipe_result`` name
    :type colorbar_title: str, optional
    :param no_flow_color: Color of pipes without flow when ``pipe_result`` is used. Such pipes
        are excluded from the color range (e.g. their temperature is just the ambient
        temperature). Pass None to color them like all other pipes
    :type no_flow_color: str, optional
    :param show_components: Draw ext_grids, sources, sinks, pumps, valves, etc.
    :type show_components: bool
    :param show_junctions: Draw junction markers
    :type show_junctions: bool
    :param height: Figure height in pixels
    :type height: int
    :return: The plotly figure (use ``fig.show()`` or ``fig.write_html(...)``)
    :rtype: plotly.graph_objects.Figure
    """
    try:
        import plotly.graph_objects as go
    except ImportError as err:  # pragma: no cover
        raise ImportError("plotly >= 5.24 is required for create_map_figure. "
                          "Install it with: pip install 'pandapipes[plotting]'") from err
    if not hasattr(go, "Scattermap"):  # pragma: no cover
        raise ImportError("plotly >= 5.24 is required (go.Scattermap not found).")

    geo = _junction_lonlat(net)
    if geo.empty:
        raise ValueError("net.junction_geodata is empty - there is nothing to plot.")
    fig = go.Figure()

    pipes = _pipe_lines(net, geo)
    if pipe_result is None:
        _add_lines(fig, [pipes], "Pipes", pipe_color, width=2)
    else:
        _add_colored_pipes(fig, net, pipes, pipe_result, colorscale, colorbar_title, no_flow_color)

    if show_components:
        _add_components(fig, net, geo)
    if show_junctions:
        fig.add_trace(go.Scattermap(
            lon=geo["lon"], lat=geo["lat"], mode="markers", name="Junctions",
            marker={"size": 4, "color": "#444444"},
            text=[f"Junction {idx}" for idx in geo.index], hoverinfo="text"))

    center, fitted_zoom = _fit_view(geo, height)
    layout_map = {"style": map_style, "center": center, "zoom": fitted_zoom if zoom is None else zoom}
    fig.update_layout(map=layout_map, height=height, margin={"l": 0, "r": 0, "t": 0, "b": 0},
                      legend={"x": 0.01, "y": 0.99})
    return fig


def _fit_view(geo, height_px, width_px=1000, margin=0.3):
    """Return map center and zoom so that all junctions are visible (web-mercator, 512 px tiles)."""
    def mercator_y(lat):
        return np.degrees(np.log(np.tan(np.pi / 4 + np.radians(lat) / 2)))

    lon_span = max(float(np.ptp(geo["lon"])), 1e-6)
    y_min, y_max = mercator_y(geo["lat"].min()), mercator_y(geo["lat"].max())
    y_span = max(float(y_max - y_min), 1e-6)
    zoom = min(np.log2(360 * width_px / (512 * lon_span)), np.log2(360 * height_px / (512 * y_span)))
    center_y = (y_min + y_max) / 2
    center = {"lon": float((geo["lon"].min() + geo["lon"].max()) / 2),
              "lat": float(np.degrees(2 * np.arctan(np.exp(np.radians(center_y))) - np.pi / 2))}
    return center, float(np.clip(zoom - margin, 0, 18))


def _junction_lonlat(net):
    """Return junction coordinates as DataFrame with columns lon/lat, validated as WGS84."""
    geodata = net.get("junction_geodata")
    if geodata is None or geodata.empty:
        return _empty_frame()
    if not {"x", "y"}.issubset(geodata.columns):
        raise ValueError("net.junction_geodata needs the columns 'x' (longitude) and "
                         "'y' (latitude).")
    geo = geodata[["x", "y"]].rename(columns={"x": "lon", "y": "lat"}).astype(float)
    missing = geo.isna().any(axis=1)
    if missing.any():
        warnings.warn(f"{int(missing.sum())} junctions have no coordinates; they and the elements "
                      f"connected to them are not drawn.")
        geo = geo[~missing]
    if not _is_lonlat(geo["lon"], geo["lat"]):
        raise ValueError(_CRS_HINT.format(what="junction"))
    return geo


def _is_lonlat(lon, lat):
    return bool(np.all(np.abs(lon) <= 180) and np.all(np.abs(lat) <= 90))


def _empty_frame():
    return pd.DataFrame(columns=["lon", "lat"])


def _pipe_lines(net, geo):
    """Return {pipe index: (lons, lats)} - from pipe_geodata if present, else straight lines."""
    lines = {}
    pipe_geodata = net.get("pipe_geodata")
    has_coords = pipe_geodata is not None and "coords" in getattr(pipe_geodata, "columns", [])
    for idx, row in net.pipe.iterrows():
        coords = pipe_geodata.at[idx, "coords"] if has_coords and idx in pipe_geodata.index \
            else None
        if coords is not None and len(coords) > 1:
            lons, lats = zip(*[(c[0], c[1]) for c in coords])
            if not _is_lonlat(np.array(lons), np.array(lats)):
                raise ValueError(_CRS_HINT.format(what="pipe route"))
        elif row["from_junction"] in geo.index and row["to_junction"] in geo.index:
            pair = geo.loc[[row["from_junction"], row["to_junction"]]]
            lons, lats = pair["lon"].tolist(), pair["lat"].tolist()
        else:
            continue
        lines[idx] = (list(lons), list(lats))
    return lines


def _joined(lines, texts=None):
    """Join several lines into one trace, separated by None so they stay disconnected."""
    lon, lat, text = [], [], []
    for idx, (lons, lats) in lines.items():
        lon += list(lons) + [None]
        lat += list(lats) + [None]
        label = texts[idx] if texts is not None else None
        text += [label] * len(lons) + [None]
    return lon, lat, text


def _add_lines(fig, line_dicts, name, color, width, texts=None):
    import plotly.graph_objects as go
    lines = {}
    for line_dict in line_dicts:
        lines.update(line_dict)
    if not lines:
        return
    lon, lat, text = _joined(lines, texts)
    fig.add_trace(go.Scattermap(
        lon=lon, lat=lat, mode="lines", name=name, line={"color": color, "width": width},
        text=text if texts is not None else None,
        hoverinfo="text" if texts is not None else "skip"))


def _add_colored_pipes(fig, net, pipes, pipe_result, colorscale, colorbar_title, no_flow_color):
    """Color pipes by a res_pipe column using few traces (one per color bin) for speed."""
    import plotly.graph_objects as go
    from plotly.colors import sample_colorscale

    res = net.get("res_pipe")
    if res is None or res.empty:
        raise ValueError("net.res_pipe is empty - run a pipeflow calculation before using "
                         "'pipe_result'.")
    if pipe_result not in res.columns:
        raise ValueError(f"'{pipe_result}' is not a column of net.res_pipe. "
                         f"Available: {', '.join(map(str, res.columns))}")
    values = res[pipe_result].astype(float)
    no_flow = pd.Series(False, index=values.index)
    if no_flow_color is not None and "mdot_from_kg_per_s" in res.columns:
        no_flow = res["mdot_from_kg_per_s"].abs() < 1e-9
        if no_flow.all():  # nothing to compare against, color everything
            no_flow[:] = False
    colored = values[~no_flow]
    vmin, vmax = float(colored.min()), float(colored.max())
    span = vmax - vmin if vmax > vmin else 1.0
    bins = np.clip(((values - vmin) / span * _N_COLOR_BINS).astype(int), 0, _N_COLOR_BINS - 1)
    bin_colors = sample_colorscale(colorscale, [(b + 0.5) / _N_COLOR_BINS
                                                for b in range(_N_COLOR_BINS)])
    texts = {idx: f"Pipe {idx}: {values[idx]:.4g} ({pipe_result})" for idx in values.index}
    if no_flow.any():
        idle = {idx: pipes[idx] for idx in values.index[no_flow] if idx in pipes}
        _add_lines(fig, [idle], "Pipes without flow", no_flow_color, width=3, texts=texts)
    for b in range(_N_COLOR_BINS):
        members = {idx: pipes[idx] for idx in values.index[(bins == b) & ~no_flow] if idx in pipes}
        _add_lines(fig, [members], f"Pipes ({pipe_result})", bin_colors[b], width=3,
                   texts=texts)
        if members:
            fig.data[-1].showlegend = False
    # An invisible marker trace carries the colorbar.
    first = next(iter(pipes.values()))
    fig.add_trace(go.Scattermap(
        lon=[first[0][0]] * 2, lat=[first[1][0]] * 2, mode="markers", hoverinfo="skip",
        showlegend=False, marker={
            "size": 0, "color": [vmin, vmax], "cmin": vmin, "cmax": vmax, "colorscale": colorscale,
            "showscale": True,
            "colorbar": {"title": colorbar_title or pipe_result, "y": 0.88, "yanchor": "top", "len": 0.75}}))


def _add_components(fig, net, geo):
    import plotly.graph_objects as go

    # Valves: line between the junction and the element if et == 'ju', else marker on the junction.
    if "valve" in net and not net.valve.empty:
        valve_lines, valve_points = {}, {}
        for idx, row in net.valve.iterrows():
            j = row["junction"]
            if row.get("et", "ju") == "ju" and row["element"] in geo.index and j in geo.index:
                pair = geo.loc[[j, row["element"]]]
                valve_lines[idx] = (pair["lon"].tolist(), pair["lat"].tolist())
            elif j in geo.index:
                valve_points[idx] = j
        _add_lines(fig, [valve_lines], "Valves", "#8c564b", width=5)
        if valve_points:
            pts = geo.loc[list(valve_points.values())]
            fig.add_trace(go.Scattermap(
                lon=pts["lon"], lat=pts["lat"], mode="markers", name="Valves",
                marker={"size": 9, "color": "#8c564b"},
                text=[f"Valve {i}" for i in valve_points], hoverinfo="text"))

    for table, (from_col, to_col) in _LINE_COMPONENTS.items():
        if table not in net or net[table].empty:
            continue
        lines = {}
        for idx, row in net[table].iterrows():
            if row[from_col] in geo.index and row[to_col] in geo.index:
                pair = geo.loc[[row[from_col], row[to_col]]]
                lines[idx] = (pair["lon"].tolist(), pair["lat"].tolist())
        texts = {idx: f"{table} {idx}" for idx in lines}
        _add_lines(fig, [lines], table, "#7f7f7f", width=5, texts=texts)

    for table, color in _NODE_COMPONENTS.items():
        if table not in net or net[table].empty:
            continue
        rows = net[table][net[table]["junction"].isin(geo.index)]
        if rows.empty:
            continue
        pts = geo.loc[rows["junction"]]
        fig.add_trace(go.Scattermap(
            lon=pts["lon"], lat=pts["lat"], mode="markers", name=table,
            marker={"size": 7, "color": color},
            text=[f"{table} {i}" for i in rows.index], hoverinfo="text"))
