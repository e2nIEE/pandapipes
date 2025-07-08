"""
mapbox_plot.py

This module adds Mapbox‐based (Plotly) plotting functionality to pandapipes.
It provides helper functions for setting/retrieving the Mapbox token and a
main function to create an interactive Plotly figure that plots all network
components on a real map.

Inspired by the pandapower implementation, it supports:
  • Junctions (from net.junction_geodata)
  • Pipes – if net.pipe_geodata exists, its “coords” are used;
      otherwise, the from/to junctions (via net.pipe) are used.
  • External Grids, Sources, and Sinks – using the junction reference.
  • Valves – if a dedicated “junction” column exists use it; otherwise compute
      the midpoint from “from_junction” and “to_junction”.
  • Pumps – drawn as lines between from/to junctions.
  • Line‐type components (Heat exchangers, Press controls, Compressors,
      Flow controls, Heat consumers) – drawn as lines using “from_junction” and “to_junction”.
      
New functionality:
  • Optionally color pipes based on (e.g.) temperature values (from net.res_pipe)
    and display a colorbar.
"""
from pandapipes.plotting import colormaps

import os
import plotly.graph_objects as go
import matplotlib.colors as mcolors

# Module‐level variable to hold a user‐defined Mapbox access token.
_MAPBOX_TOKEN = None


def set_mapbox_token(token):
    """
    Save the user's Mapbox API token for authenticated access.
    
    Parameters
    ----------
    token : str
         A valid Mapbox access token.
    """
    global _MAPBOX_TOKEN
    _MAPBOX_TOKEN = token


def _get_mapbox_token():
    """
    Retrieve the stored Mapbox token. If none has been set,
    try reading from the environment variable MAPBOX_ACCESS_TOKEN.
    
    Returns
    -------
    token : str
         The Mapbox token (or an empty string if none is available).
    """
    global _MAPBOX_TOKEN
    if _MAPBOX_TOKEN is None:
        _MAPBOX_TOKEN = os.getenv("MAPBOX_ACCESS_TOKEN", "")
    return _MAPBOX_TOKEN


def _on_map_test(x, y):
    """
    Test whether given coordinates (x, y) are plausible geographic coordinates.
    (Assuming x is longitude and y is latitude.)
    
    Parameters
    ----------
    x : float
         Longitude.
    y : float
         Latitude.
         
    Returns
    -------
    bool
         True if x is between -180 and 180 and y between -90 and 90.
    """
    return (-180 <= x <= 180) and (-90 <= y <= 90)


def create_mapbox_figure(net, mapbox_access_token=None, map_style="streets", zoom=10,
                         pipe_temperature_coloring=False, pipe_temperature_field="t_to_k",
                         pipe_color=None, show_colorbar=False, 
                         pipe_temperature_colorscale=[[0, "blue"], [1, "red"]],
                         renderer='browser'):  # Keep parameter for documentation
    """
    Create an interactive Plotly Mapbox figure for a pandapipes network.
    
    The function gathers geodata from all network components. For components that
    do not have dedicated geodata (e.g. sinks, sources, heat exchangers, valves, pumps,
    compressors), the referenced junction geodata (or midpoints between "from" and "to" nodes)
    are used.
    
    Parameters
    ----------
    net : pandapipes network
         The network object. It is expected that:
           - net.junction_geodata is a DataFrame containing the junction coordinates.
           - net.pipe_geodata is optional (if provided, each row should have a "coords" column,
             a list of coordinate pairs).
           - Other component tables (net.pipe, net.ext_grid, net.source, net.sink, net.valve,
             net.pump, net.heat_exchanger, net.press_control, net.compressor, net.flow_control,
             net.heat_consumer) are available.
    mapbox_access_token : str, optional
         A valid Mapbox access token. If None, _get_mapbox_token() is used.
    map_style : str, optional
         The Mapbox style (e.g. "streets", "light", "dark", "satellite"). Default is "streets".
    zoom : int, optional
         The initial zoom level. Default is 10.
         
    Additional parameters for pipe temperature-based coloring:
    pipe_temperature_coloring : bool, optional
         If True, pipes are colored based on the temperature value taken from net.res_pipe.
         (Default: False)
    pipe_temperature_field : str, optional
         The field name in net.res_pipe to use for temperature (default: "t_to_k").
    pipe_color : list or str, optional
         If provided (and if pipe_temperature_coloring is False), these colors are used for pipes.
         If a list, it is assumed the ordering corresponds to the pipe indices.
    show_colorbar : bool, optional
         If True (and pipe_temperature_coloring is True), a colorbar is added to the figure.
    pipe_temperature_colorscale : list, optional
         The colorscale to use for the colorbar (default: [[0, "blue"], [1, "red"]]).
    renderer : str, optional
         The Plotly renderer to use for display (default: 'browser'). Options include
         'browser', 'notebook', 'json', etc.
    
    Returns
    -------
    fig : plotly.graph_objects.Figure
         The interactive Plotly figure with the network overlaid on a Mapbox basemap.
    """
    token = mapbox_access_token if mapbox_access_token is not None else _get_mapbox_token()
    if not token:
        print("Warning: No Mapbox access token provided. The map may not render correctly.")

    traces = []
    center = {"lat": 0, "lon": 0}

    # --- Junctions ---
    if hasattr(net, "junction_geodata") and net.junction_geodata is not None and not net.junction_geodata.empty:
        df = net.junction_geodata
        if "lat" in df.columns and "lon" in df.columns:
            lat = df["lat"]
            lon = df["lon"]
        elif "y" in df.columns and "x" in df.columns:
            lat = df["y"]
            lon = df["x"]
        else:
            raise ValueError("Junction geodata must have columns 'lat'/'lon' or 'x'/'y'.")
        center = {"lat": float(lat.mean()), "lon": float(lon.mean())}
        traces.append(go.Scattermapbox(
            lon=lon,
            lat=lat,
            mode="markers",
            marker=dict(size=8, color="red"),
            text=["Junction {}".format(i) for i in df.index],
            name="Junctions"
        ))
    else:
        print("Warning: No junction geodata available.")

    # --- Pipes ---
    # If temperature-based coloring is enabled and pipe results are available,
    # compute the min/max and a colormap.
    if pipe_temperature_coloring:
        if hasattr(net, "res_pipe") and net.res_pipe is not None and not net.res_pipe.empty:
            pipe_temps = net.res_pipe[pipe_temperature_field].values
            min_temp = float(pipe_temps.min())
            max_temp = float(pipe_temps.max())
            from pandapipes.plotting import colormaps
            cmap, norm = colormaps.cmap_continuous([(min_temp, "blue"), (max_temp, "red")])
        else:
            print("Warning: pipe_temperature_coloring is enabled but no pipe results are available. Defaulting to gray.")
            cmap = None
            norm = None
            min_temp = None
            max_temp = None
    else:
        cmap = None
        norm = None
        min_temp = None
        max_temp = None

    # Prefer pipe_geodata if available
    if hasattr(net, "pipe_geodata") and net.pipe_geodata is not None and not net.pipe_geodata.empty:
        for idx, row in net.pipe_geodata.iterrows():
            coords = row.get("coords")
            if coords is None:
                continue
            if isinstance(coords, list) and len(coords) > 0:
                # Assume each coordinate pair is (x, y) which represents (lon, lat)
                lon_list, lat_list = zip(*coords)
                # Determine color for this pipe trace
                if pipe_temperature_coloring:
                    if cmap is not None:
                        try:
                            temp = net.res_pipe.loc[idx, pipe_temperature_field]
                        except Exception:
                            temp = min_temp
                        color = mcolors.rgb2hex(cmap(norm(temp)))
                    else:
                        color = "gray"
                elif pipe_color is not None:
                    if isinstance(pipe_color, (list, tuple)):
                        color = pipe_color[idx] if idx < len(pipe_color) else "gray"
                    else:
                        color = pipe_color
                else:
                    color = "gray"
                traces.append(go.Scattermapbox(
                    lon=lon_list,
                    lat=lat_list,
                    mode="lines",
                    line=dict(width=3, color=color),
                    name="Pipe {}".format(idx)
                ))
    elif hasattr(net, "pipe") and net.pipe is not None and not net.pipe.empty:
        # Fallback: use junction_geodata via from/to indices
        for idx, row in net.pipe.iterrows():
            from_idx = row["from_junction"]
            to_idx = row["to_junction"]
            try:
                if "lat" in net.junction_geodata.columns and "lon" in net.junction_geodata.columns:
                    lat0 = net.junction_geodata.loc[from_idx, "lat"]
                    lon0 = net.junction_geodata.loc[from_idx, "lon"]
                    lat1 = net.junction_geodata.loc[to_idx, "lat"]
                    lon1 = net.junction_geodata.loc[to_idx, "lon"]
                elif "y" in net.junction_geodata.columns and "x" in net.junction_geodata.columns:
                    lat0 = net.junction_geodata.loc[from_idx, "y"]
                    lon0 = net.junction_geodata.loc[from_idx, "x"]
                    lat1 = net.junction_geodata.loc[to_idx, "y"]
                    lon1 = net.junction_geodata.loc[to_idx, "x"]
                else:
                    continue
            except Exception:
                continue
            if pipe_temperature_coloring:
                if cmap is not None:
                    try:
                        temp = net.res_pipe.loc[idx, pipe_temperature_field]
                    except Exception:
                        temp = min_temp
                    color = mcolors.rgb2hex(cmap(norm(temp)))
                else:
                    color = "gray"
            elif pipe_color is not None:
                if isinstance(pipe_color, (list, tuple)):
                    color = pipe_color[idx] if idx < len(pipe_color) else "gray"
                else:
                    color = pipe_color
            else:
                color = "gray"
            traces.append(go.Scattermapbox(
                lon=[lon0, lon1],
                lat=[lat0, lat1],
                mode="lines",
                line=dict(width=3, color=color),
                name="Pipe {}".format(idx)
            ))

    # --- External Grids ---
    if hasattr(net, "ext_grid") and net.ext_grid is not None and not net.ext_grid.empty:
        eg_indices = net.ext_grid.junction.values
        try:
            coords = net.junction_geodata.loc[eg_indices]
        except Exception:
            coords = None
        if coords is not None and not coords.empty:
            if "lat" in coords.columns and "lon" in coords.columns:
                lat = coords["lat"]
                lon = coords["lon"]
            elif "y" in coords.columns and "x" in coords.columns:
                lat = coords["y"]
                lon = coords["x"]
            else:
                lat, lon = [], []
            traces.append(go.Scattermapbox(
                lon=lon,
                lat=lat,
                mode="markers",
                marker=dict(size=10, color="orange"),
                text=["Ext Grid {}".format(i) for i in coords.index],
                name="External Grids"
            ))

    # --- Sources ---
    if hasattr(net, "source") and net.source is not None and not net.source.empty:
        src_indices = net.source.junction.values
        try:
            coords = net.junction_geodata.loc[src_indices]
        except Exception:
            coords = None
        if coords is not None and not coords.empty:
            if "lat" in coords.columns and "lon" in coords.columns:
                lat = coords["lat"]
                lon = coords["lon"]
            elif "y" in coords.columns and "x" in coords.columns:
                lat = coords["y"]
                lon = coords["x"]
            else:
                lat, lon = [], []
            traces.append(go.Scattermapbox(
                lon=lon,
                lat=lat,
                mode="markers",
                marker=dict(size=10, color="blue"),
                text=["Source {}".format(i) for i in coords.index],
                name="Sources"
            ))

    # --- Sinks ---
    if hasattr(net, "sink") and net.sink is not None and not net.sink.empty:
        sink_indices = net.sink.junction.values
        try:
            coords = net.junction_geodata.loc[sink_indices]
        except Exception:
            coords = None
        if coords is not None and not coords.empty:
            if "lat" in coords.columns and "lon" in coords.columns:
                lat = coords["lat"]
                lon = coords["lon"]
            elif "y" in coords.columns and "x" in coords.columns:
                lat = coords["y"]
                lon = coords["x"]
            else:
                lat, lon = [], []
            traces.append(go.Scattermapbox(
                lon=lon,
                lat=lat,
                mode="markers",
                marker=dict(size=10, color="green"),
                text=["Sink {}".format(i) for i in coords.index],
                name="Sinks"
            ))

    # --- Valves ---
    if hasattr(net, "valve") and net.valve is not None and not net.valve.empty:
        if "junction" in net.valve.columns:
            valve_indices = net.valve.junction.values
            try:
                coords = net.junction_geodata.loc[valve_indices]
            except Exception:
                coords = None
            if coords is not None and not coords.empty:
                if "lat" in coords.columns and "lon" in coords.columns:
                    lat = coords["lat"]
                    lon = coords["lon"]
                elif "y" in coords.columns and "x" in coords.columns:
                    lat = coords["y"]
                    lon = coords["x"]
                else:
                    lat, lon = [], []
                traces.append(go.Scattermapbox(
                    lon=lon,
                    lat=lat,
                    mode="markers",
                    marker=dict(size=8, color="cyan"),
                    text=["Valve {}".format(i) for i in coords.index],
                    name="Valves"
                ))
        elif "from_junction" in net.valve.columns and "to_junction" in net.valve.columns:
            for idx, row in net.valve.iterrows():
                from_idx = row["from_junction"]
                to_idx = row["to_junction"]
                try:
                    if "lat" in net.junction_geodata.columns and "lon" in net.junction_geodata.columns:
                        lat0 = net.junction_geodata.loc[from_idx, "lat"]
                        lon0 = net.junction_geodata.loc[from_idx, "lon"]
                        lat1 = net.junction_geodata.loc[to_idx, "lat"]
                        lon1 = net.junction_geodata.loc[to_idx, "lon"]
                    elif "y" in net.junction_geodata.columns and "x" in net.junction_geodata.columns:
                        lat0 = net.junction_geodata.loc[from_idx, "y"]
                        lon0 = net.junction_geodata.loc[from_idx, "x"]
                        lat1 = net.junction_geodata.loc[to_idx, "y"]
                        lon1 = net.junction_geodata.loc[to_idx, "x"]
                    else:
                        continue
                except Exception:
                    continue
                mid_lat = (lat0 + lat1) / 2.0
                mid_lon = (lon0 + lon1) / 2.0
                traces.append(go.Scattermapbox(
                    lon=[mid_lon],
                    lat=[mid_lat],
                    mode="markers",
                    marker=dict(size=8, color="cyan"),
                    text="Valve {}".format(idx),
                    name="Valves"
                ))

    # --- Pumps ---
    if hasattr(net, "pump") and net.pump is not None and not net.pump.empty:
        for idx, row in net.pump.iterrows():
            from_idx = row["from_junction"]
            to_idx = row["to_junction"]
            try:
                if "lat" in net.junction_geodata.columns and "lon" in net.junction_geodata.columns:
                    lat0 = net.junction_geodata.loc[from_idx, "lat"]
                    lon0 = net.junction_geodata.loc[from_idx, "lon"]
                    lat1 = net.junction_geodata.loc[to_idx, "lat"]
                    lon1 = net.junction_geodata.loc[to_idx, "lon"]
                elif "y" in net.junction_geodata.columns and "x" in net.junction_geodata.columns:
                    lat0 = net.junction_geodata.loc[from_idx, "y"]
                    lon0 = net.junction_geodata.loc[from_idx, "x"]
                    lat1 = net.junction_geodata.loc[to_idx, "y"]
                    lon1 = net.junction_geodata.loc[to_idx, "x"]
                else:
                    continue
            except Exception:
                continue
            traces.append(go.Scattermapbox(
                lon=[lon0, lon1],
                lat=[lat0, lat1],
                mode="lines",
                line=dict(width=3, color="purple", dash="dot"),
                name="Pump {}".format(idx)
            ))

    # --- Other line-type components ---
    def add_line_component(component_name, color):
        if hasattr(net, component_name) and getattr(net, component_name) is not None and not getattr(net, component_name).empty:
            comp = getattr(net, component_name)
            for idx, row in comp.iterrows():
                if "from_junction" in row and "to_junction" in row:
                    from_idx = row["from_junction"]
                    to_idx = row["to_junction"]
                    try:
                        if "lat" in net.junction_geodata.columns and "lon" in net.junction_geodata.columns:
                            lat0 = net.junction_geodata.loc[from_idx, "lat"]
                            lon0 = net.junction_geodata.loc[from_idx, "lon"]
                            lat1 = net.junction_geodata.loc[to_idx, "lat"]
                            lon1 = net.junction_geodata.loc[to_idx, "lon"]
                        elif "y" in net.junction_geodata.columns and "x" in net.junction_geodata.columns:
                            lat0 = net.junction_geodata.loc[from_idx, "y"]
                            lon0 = net.junction_geodata.loc[from_idx, "x"]
                            lat1 = net.junction_geodata.loc[to_idx, "y"]
                            lon1 = net.junction_geodata.loc[to_idx, "x"]
                        else:
                            continue
                    except Exception:
                        continue
                    traces.append(go.Scattermapbox(
                        lon=[lon0, lon1],
                        lat=[lat0, lat1],
                        mode="lines",
                        line=dict(width=3, color=color),
                        name="{} {}".format(component_name.capitalize(), idx)
                    ))
    add_line_component("heat_exchanger", color="brown")
    add_line_component("press_control", color="pink")
    add_line_component("compressor", color="black")
    add_line_component("flow_control", color="orange")
    add_line_component("heat_consumer", color="yellow")

    if pipe_temperature_coloring and show_colorbar and min_temp is not None and max_temp is not None:
        traces.append(go.Scattermapbox(
            lon=[center["lon"]],
            lat=[center["lat"]],
            mode="markers",
            marker=dict(
                size=0,
                color=[min_temp],  # Dummy value
                colorscale=pipe_temperature_colorscale,
                cmin=min_temp,
                cmax=max_temp,
                colorbar=dict(
                    title="Pipe Temperature (K)"
                    
                ),
                showscale=True
            ),
            showlegend=False,
            hoverinfo="none"
        ))

    layout = go.Layout(
        mapbox=dict(
            accesstoken=token,
            style=map_style,
            center=center,
            zoom=zoom
        ),
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        showlegend=True
    )
    
    fig = go.Figure(data=traces, layout=layout)
    return fig