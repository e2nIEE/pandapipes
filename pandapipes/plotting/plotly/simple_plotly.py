# Copyright (c) 2022 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import pandas as pd

from pandapipes.plotting.plotly.traces import create_junction_trace, create_pipe_trace, \
    create_valve_trace, create_compressor_trace, create_press_control_trace
from pandapower.plotting.plotly.simple_plotly import _simple_plotly_generic, draw_traces

try:
    from pandaplan.core import pplog as logging
except ImportError:
    import logging
logger = logging.getLogger(__name__)


def get_hoverinfo(net, element, precision=3, sub_index=None):
    hover_index = net[element].index
    if element == "junction":
        sink_str, source_str = [], []
        if hasattr(net, "sink"):
            for ln in [net.sink.loc[net.sink.junction == b, "mdot_kg_per_s"].sum() for b in
                       net.junction.index]:
                sink_str.append("Sink: {:.3f} kg/s<br />".format(ln) if ln != 0. else "")
        if hasattr(net, "source"):
            for s in [net.source.loc[net.source.junction == b, "mdot_kg_per_s"].sum() for b in
                      net.junction.index]:
                source_str.append("Source: {:.3f} kg/s<br />".format(s) if s != 0. else "")
        hoverinfo = (
                    "Index: " + net.junction.index.astype(str) + "<br />"
                    + "Name: " + net.junction['name'].astype(str) + "<br />" 
                    + "Height: " + net.junction['height_m'].astype(str) + " m <br />" 
                    + "Pressure: " + net.junction['pn_bar'].round(precision).astype(str) + " bar <br />"
                    + "Temp.: " + net.junction['tfluid_k'].round(precision).astype(str) + " K <br />"
                    # + sink_str + source_str # TODO: fix this (correct indexing)
                    ).tolist()
    elif element == "pipe":
        hoverinfo = (
                    "Index: " + net.pipe.index.astype(str) + "<br />"
                    + "Name: " + net.pipe['name'].astype(str) + "<br />"
                    + "Length: " + net.pipe['length_km'].round(precision).astype(str) + " km <br />"
                    + "Diameter: " + ((net.pipe['diameter_m']).round(precision)*1e3).astype(str) + " mm"
                    + "<br />" 
                    + "k: " + (net.pipe['k_mm']).round(precision).astype(str) + " mm <br />"
                    ).tolist()
    elif element == "pump":
        hoverinfo = (
                    "Index: " + net.pump.index.astype(str) + "<br />"
                    + "Name: " + net.pump['name'].astype(str) + "<br />"
                    + "Std. Type: " + net.pump['std_type'] + "<br />"
                    ).tolist()
    elif element == "compressor":
        hoverinfo = (
                    "Index: " + net.compressor.index.astype(str) + "<br />" 
                    + "Name: " + net.compressor['name'].astype(str) + "<br />" 
                    + "Pressure ratio: " + net.compressor['pressure_ratio'] + "<br />"
                    ).tolist()
    elif element == "pressure_control":
        hoverinfo = (
                    "Index: " + net.pressure_control.index.astype(str) + "<br />"
                    + "Name: " + net.pressure_control['name'].astype(str) + "<br />" 
                    + "controlled junction:" + net.pressure_control['controlled_junction'] + "<br />" 
                    + "controlled pressure:" + net.pressure_control['controlled_p_bar'] + " bar<br />"
                    ).tolist()
    elif element == "ext_grid":
        hoverinfo = (
                    "Index: " + net.ext_grid.index.astype(str) + "<br />"
                    + "Name: " + net.ext_grid['name'].astype(str) + "<br />"
                    + "Pressure: " + net.ext_grid['p_bar'].round(precision).astype(str) + " bar <br />"
                    + "Temp.: " + net.ext_grid['t_k'].round(precision).astype(str) + " K <br />"
                    ).tolist()

        hover_index = net.ext_grid.junction.tolist()
    elif element == "valve":
        hoverinfo = (
                    "Index: " + net.valve.index.astype(str) + "<br />"
                    + "Name: " + net.valve['name'].astype(str) + "<br />"
                    + "Open: " + net.valve['opened'].astype(str) + "<br />"
                    + "Diameter: " + ((net.valve['diameter_m']).round(precision)*1e3).astype(str)
                    + " mm <br />"
                    ).tolist()
    else:
        return None
    hoverinfo = pd.Series(index=hover_index, data=hoverinfo)
    if sub_index is not None:
        hoverinfo = hoverinfo.loc[list(sub_index)]
    return hoverinfo


def simple_plotly(net, use_pipe_geodata=None, on_map=False,
                  projection=None, map_style='basic', figsize=1, aspectratio='auto', pipe_width=1,
                  junction_size=10, ext_grid_size=20.0, valve_size=3.0, compressor_size=3,
                  junction_color="blue", pipe_color='grey', pressure_reg_color='green',
                  ext_grid_color="yellow", valve_color="black", compressor_color="cyan",
                  filename='temp-plot.html',
                  auto_open=True, showlegend=True, additional_traces=None):
    respect_valves = False # TODO, not implemented yet
    node_element = "junction"
    branch_element = "pipe"
    trans_element = "pump"
    separator_element = "valve"

    traces, settings = _simple_plotly_generic(net=net,
                                              respect_separators=respect_valves,
                                              use_branch_geodata=use_pipe_geodata,
                                              on_map=on_map,
                                              projection=projection,
                                              map_style=map_style,
                                              figsize=figsize,
                                              aspectratio=aspectratio,
                                              branch_width=pipe_width,
                                              node_size=junction_size,
                                              ext_grid_size=ext_grid_size,
                                              node_color=junction_color,
                                              branch_color=pipe_color,
                                              trafo_color=pressure_reg_color,
                                              trafo3w_color=pressure_reg_color,
                                              ext_grid_color=ext_grid_color,
                                              node_element=node_element,
                                              branch_element=branch_element,
                                              trans_element=trans_element,
                                              trans3w_element=None,
                                              separator_element=separator_element,
                                              branch_trace_func=create_pipe_trace,
                                              node_trace_func=create_junction_trace,
                                              hoverinfo_func=get_hoverinfo,
                                              filename=filename,
                                              auto_open=auto_open,
                                              showlegend=showlegend)
    if "valve" in net.keys() and len(net.valve) > 0:
        if "source_name" in net.valve.columns:
            info = net.valve.apply(lambda x: f"{x['name']} <br />{x['source_name']}", axis=1)
        else:
            info = None
        traces.extend(create_valve_trace(net, valves=net.valve.loc[net.valve.opened].index,
                                         size=valve_size, color=valve_color,
                                         infofunc=info,
                                         trace_name="open valves"))
        traces.extend(create_valve_trace(net, valves=net.valve.loc[~net.valve.opened].index,
                                         size=valve_size, color=valve_color, dash="dot",
                                         infofunc=info,
                                         trace_name="closed valves"))

    if "compressor" in net.keys() and len(net.compressor) > 0:
        traces.extend(create_compressor_trace(net, size=compressor_size, color=compressor_color))
    if "press_control" in net.keys() and len(net.press_control) > 0:
            traces.extend(
                create_press_control_trace(net, size=compressor_size, color="cyan",
                                           trace_name='compressors SPO',
                                           legendgroup='compressors SPO'))
            traces.extend(create_junction_trace(net, net.press_control.controlled_junction.values,
                                                color="red", size=10,
                                                trace_name="controlled junction",
                                                legendgroup='compressors SPO'))
    if additional_traces:
        if isinstance(additional_traces, dict):
            traces.append(additional_traces)
        else:
            traces.extend(additional_traces)

    return draw_traces(traces, **settings)


if __name__ == '__main__':
    from pandapipes.networks import gas_versatility
    net = gas_versatility()
    simple_plotly(net)
