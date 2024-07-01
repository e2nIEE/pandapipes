# Copyright (c) 2022 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

from pandapower.plotting.plotly.traces import _create_node_trace, _create_branch_trace, \
    draw_traces


def create_junction_trace(net, junctions=None, size=5, patch_type="circle", color="blue",
                          infofunc=None, trace_name='junctions', legendgroup=None, cmap=None,
                          cmap_vals=None, cbar_title=None, cmin=None, cmax=None, cpos=1.0,
                          colormap_column="p_bar"):

    node_element = "junction"
    branch_element = "pipe"

    return _create_node_trace(net=net, nodes=junctions, size=size , patch_type=patch_type,
                              color=color, infofunc=infofunc, trace_name=trace_name,
                              legendgroup=legendgroup, cmap=cmap, cmap_vals=cmap_vals,
                              cbar_title=cbar_title, cmin=cmin, cmax=cmax, cpos=cpos,
                              colormap_column=colormap_column,
                              node_element=node_element, branch_element=branch_element)


def create_pipe_trace(net, pipes=None, use_pipe_geodata=True, respect_valves=False, width=1.0,
                      color='grey', infofunc=None, trace_name='pipes', legendgroup='pipes',
                      cmap=None, cbar_title=None, show_colorbar=True, cmap_vals=None, cmin=None,
                      cmax=None, cpos=1.1):
    branch_element = "pipe"
    node_element = "junction"
    separator_element = "valve"

    return _create_branch_trace(net=net, branches=pipes, use_branch_geodata=use_pipe_geodata,
                                respect_separators=respect_valves, width=width, color=color,
                                infofunc=infofunc, trace_name=trace_name, legendgroup=legendgroup,
                                cmap=cmap, cbar_title=cbar_title, show_colorbar=show_colorbar,
                                cmap_vals=cmap_vals, cmin=cmin, cmax=cmax, cpos=cpos,
                                branch_element=branch_element, separator_element=separator_element,
                                node_element=node_element, cmap_vals_category="vmean_m_s")
