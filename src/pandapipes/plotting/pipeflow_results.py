# Copyright (c) 2020-2025 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.


import pandapipes.topology as top
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from warnings import warn


def pressure_profile_to_junction_geodata(net):
    """
    Calculates pressure profile for a pandapipes network.

     INPUT:
        **net** (pandapipesNet) - Variable that contains a pandapipes network.

     OUTPUT:
        **bgd** - Returns a pandas DataFrame containing distance to the closest ext_grid as x \
                  coordinate and pressure level as y coordinate for each junction.

     EXAMPLE:
        import pandapipes.networks as nw
        import pandapipes.plotting as plotting
        import pandapipes as pp

        net = nw.schutterwald()
        pp.pipeflow(net)
        bgd = plotting.pressure_profile_to_junction_geodata(net)

    """
    if "res_junction" not in net:
        raise ValueError("no results in this pandapipes network")

    dist = top.calc_minimum_distance_to_junctions(net, net.ext_grid.junction.values)
    junctions = net.junction.index.values
    return pd.DataFrame({"x": dist.loc[junctions].values,
                         "y": net.res_junction.p_bar.loc[junctions].values},
                        index=junctions)


def plot_pressure_profile(net, ax=None, x0_junctions=None, plot_pressure_controller=True, xlabel="Distance from Slack in km",
                          ylabel="Pressure in bar", x0=0, pipe_color="tab:grey", pc_color="r",
                          junction_color="tab:blue", junction_size=3, pipes=None, **kwargs):
    """Plot the pressure profile depending on the distance from the x0_junction (slack).

    Parameters
    ----------
    net : pp.PandapowerNet
        net including pipeflow results
    ax : matplotlib.axes, optional
        axis to plot to, by default None
    x0_junctions : Any[list[int], pd.Index[int]], optional
        list of junction indices which should be at position x0. If None, all in service slack junctions are considered,
        by default None. For circ pumps, if no x0_junctions are given the flow junctions are chosen automatically.
    plot_pressure_controller : bool, optional
        Whether vertical lines should be plotted to display the pressure drop of the pressure controller,
        by default True
    xlabel : str, optional
        xlable of the figure, by default "Distance from Slack in km"
    ylabel : str, optional
        ylable of the figure, by default "Pressure in bar"
    x0 : int, optional
        x0_junction position at the xaxis, by default 0
    pipe_color : str, optional
        color used to plot the pipes, by default "tab:grey"
    pc_color : str, optional
        color used to plot the pressure controller, by default "r"
    junction_color : [str, dict[int, str]], optional
        colors used to plot the junctions. Can be passed as string (to give all junctions the same color),
        or as dict, by default "tab:blue"
    junction_size : int, optional
        size of junction representations, by default 3
    pipes : Any[list[int], pd.Index[int]], optional
        list of pipe indices which should be plottet. If None, all pipes are plotted, by default None

    Returns
    -------
    matplotlib.axes
        axis of the plot

    """
    if ax is None:
        plt.figure(facecolor="white", dpi=120)
        ax = plt.gca()
    if not net.converged:
        raise ValueError("no results in this pandapipes network")
    if pipes is None:
        pipes = net.pipe.index
    if x0_junctions is None:
        x0_junctions = set(net.ext_grid[net.ext_grid.in_service].junction.values)
        if hasattr(net, "circ_pump_pressure"):
            x0_junctions |= set(net.circ_pump_pressure[net.circ_pump_pressure.in_service].flow_junction.values)
        if hasattr(net, "circ_pump_mass"):
            x0_junctions |= set(net.circ_pump_mass[net.circ_pump_mass.in_service].flow_junction.values)
        x0_junctions = list(x0_junctions)

    unsupplied_junctions = list(top.unsupplied_junctions(net, slacks=set(x0_junctions)))
    if len(unsupplied_junctions) > 0:
        warn(UserWarning(f'There are unsupplied junctions in the net, they will not be plottet: {unsupplied_junctions}'))
    pipe_table = net.pipe[net.pipe.in_service & net.pipe.index.isin(pipes) &
                          ~net.pipe.from_junction.isin(unsupplied_junctions) &
                          ~net.pipe.to_junction.isin(unsupplied_junctions)]

    d = top.calc_distance_to_junctions(net, x0_junctions)

    x = np.array([d.loc[pipe_table.from_junction].values, d.loc[pipe_table.to_junction].values]) + x0
    y = np.array([net.res_junction.p_bar.loc[pipe_table.from_junction].values, net.res_junction.p_bar.loc[pipe_table.to_junction].values])
    linewidth = kwargs.get("linewidth", 1)
    ax.plot(x, y, linewidth=linewidth, color=pipe_color, **kwargs)

    x = d.values + x0
    y = net.res_junction.p_bar.loc[d.index]
    ax.plot(x, y, 'o', color=junction_color, ms=junction_size)

    if plot_pressure_controller and ("press_control" in net.keys()):
        pressure_controller = net.press_control.query('in_service')
        x = np.array([d.loc[pressure_controller.from_junction].values, d.loc[pressure_controller.to_junction].values]) + x0
        y = np.array([net.res_junction.p_bar.loc[pressure_controller.from_junction].values, net.res_junction.p_bar.loc[pressure_controller.to_junction].values])
        ax.plot(x, y, color=pc_color, **{k: v for k, v in kwargs.items() if not k == "color"})

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    return ax
