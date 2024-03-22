# Copyright (c) 2020-2024 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.


import pandapipes.topology as top
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations


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
                          junction_colors="tab:blue", junction_size=3, pipes=None, **kwargs):
    """Plot the pressure profile depending on the distance from the x0_junction (slack).

    Parameters
    ----------
    net : pp.PandapowerNet
        net including pipeflow results
    ax : matplotlib.axes, optional
        axis to plot to, by default None
    x0_junctions : Any[list[int], pd.Index[int]], optional
        list of junction indices which should be at position x0. If None, all in service slack junctions are considered,
        by default None
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
    junction_colors : [str, dict[int, str]], optional
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
        x0_junctions = net.ext_grid[net.ext_grid.in_service].junction.values.tolist()

    d = top.calc_distance_to_junctions(net, x0_junctions)
    pipe_table = net.pipe[net.pipe.in_service & net.pipe.index.isin(pipes)]
    for from_junction, to_junction in zip(pipe_table.from_junction, pipe_table.to_junction):
        if from_junction not in d.index:
            continue
        x = [x0 + d.at[from_junction], x0 + d.at[to_junction]]
        try:
            y = [net.res_junction.p_bar.at[from_junction], net.res_junction.p_bar.at[to_junction]]
        except Exception as e:
            raise UserWarning(e)  # todo: logger
        if "linewidth" in kwargs:
            ax.plot(x, y, color=pipe_color, **kwargs)
        else:
            ax.plot(x, y, linewidth=1, color=pipe_color, **kwargs)
        if junction_colors is not None:
            if isinstance(junction_colors, str):
                junction_colors = {b: junction_colors for b in net.junction.index}
            for junction, x, y in zip((from_junction, to_junction), x, y):
                if junction in junction_colors:
                    ax.plot(x, y, 'o', color=junction_colors[junction], ms=junction_size)
        kwargs = {k: v for k, v in kwargs.items() if not k == "label"}

    # pressure controller geodata
    if plot_pressure_controller:
        pc_table = "press_control"
        if pc_table in net.keys():
            pressure_controller = net[pc_table].query('in_service')
            for pcid, pc in pressure_controller.iterrows():
                pc_junctions = [pc[b_col] for b_col in ('from_junction', 'to_junction') if b_col in pc.index]
                if any([b not in d.index.values or b not in net.res_junction.index.values for b in pc_junctions]):
                    print('cannot add press_control %d to plot' % pcid)  # todo: logger
                    #logger.info('cannot add press_control %d to plot' % pcid)
                    continue

                for bi, bj in combinations(pc_junctions, 2):
                    pc_coords = ([x0 + d.loc[bi], x0 + d.loc[bj]],
                                 [net.res_junction.at[bi, 'p_bar'], net.res_junction.at[bj, 'p_bar']])
                    ax.plot(*pc_coords, color=pc_color,
                            **{k: v for k, v in kwargs.items() if not k == "color"})
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    return ax
