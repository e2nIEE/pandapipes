import seaborn
import numpy as np

import matplotlib.pyplot as plt
import pandapipes.plotting as plot
from pandapipes.plotting import simple_plot
from pandapower.plotting import cmap_continuous, create_annotation_collection

from workshop.pps.net import pipe_net_example, gas_net_example, gas_net_example2, heat_net_example


def create_simple_plot(net):
    simple_plot(net)


def create_pipe_net_collections(net):
    colors = seaborn.color_palette('colorblind')
    jc = plot.create_junction_collection(net, color=colors[0])
    pc = plot.create_pipe_collection(net, color=colors[1])
    collections = [jc, pc]
    return collections


def create_annotations(net):
    collections = create_pipe_net_collections(net)

    coords = net.junction_geodata[['x', 'y']].values
    jic = create_annotation_collection(size=20,
                                       texts=np.char.mod('%.0f', net.junction.index),
                                       coords=coords, zorder=150, color='k')

    # tuples of all junction coords
    collections += [jic]
    plot.draw_collections(collections)
    plt.show()
    return collections


def create_gas_net_collections(net):
    colors = seaborn.color_palette('colorblind')
    collections = create_pipe_net_collections(net)
    sc = plot.create_sink_collection(net, color=colors[3], size=20)
    soc = plot.create_source_collection(net, color=colors[4], size=20)
    ec = plot.create_ext_grid_collection(net, color=colors[5], size=20)
    collections += [sc, soc, ec]
    plot.draw_collections(collections)
    plt.show()
    return collections


def create_gas_net_collections2(net):
    collections = create_gas_net_collections(net)
    pcc = plot.create_pressure_control_collection(net, size=20, color='grey')
    vc = plot.create_valve_collection(net, color='grey', size=5)
    pc = plot.create_pump_collection(net, color='grey', size=20)
    collections += [pcc, vc, pc]
    plot.draw_collections(collections)
    plt.show()
    return collections


def create_heat_net_collections(net):
    collections = create_pipe_net_collections(net)
    fc = plot.create_flow_control_collection(net, size=20, color='grey')
    cpp = plot.create_pump_collection(net, table_name='circ_pump_pressure', color='grey', size=20,
                                      fj_col='return_junction',
                                      tj_col='flow_junction')
    cpm = plot.create_pump_collection(net, table_name='circ_pump_mass', color='grey', size=20, fj_col='return_junction',
                                      tj_col='flow_junction')
    hc = plot.create_heat_exchanger_collection(net, color='grey', size=20)
    collections += [fc, cpp, cpm, hc]
    plot.draw_collections(collections)
    plt.show()
    return collections


def plot_heat_results(net, junction_results='t_k'):
    collections = create_heat_net_collections(net)

    # color map for velocity
    cmap_list_v = [(0.0, "green"), (1.25, "yellow"), (2.5, "red")]
    cmap_v, norm_v = cmap_continuous(cmap_list_v)
    vc = plot.create_pipe_collection(net, linewidths=1,
                                     cmap=cmap_v, norm=norm_v,
                                     z=net.res_pipe.v_mean_m_per_s.abs(),
                                     cbar_title="mean gas velocity [m/s]")
    if junction_results == 't_k':
        # color map for temperature
        cmap_list_t = [(300, "blue"), (320, "yellow"), (340, "green")]
        cmap_t, norm_t = cmap_continuous(cmap_list_t)
        jc = plot.create_junction_collection(net, size=20, cmap=cmap_t, norm=norm_t,
                                             z=net.res_junction.t_k,
                                             cbar_title="junction temperature [k]")
    elif junction_results == 'p_bar':
        # color map for pressure
        max_p = net.res_junction.p_bar.max()
        cmap_list_p = [(4, "red"), (max_p / 2, "yellow"), (max_p, "green")]
        cmap_p, norm_p = cmap_continuous(cmap_list_p)
        jc = plot.create_junction_collection(net, size=20, cmap=cmap_p, norm=norm_p,
                                             z=net.res_junction.p_bar,
                                             cbar_title="junction pressure [bar]")
    else:
        raise (UserWarning('junction results %s not implemented!' % junction_results))

    collections += [jc, vc]
    plot.draw_collections(collections, figsize=(8, 6))
    plt.show()
    return collections


if __name__ == '__main__':
    create_simple_plot(pipe_net_example())
    create_annotations(pipe_net_example())
    create_gas_net_collections2(gas_net_example2())
    create_heat_net_collections(heat_net_example())
    plot_heat_results(heat_net_example(), 'p_bar')
