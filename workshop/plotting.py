import seaborn
import numpy as np
import pandapipes as ps

import matplotlib.pyplot as plt
import pandapipes.plotting as plot
from pandapipes.plotting import simple_plot
from pandapower.plotting import cmap_continuous, create_annotation_collection

from workshop.pipe_net import pipe_net_example, gas_net_example, gas_net_example2, heat_net_example


def create_simple_plot():
    net = pipe_net_example()
    simple_plot(net)


def create_pipe_net_collections(net):
    #net = pipe_net_example()
    colors = seaborn.color_palette('colorblind')
    jc = plot.create_junction_collection(net, color=colors[0])
    pc = plot.create_pipe_collection(net, color=colors[1])
    collections = [jc, pc]
    plot.draw_collections(collections)
    plt.show()
    return collections


def create_annotations():
    net = pipe_net_example()
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


def create_gas_net_collections():
    net = gas_net_example()
    colors = seaborn.color_palette('colorblind')
    collections = create_pipe_net_collections(net)
    sc = plot.create_sink_collection(net, color=colors[3], size=20)
    soc = plot.create_source_collection(net, color=colors[4], size=20)
    ec = plot.create_ext_grid_collection(net, color=colors[5], size=20)
    collections += [sc, soc, ec]
    plot.draw_collections(collections)
    plt.show()
    return collections


def create_gas_net_collections2():
    net = gas_net_example2()
    collections = create_gas_net_collections()
    pcc = plot.create_pressure_control_collection(net, size=20, color='grey')
    vc = plot.create_valve_collection(net, color='grey', size=5)
    pc = plot.create_pump_collection(net, color='grey', size=20)
    collections += [pcc, vc, pc]
    plot.draw_collections(collections)
    plt.show()


def create_heat_net_collections():
    net = heat_net_example()
    collections = create_pipe_net_collections(net)
    fc = plot.create_flow_control_collection(net, size=20, color='grey')
    cpp = plot.create_pump_collection(net, table_name='circ_pump_pressure', color='grey', size=20, fj_col='return_junction',
                                      tj_col='flow_junction')
    cpm = plot.create_pump_collection(net, table_name='circ_pump_mass', color='grey', size=20, fj_col='return_junction',
                                      tj_col='flow_junction')
    hc = plot.create_heat_exchanger_collection(net, color='grey', size=20)
    collections += [fc, cpp, cpm, hc]
    plot.draw_collections(collections)
    plt.show()


def plot_gas_results(net, junction_size=15):
    sic = plot.create_sink_collection(net, patch_edgecolor='grey', line_color='grey')
    if hasattr(net, 'source'):
        src = plot.create_source_collection(net, patch_edgecolor='grey',
                                            line_color='grey')
    else:
        src = None
    ec = plot.create_ext_grid_collection(net)
    if hasattr(net, 'valve'):
        vc = plot.create_valve_collection(net, color='grey')
    else:
        vc = None

    # color map for pressure
    max_p = net.res_junction.p_bar.max()
    cmap_list_p = [(0, "red"), (max_p / 2, "yellow"), (max_p, "green")]
    cmap_p, norm_p = cmap_continuous(cmap_list_p)

    jc = plot.create_junction_collection(net, size=junction_size, cmap=cmap_p, norm=norm_p,
                                         z=net.res_junction.p_bar,
                                         cbar_title="junction pressure [bar]")

    # color map for velocity
    cmap_list_p = [(-0.007, "green"), (0.00, "yellow"), (0.007, "red")]
    cmap_v, norm_v = cmap_continuous(cmap_list_p)
    pc = plot.create_pipe_collection(net, linewidths=1,
                                     cmap=cmap_v, norm=norm_v,
                                     z=net.res_pipe.v_mean_m_per_s.abs(),
                                     cbar_title="mean gas velocity [m/s]")

    pcc = plot.create_pressure_control_collection(net, size=junction_size, color='grey')
    plot.draw_collections([sic, src, ec, jc, pc, vc, pcc], figsize=(8, 6))


if __name__ == '__main__':
    create_gas_net_collections2()
