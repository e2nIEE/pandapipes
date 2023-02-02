import seaborn
import pandapower.plotting as ppplot
from pandapower.plotting import simple_plot
try:
    from workshop.pp.net import power_net_example
except ModuleNotFoundError:
    from pp.net import power_net_example
import numpy as np
import matplotlib.pyplot as plt


def create_power_net_collections(net):
    colors = seaborn.color_palette('colorblind', n_colors=15)
    bc = ppplot.create_bus_collection(net, buses=net.bus.index, color='black', size=5, zorder=2)
    lc = ppplot.create_line_collection(net, lines=net.line.index, color=colors[1], zorder=1,
                                       use_bus_geodata=True)
    ec = ppplot.create_ext_grid_collection(net, ext_grids=net.ext_grid.index, size=20, zorder=5,
                                           patch_edgecolor=colors[2],
                                           orientation=-np.pi / 2)
    tc = ppplot.create_trafo_collection(net, trafos=net.trafo.index, size=10, color=colors[3],
                                        zorder=1)
    sc = ppplot.create_bus_bus_switch_collection(net, size=5, zorder=1)
    ld = ppplot.create_load_collection(net, size=20, zorder=1, patch_edgecolor=colors[4],
                                       color=colors[4])
    sg = ppplot.create_sgen_collection(net, size=20, zorder=1, patch_edgecolor=colors[5],
                                       color=colors[5], orientation=np.pi / 2)
    vba = ppplot.create_bus_collection(net, buses=net.bus.index[net.res_bus.vm_pu > 1.04],
                                       color='red', size=5, zorder=11)
    vbb = ppplot.create_bus_collection(net, buses=net.bus.index[net.res_bus.vm_pu < 0.96],
                                       color='blue', size=5, zorder=11)
    vt = ppplot.create_trafo_collection(net,
                                        trafos=net.trafo.index[net.res_trafo.loading_percent > 90],
                                        size=10, color='red', zorder=11)
    vl = ppplot.create_line_collection(net, lines=net.line.index[net.res_line.loading_percent > 90],
                                       color='red', zorder=11, use_bus_geodata=True)

    coll = [bc, lc, ec, tc, sc, ld, sg, vba, vbb, vt, vl]
    return coll


def plot_lineloading(net, show_loadings=False):
    net = net.deepcopy()

    cmap_list = [(20, "green"), (50, "yellow"), (150, "red")]
    cmap, norm = ppplot.cmap_continuous(cmap_list)
    lc = ppplot.create_line_collection(net, net.line.index, zorder=1, cmap=cmap, norm=norm,
                                       linewidths=2)

    if show_loadings:
        loading_lst = net.res_line.loading_percent.tolist()  # list of all junction indices
        coords = zip(0.5 * net.bus_geodata.x.loc[net.line.from_bus].values
                     + 0.5 * net.bus_geodata.x.loc[net.line.to_bus].values,
                     0.5 * net.bus_geodata.y.loc[net.line.from_bus].values
                     + 0.5 * net.bus_geodata.y.loc[net.line.to_bus].values)
        # tuples of all junction coords

        loadingcol = ppplot.create_annotation_collection(size=0.15, texts=np.char.mod('%d',
                                                                                      loading_lst),
                                                         coords=coords,
                                                         zorder=150, color='k')
    else:
        loadingcol = None

    ppplot.draw_collections([lc, loadingcol], figsize=(8, 6))


if __name__ == '__main__':
    net = power_net_example()

    simple_plot(net, bus_size=0.2, ext_grid_size=0.2, plot_loads=True, plot_sgens=True)

    collections = create_power_net_collections(net)

    ppplot.draw_collections(collections)

    plt.show()
