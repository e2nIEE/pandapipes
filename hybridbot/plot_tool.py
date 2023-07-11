import matplotlib.pyplot as plt
from pandapower.plotting.collections import create_annotation_collection, create_line_collection
import numpy as np
import pandapipes.plotting as plot


def annotated_plot(net):
    plt.figure()

    f = 0.5
    simple_collections = plot.create_simple_collections(net, pipe_width=f*1.8, junction_size=f*0.8, plot_sinks=True,
                                                        plot_sources=True, sink_size=f, source_size=f, ext_grid_size=f*0.6,
                                                        junction_color="r", pipe_color="k",linewidths=8.5,
                                                        linestyle_pipes="solid", as_dict=False,zorder=10)
    nc0 = plot.create_junction_collection(net, size=80,color="r", zorder=100, label='junction')
    simple_collections.append(nc0)

    coords = net.junction_geodata.values
    jic = create_annotation_collection(size=350, texts=np.char.mod('%.0f', net.junction.index), coords=coords, zorder=200,
                                       color='k')
    simple_collections.append(jic)

    axeddin = plot.draw_collections(simple_collections, figsize=(12, 12))
    #ncc = plot.draw_collections(nc0, figsize=(12, 12))

    plt.legend()
    plt.show()