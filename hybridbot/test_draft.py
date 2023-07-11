##%

from pandapipes.plotting.plotly.simple_plotly import simple_plotly
import pandapipes.plotting as plot
from pandapipes.test.api.test_special_networks import simple_fluid, same_fluid_twice_defined, \
    test_two_fluids_grid_simple, \
    test_three_fluids_grid_simple, \
    test_multiple_fluids_grid_line_ascending, \
    test_multiple_fluids_grid, \
    test_multiple_fluids_grid_meshed_valve, \
    test_multiple_fluids_grid_source, \
    test_multiple_fluids_grid_feed_back, \
    test_multiple_fluids_feeder, \
    test_multiple_fluids_grid_valve, \
    test_two_node_net_with_two_different_fluids, \
    test_four_fluids_grid_simple, \
    test_two_fluids_two_pipes_grid_simple

#from pandapipes.hybridbot.plot_tool import annotated_plot

import pandapipes

# Mass flow conservation tests
# assert np.isclose(net.res_ext_grid.values.sum() + net.res_sink.values.sum() - net.res_source.values.sum(), 0)

#test_two_fluids_grid_simple_gases(use_numba=True)
#test_three_fluids_grid_simple_gases(use_numba=True)

test_two_fluids_grid_simple(use_numba=True)
test_three_fluids_grid_simple(use_numba=True)
test_four_fluids_grid_simple(use_numba=True)
test_two_fluids_two_pipes_grid_simple(use_numba=True)
test_multiple_fluids_grid_line_ascending(use_numba=True)
test_multiple_fluids_grid(use_numba=True)
test_multiple_fluids_grid_meshed_valve(use_numba=True)
test_multiple_fluids_grid_source(use_numba=True)
test_multiple_fluids_grid_feed_back(use_numba=True)
test_multiple_fluids_feeder(use_numba=True)
test_multiple_fluids_grid_valve(use_numba=True)
test_two_node_net_with_two_different_fluids(use_numba=True)



##%
if 0:
    from pandapipes.test.api.test_special_networks import test_multiple_fluids_sink_source

    test_multiple_fluids_sink_source(use_numba=True)

##%
if 0:
    # from pandapipes.test.api.test_special_networks import test_schutterwald_hydrogen
    from pandapipes import networks as nets_pps

    def test_schutterwald_hydrogen():
        net = nets_pps.schutterwald()
        pandapipes.create_sources(net, [5, 168, 193], 6.6e-3, 'hydrogen')
        pandapipes.pipeflow(net, iter=100)

    test_schutterwald_hydrogen()

##%
def test_t_cross_mixture():
    net = pandapipes.create_empty_network()
    j1 = pandapipes.create_junction(net, 1, 273)
    j2 = pandapipes.create_junction(net, 1, 273)
    j3 = pandapipes.create_junction(net, 1, 273)
    j4 = pandapipes.create_junction(net, 1, 273)
    pandapipes.create_ext_grid(net, j1, 1, 273, 'hgas')
    pandapipes.create_pipe_from_parameters(net, j1, j2, 1, 0.1, 0.1)
    pandapipes.create_pipe_from_parameters(net, j2, j3, 1, 0.1, 0.1)
    pandapipes.create_pipe_from_parameters(net, j2, j4, 1, 0.1, 0.1)
    pandapipes.create_sink(net, j3, 0.01)
    pandapipes.create_sink(net, j4, 0.02)
    pandapipes.create_source(net, j3, 0.02, 'lgas')
    pandapipes.create_source(net, j4, 0.03, 'hydrogen')
    annotated_plot(net)
    pandapipes.pipeflow(net, iter=100, use_numba=False)
test_t_cross_mixture()


##%
#
def t_cross_mixture():
    a = 1
    net = pandapipes.create_empty_network()
    j1 = pandapipes.create_junction(net, 1, 273)
    j2 = pandapipes.create_junction(net, 1, 273)
    j3 = pandapipes.create_junction(net, 1, 273)
    j4 = pandapipes.create_junction(net, 1, 273)
    j5 = pandapipes.create_junction(net, 1, 273)
    j6 = pandapipes.create_junction(net, 1, 273)
    j7 = pandapipes.create_junction(net, 1, 273)
    j8 = pandapipes.create_junction(net, 1, 273)
    pandapipes.create_ext_grid(net, j1, 1, 273, 'hgas')
    pandapipes.create_pipe_from_parameters(net, j1, j2, 1, 0.1, 0.1)
    pandapipes.create_pipe_from_parameters(net, j2, j3, 1, 0.1, 0.1)
    pandapipes.create_pipe_from_parameters(net, j2, j4, 1, 0.1, 0.1)
    pandapipes.create_pipe_from_parameters(net, j5, j3, 1, 0.1, 0.1)
    pandapipes.create_pipe_from_parameters(net, j7, j4, 1, 0.1, 0.1)
    pandapipes.create_pipe_from_parameters(net, j8, j4, 1, 0.1, 0.1)
    pandapipes.create_pipe_from_parameters(net, j6, j3, 1, 0.1, 0.1)

    pandapipes.create_sink(net, j5, 0.01)
    pandapipes.create_sink(net, j7, 0.02)
    pandapipes.create_source(net, j6, 0.02, 'lgas')
    pandapipes.create_source(net, j8, 0.03, 'hydrogen')
    pandapipes.pipeflow(net, iter=100, use_numba=False)
    #_ = simple_plotly(net)
    #plot.create_junction_collection(net)
    #collection = plot.create_junction_collection(net, junctions=[3], patch_type="circle", size=0.1,
    #                                             color="orange", zorder=200)
    #simple_collections = plot.create_simple_collections(net, as_dict=False)
    #coll_junction = create_node_collection
    # plot.simple_plot(net, plot_sinks=True, plot_sources=True)


t_cross_mixture()


import matplotlib.pyplot as plt
from pandapower.plotting.collections import create_annotation_collection, create_line_collection
import numpy as np
import pandapipes.plotting as plot

##
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