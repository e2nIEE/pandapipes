
####### This file is not supposed to be included in the final PR #######
####### This file is not supposed to be included in the final PR #######
####### This file is not supposed to be included in the final PR #######

##%

import pandapipes.plotting as plot
from pandapower.plotting.collections import create_annotation_collection, create_line_collection
import numpy as np
from pandapipes.test.api.test_special_networks import simple_fluid, same_fluid_twice_defined, \
    test_two_fluids_grid_simple, \
    test_three_fluids_grid_simple, \
    test_multiple_fluids_grid_line_ascending, \
    test_multiple_fluids_grid, \
    test_multiple_fluids_grid_mesehd_valve, \
    test_multiple_fluids_grid_source, \
    test_multiple_fluids_grid_feed_back, \
    test_multiple_fluids_feeder, \
    test_multiple_fluids_grid_valve, \
    test_two_node_net_with_two_different_fluids, \
    test_four_fluids_grid_simple, \
    test_two_fluids_two_pipes_grid_simple

#from pandapipes.tests_gas_mix.plot_tool import annotated_plot

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
test_multiple_fluids_grid_mesehd_valve(use_numba=True)
test_multiple_fluids_grid_source(use_numba=True)
test_multiple_fluids_grid_feed_back(use_numba=True)
test_multiple_fluids_feeder(use_numba=True)
test_multiple_fluids_grid_valve(use_numba=True)
test_two_node_net_with_two_different_fluids(use_numba=True)



##%
from pandapipes.test.api.test_special_networks import test_multiple_fluids_sink_source

test_multiple_fluids_sink_source(use_numba=True)

##%
# from pandapipes.test.api.test_special_networks import test_schutterwald_hydrogen
from pandapipes import networks as nets_pps

def test_schutterwald_hydrogen():
    net = nets_pps.schutterwald()
    pandapipes.create_sources(net, [5, 168, 193], 6.6e-3, 'hydrogen')
    pandapipes.pipeflow(net, iter=100)

test_schutterwald_hydrogen()

##%
# original test
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

    coords = net.junction_geodata.values
    jic = create_annotation_collection(size=350, texts=np.char.mod('%.0f', net.junction.index), coords=coords, zorder=200,
                                       color='k')
    collection =  [jic]
    plot.draw_collections(collection)
    pandapipes.pipeflow(net, iter=100, use_numba=False)

test_t_cross_mixture()

##%
# case 3.a
def test_t_cross_mixture():
    factor = 3
    net = pandapipes.create_empty_network()
    j1 = pandapipes.create_junction(net, 1, 273)
    j2 = pandapipes.create_junction(net, 1, 273)
    j3 = pandapipes.create_junction(net, 1, 273)
    j4 = pandapipes.create_junction(net, 1, 273)
    pandapipes.create_ext_grid(net, j1, 1, 273, 'hgas')
    pandapipes.create_pipe_from_parameters(net, j1, j2, 1, 0.1, 0.1)
    pandapipes.create_pipe_from_parameters(net, j2, j3, 1, 0.1, 0.1)
    pandapipes.create_pipe_from_parameters(net, j2, j4, 1, 0.1, 0.1)
    pandapipes.create_sink(net, j3, factor * 0.01)
    pandapipes.create_sink(net, j4, factor * 0.02)
    pandapipes.create_source(net, j3, 0.02, 'lgas')
    pandapipes.create_source(net, j4, 0.03, 'hydrogen')
    coords = net.junction_geodata.values
    jic = create_annotation_collection(size=350, texts=np.char.mod('%.0f', net.junction.index), coords=coords, zorder=200,
                                       color='k')
    collection =  [jic]
    plot.draw_collections(collection)
    pandapipes.pipeflow(net, iter=100, use_numba=False)
test_t_cross_mixture()

##%
# case 3.b
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

t_cross_mixture()

## converges but returns unreasonable results
# case 4.a
def test_t_cross_mixture():
    factor = 1
    net = pandapipes.create_empty_network()
    j1 = pandapipes.create_junction(net, 1, 273)
    j2 = pandapipes.create_junction(net, 1, 273)
    j3 = pandapipes.create_junction(net, 1, 273)
    j4 = pandapipes.create_junction(net, 1, 273)
    pandapipes.create_ext_grid(net, j1, 1, 273, 'lgas')
    pandapipes.create_pipe_from_parameters(net, j1, j2, 1, 0.1, 0.1)
    pandapipes.create_pipe_from_parameters(net, j2, j3, 1, 0.1, 0.1)
    pandapipes.create_pipe_from_parameters(net, j2, j4, 1, 0.1, 0.1)
    pandapipes.create_sink(net, j3, factor * 0.01)
    pandapipes.create_sink(net, j4, factor * 0.02)
    pandapipes.create_source(net, j3, 0.02, 'hgas')
    pandapipes.create_source(net, j4, 0.03, 'hydrogen')
    pandapipes.pipeflow(net, iter=100, use_numba=False)
    return net

test_t_cross_mixture()
