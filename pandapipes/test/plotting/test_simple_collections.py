# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import copy

import pandapipes.plotting as plot
from matplotlib.collections import PatchCollection, LineCollection
from pandapipes.test.test_toolbox import net_plotting, net_out_of_service_plotting


def test_simple_collections(net_plotting):
    net = copy.deepcopy(net_plotting)
    collections = plot.create_simple_collections(net, plot_sinks=True, plot_sources=True)

    assert len(collections) == len([comp for comp in net["component_list"]
                                    if not net[comp.table_name()].empty])

    assert len(collections["junction"].get_paths()) == len(net.junction)
    assert len(collections["pipe"].get_paths()) == len(net.pipe)
    assert len(collections["ext_grid"].get_paths()) == len(net.ext_grid)

    assert len(collections["source"]) == 2
    assert isinstance(collections["source"][0], PatchCollection)
    assert isinstance(collections["source"][1], LineCollection)
    assert len(collections["source"][0].get_paths()) == len(net.source)
    assert len(collections["source"][1].get_paths()) == 3 * len(net.source)

    assert len(collections["sink"]) == 2
    assert isinstance(collections["sink"][0], PatchCollection)
    assert isinstance(collections["sink"][1], LineCollection)
    assert len(collections["sink"][0].get_paths()) == len(net.sink)
    assert len(collections["sink"][1].get_paths()) == len(net.sink)

    assert len(collections["valve"]) == 2
    assert isinstance(collections["valve"][0], PatchCollection)
    assert isinstance(collections["valve"][1], LineCollection)
    assert len(collections["valve"][0].get_paths()) == 2 * len(net.valve)
    assert len(collections["valve"][1].get_paths()) == 2 * len(net.valve)

    assert len(collections["heat_exchanger"]) == 2
    assert isinstance(collections["heat_exchanger"][0], PatchCollection)
    assert isinstance(collections["heat_exchanger"][1], LineCollection)
    assert len(collections["heat_exchanger"][0].get_paths()) == 2 * len(net.heat_exchanger)
    assert len(collections["heat_exchanger"][1].get_paths()) == 2 * len(net.heat_exchanger)

    assert len(collections["pump"]) == 2
    assert isinstance(collections["pump"][0], PatchCollection)
    assert isinstance(collections["pump"][1], LineCollection)
    assert len(collections["pump"][0].get_paths()) == len(net.pump)
    assert len(collections["pump"][1].get_paths()) == 4 * len(net.pump)

    assert len(collections["circ_pump_pressure"]) == 2
    assert isinstance(collections["circ_pump_pressure"][0], PatchCollection)
    assert isinstance(collections["circ_pump_pressure"][1], LineCollection)
    assert len(collections["circ_pump_pressure"][0].get_paths()) == len(net.circ_pump_pressure)
    assert len(collections["circ_pump_pressure"][1].get_paths()) == 4 * len(net.circ_pump_pressure)

    assert len(collections["circ_pump_mass"]) == 2
    assert isinstance(collections["circ_pump_mass"][0], PatchCollection)
    assert isinstance(collections["circ_pump_mass"][1], LineCollection)
    assert len(collections["circ_pump_mass"][0].get_paths()) == len(net.circ_pump_mass)
    assert len(collections["circ_pump_mass"][1].get_paths()) == 4 * len(net.circ_pump_mass)

def test_simple_collections_out_of_service(net_out_of_service_plotting):
    net = copy.deepcopy(net_out_of_service_plotting)
    collections = plot.create_simple_collections(net, plot_sinks=True, plot_sources=True)

    assert len(collections) == len([comp for comp in net["component_list"]
                                    if not net[comp.table_name()].empty])

    assert len(collections["junction"].get_paths()) == len(net.junction[net.junction.in_service])
    assert len(collections["pipe"].get_paths()) == len(net.pipe[net.pipe.in_service])
    assert len(collections["ext_grid"].get_paths()) == len(net.ext_grid[net.ext_grid.in_service])

    assert len(collections["source"]) == 2
    assert isinstance(collections["source"][0], PatchCollection)
    assert isinstance(collections["source"][1], LineCollection)
    assert len(collections["source"][0].get_paths()) == len(net.source[net.source.in_service])
    assert len(collections["source"][1].get_paths()) == 3 * len(net.source[net.source.in_service])

    assert len(collections["sink"]) == 2
    assert isinstance(collections["sink"][0], PatchCollection)
    assert isinstance(collections["sink"][1], LineCollection)
    assert len(collections["sink"][0].get_paths()) == len(net.sink[net.sink.in_service])
    assert len(collections["sink"][1].get_paths()) == len(net.sink[net.sink.in_service])

    assert len(collections["valve"]) == 2
    assert isinstance(collections["valve"][0], PatchCollection)
    assert isinstance(collections["valve"][1], LineCollection)
    assert len(collections["valve"][0].get_paths()) == 2 * len(net.valve)
    assert len(collections["valve"][1].get_paths()) == 2 * len(net.valve)

    assert len(collections["heat_exchanger"]) == 2
    assert isinstance(collections["heat_exchanger"][0], PatchCollection)
    assert isinstance(collections["heat_exchanger"][1], LineCollection)
    assert len(collections["heat_exchanger"][0].get_paths()) == 2 * len(net.heat_exchanger[
                                                                            net.heat_exchanger.in_service])
    assert len(collections["heat_exchanger"][1].get_paths()) == 2 * len(net.heat_exchanger[
                                                                            net.heat_exchanger.in_service])

    assert len(collections["pump"]) == 2
    assert isinstance(collections["pump"][0], PatchCollection)
    assert isinstance(collections["pump"][1], LineCollection)
    assert len(collections["pump"][0].get_paths()) == len(net.pump[net.pump.in_service])
    assert len(collections["pump"][1].get_paths()) == 4 * len(net.pump[net.pump.in_service])

    assert len(collections["circ_pump_pressure"]) == 2
    assert isinstance(collections["circ_pump_pressure"][0], PatchCollection)
    assert isinstance(collections["circ_pump_pressure"][1], LineCollection)
    assert len(collections["circ_pump_pressure"][0].get_paths()) == len(net.circ_pump_pressure[
                                                                            net.circ_pump_pressure.in_service])
    assert len(collections["circ_pump_pressure"][1].get_paths()) == 4 * len(net.circ_pump_pressure[
                                                                                net.circ_pump_pressure.in_service])

    assert len(collections["circ_pump_mass"]) == 2
    assert isinstance(collections["circ_pump_mass"][0], PatchCollection)
    assert isinstance(collections["circ_pump_mass"][1], LineCollection)
    assert len(collections["circ_pump_mass"][0].get_paths()) == len(net.circ_pump_mass[
                                                                        net.circ_pump_mass.in_service])
    assert len(collections["circ_pump_mass"][1].get_paths()) == 4 * len(net.circ_pump_mass[
                                                                            net.circ_pump_mass.in_service])

