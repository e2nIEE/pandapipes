# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import copy

import pandapipes.plotting as plot
from matplotlib.collections import PatchCollection, LineCollection
from pandapipes.test.test_toolbox import net_plotting


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
