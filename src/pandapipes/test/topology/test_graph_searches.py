# Copyright (c) 2020-2025 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import pandapipes.networks as nw
import pandapipes.topology as top
import networkx as nx
import pytest


def test_connected_components():
    net = nw.gas_meshed_two_valves()
    mg = top.create_nxgraph(net)
    assert len(list(top.connected_components(mg))) == 1
    mg = top.create_nxgraph(net, include_valves=False)
    assert len(list(top.connected_components(mg))) == 2
    mg = top.create_nxgraph(net, include_pipes=False)
    assert len(list(top.connected_components(mg))) == 7
    mg = top.create_nxgraph(net, include_pipes=False, respect_status_valves=False)
    assert len(list(top.connected_components(mg))) == 6
    mg = top.create_nxgraph(net, include_pipes=False, include_valves=False)
    assert len(list(top.connected_components(mg))) == 8


def test_calc_distance_to_junctions():
    net = nw.gas_versatility()
    egs = net.ext_grid.junction
    assert top.calc_distance_to_junctions(net, egs, respect_status_valves=True).sum() == 30.93
    assert top.calc_distance_to_junctions(net, egs, respect_status_valves=False).sum() == 26.96


def test_unsupplied_buses_with_in_service():
    net = nw.gas_versatility()
    assert top.unsupplied_junctions(net) == set()
    net.pipe.loc[7, "in_service"] = False
    assert top.unsupplied_junctions(net) == {8}


def test_elements_on_path():
    net = nw.gas_versatility()
    for multi in [True, False]:
        mg = top.create_nxgraph(net, multi=multi)
        path = nx.shortest_path(mg, 0, 6)
        assert top.elements_on_path(mg, path, "pipe") == [0, 9]
        assert top.elements_on_path(mg, path) == [0, 9]
        assert top.elements_on_path(mg, path, "valve") == []
        assert top.elements_on_path(mg, path, "pump") == [0]
        with pytest.raises(ValueError) as exception_info:
            top.elements_on_path(mg, path, element="source")
        assert str(exception_info.value) == "Invalid element type source"


if __name__ == "__main__":
    pytest.main([__file__])
