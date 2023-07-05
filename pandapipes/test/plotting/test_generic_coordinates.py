# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import pytest
import copy
from pandapipes.plotting.generic_geodata import create_generic_coordinates
from pandapipes.test.test_toolbox import base_net_is_with_pumps

try:
    import igraph

    IGRAPH_INSTALLED = True
except ImportError:
    IGRAPH_INSTALLED = False


@pytest.mark.skipif(IGRAPH_INSTALLED is False, reason="Requires python-igraph.")
def test_create_generic_coordinates_igraph(base_net_is_with_pumps):
    net = copy.deepcopy(base_net_is_with_pumps)
    net.junction_geodata.drop(net.junction_geodata.index, inplace=True)
    create_generic_coordinates(net, library="igraph")
    assert len(net.junction_geodata) == len(net.junction)


@pytest.mark.xfail(reason="The current implementation is not working properly, as multigraph edges "
                          "as AtlasViews are accessed with list logic.")
def test_create_generic_coordinates_nx(base_net_is_with_pumps):
    net = copy.deepcopy(base_net_is_with_pumps)
    net.junction_geodata.drop(net.junction_geodata.index, inplace=True)
    create_generic_coordinates(net, library="networkx")
    assert len(net.junction_geodata) == len(net.junction)


if __name__ == "__main__":
    pytest.main(["test_generic_coordinates.py"])
