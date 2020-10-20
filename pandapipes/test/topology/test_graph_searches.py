# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import pytest
import pandapipes.networks as nw 
import pandapipes.topology as top

def test_connected_components():
    net = nw.gas_meshed_two_valves()
    mg = top.create_nxgraph(net)
    assert len(list(top.connected_components(mg))) == 1
    mg = top.create_nxgraph(net, include_valves=False)
    assert len(list(top.connected_components(mg))) == 2
    mg = top.create_nxgraph(net, include_pipes=False)
    assert len(list(top.connected_components(mg))) == 6
    mg = top.create_nxgraph(net, include_pipes=False, include_valves=False)
    assert len(list(top.connected_components(mg))) == 8
