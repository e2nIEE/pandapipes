# Copyright (c) 2020-2025 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import pandapipes.networks as nw
import pandapipes.topology as top


def test_include_branches():
    net = nw.gas_versatility()
    
    mg = top.create_nxgraph(net, include_pipes=False, include_valves=False, include_pumps=False)
    assert len(mg.nodes()) == len(net.junction)
    assert len(mg.edges()) == 0

    mg = top.create_nxgraph(net, include_pipes=True, include_valves=False, include_pumps=False)
    assert len(mg.nodes()) == len(net.junction)
    assert len(mg.edges()) == sum(net.pipe.in_service)

    mg = top.create_nxgraph(net, include_pipes=True, include_valves=True, include_pumps=False)
    assert len(mg.nodes()) == len(net.junction)
    assert len(mg.edges()) == sum(net.pipe.in_service) + sum(net.valve.opened)
    
    mg = top.create_nxgraph(net, include_pipes=True, include_valves=True, include_pumps=True)
    assert len(mg.nodes()) == len(net.junction)
    assert len(mg.edges()) == sum(net.pipe.in_service) + sum(net.valve.opened) \
           + sum(net.pump.in_service)
