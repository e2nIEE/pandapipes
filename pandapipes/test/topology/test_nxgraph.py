# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import pytest
import pandapipes.networks as nw 
import pandapipes.topology as top


def test_include_branches():
    net = nw.gas_versatility()
    
    mg = top.create_nxgraph(net, include_pipes=False, include_valves=False, include_pumps=False)
    assert len(mg.nodes()) == len(net.junction)
    assert len(mg.edges()) == 0

    mg = top.create_nxgraph(net, include_pipes=True, include_valves=False, include_pumps=False)
    assert len(mg.nodes()) == len(net.junction)
    assert len(mg.edges()) == len(net.pipe)

    mg = top.create_nxgraph(net, include_pipes=True, include_valves=True, include_pumps=False)
    assert len(mg.nodes()) == len(net.junction)
    assert len(mg.edges()) == len(net.pipe)

    mg = top.create_nxgraph(net, include_pipes=True, include_valves=True, include_pumps=False,
                            neglect_in_service=["valve"])
    assert len(mg.nodes()) == len(net.junction)
    assert len(mg.edges()) == len(net.pipe) + len(net.valve)
    
    mg = top.create_nxgraph(net, include_pipes=True, include_valves=True, include_pumps=True,
                            neglect_in_service=["valve"])
    assert len(mg.nodes()) == len(net.junction)
    assert len(mg.edges()) == len(net.pipe) + len(net.valve) + len(net.pump)


if __name__ == '__main__':
    pytest.main(["test_nxgraph.py"])
