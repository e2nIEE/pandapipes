# Copyright (c) 2020-2026 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import pandapipes.networks as nw
import pandapipes.topology as top
import pandapipes


def test_pi_valve_does_not_leak_pipe_index_as_node():
    """A "pi" valve's element is a pipe index, not a junction - it must not become a graph node."""
    net = pandapipes.create_empty_network(fluid="water")
    j0 = pandapipes.create_junction(net, pn_bar=1, tfluid_k=293, index=100)
    j1 = pandapipes.create_junction(net, pn_bar=1, tfluid_k=293, index=101)
    j2 = pandapipes.create_junction(net, pn_bar=1, tfluid_k=293, index=102)
    pipe_idx = pandapipes.create_pipe_from_parameters(net, j0, j1, length_km=0.1,
                                              diameter_m=0.1, index=5)
    pandapipes.create_valve(net, j1, j2, et="ju", inner_diameter_mm=100, opened=True)
    pi_valve_idx = pandapipes.create_valve(net, j0, pipe_idx, et="pi", inner_diameter_mm=100,
                                   opened=True)

    mg = top.create_nxgraph(net, respect_status_valves=True)
    assert pipe_idx not in mg.nodes()
    assert set(net.junction.index) <= set(mg.nodes())
    assert mg.has_edge(j1, j2)
    assert mg.has_edge(j0, j1)

    net.valve.loc[pi_valve_idx, "opened"] = False
    mg_closed = top.create_nxgraph(net, respect_status_valves=True)
    assert pipe_idx not in mg_closed.nodes()
    assert not mg_closed.has_edge(j0, j1)
    assert mg_closed.has_edge(j1, j2)


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
