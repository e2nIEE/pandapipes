# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
import copy

import numpy as np
import pytest

import pandapipes

MDOT = [3, 2]
QEXT = [150000, 75000]


@pytest.fixture(scope="module")
def simple_heat_net():
    net = pandapipes.create_empty_network("net", add_stdtypes=False, fluid="water")
    juncs = pandapipes.create_junctions(net, 6, pn_bar=5, tfluid_k=283.15,
                                        system=["flow"] * 3 + ["return"] * 3)
    pandapipes.create_pipes_from_parameters(
        net, juncs[[0, 1, 3, 4]], juncs[[1, 2, 4, 5]], k_mm=0.1, length_km=1, inner_diameter_mm=102.2,
        system=["flow"] * 2 + ["return"] * 2, u_w_per_m2k=10, text_k=273.15
    )
    pandapipes.create_circ_pump_const_pressure(net, juncs[-1], juncs[0], 5, 2, 400, type='pt')
    return net


def test_heat_consumer_equivalence(simple_heat_net):
    net = copy.deepcopy(simple_heat_net)
    net2 = copy.deepcopy(simple_heat_net)
    juncs = net.junction.index

    pandapipes.create_heat_consumer(net, juncs[1], juncs[4], controlled_mdot_kg_per_s=MDOT[0], qext_w=QEXT[0])
    pandapipes.create_heat_consumer(net, juncs[2], juncs[3], controlled_mdot_kg_per_s=MDOT[1], qext_w=QEXT[1])
    pandapipes.pipeflow(net, mode='sequential')

    j_mid = pandapipes.create_junctions(net2, 2, pn_bar=5, tfluid_k=283.15)
    pandapipes.create_flow_controls(net2, juncs[[1, 2]], j_mid, MDOT, inner_diameter_mm=102.2)
    pandapipes.create_heat_exchangers(net2, j_mid, juncs[[4, 3]], qext_w=QEXT, inner_diameter_mm=102.2)
    pandapipes.pipeflow(net2, mode='sequential')

    assert np.allclose(net.res_junction.values, net2.res_junction.iloc[:-2, :].values)


def test_heat_consumer_equivalence_bulk(simple_heat_net):
    net = copy.deepcopy(simple_heat_net)
    net2 = copy.deepcopy(simple_heat_net)
    juncs = net.junction.index

    pandapipes.create_heat_consumers(net, juncs[[1, 2]], juncs[[4, 3]], controlled_mdot_kg_per_s=MDOT, qext_w=QEXT)
    pandapipes.pipeflow(net, mode='sequential')

    j_mid = pandapipes.create_junctions(net2, 2, pn_bar=5, tfluid_k=283.15)
    pandapipes.create_flow_controls(net2, juncs[[1, 2]], j_mid, MDOT, inner_diameter_mm=102.2)
    pandapipes.create_heat_exchangers(net2, j_mid, juncs[[4, 3]], qext_w=QEXT, inner_diameter_mm=102.2)
    pandapipes.pipeflow(net2, mode='sequential')

    assert np.allclose(net.res_junction.values, net2.res_junction.iloc[:-2, :].values)


@pytest.mark.parametrize("use_numba", [True, False])
def test_heat_consumer_equivalence2(use_numba):
    net = pandapipes.create_empty_network("net", add_stdtypes=False, fluid="water")

    mdot = [1, 1]
    qext = [150000, 75000]

    juncs = pandapipes.create_junctions(net, 6, pn_bar=5, tfluid_k=286, system=["flow"] * 3 + ["return"] * 3)
    pandapipes.create_pipes_from_parameters(net, juncs[[0, 1, 3, 4]], juncs[[1, 2, 4, 5]], k_mm=0.1, length_km=1,
                                            inner_diameter_mm=102.2, system=["flow"] * 2 + ["return"] * 2, u_w_per_m2k=10,
                                            text_k=273.15)
    pandapipes.create_circ_pump_const_pressure(net, juncs[-1], juncs[0], 5, 2, 300, type='pt')

    net2 = copy.deepcopy(net)
    net3 = copy.deepcopy(net)
    net4 = copy.deepcopy(net)
    net5 = copy.deepcopy(net)

    pandapipes.create_heat_consumer(net, juncs[1], juncs[4], controlled_mdot_kg_per_s=mdot[0], qext_w=qext[0])
    pandapipes.create_heat_consumer(net, juncs[2], juncs[3], controlled_mdot_kg_per_s=mdot[1], qext_w=qext[1])
    pandapipes.pipeflow(net, mode="bidirectional", iter=5, use_numba=use_numba)
    tout1 = net.res_heat_consumer.t_outlet_k.iloc[1]
    dt1 = net.res_heat_consumer.deltat_k.iloc[1]

    pandapipes.create_heat_consumer(net2, juncs[1], juncs[4], controlled_mdot_kg_per_s=mdot[0], qext_w=qext[0])
    pandapipes.create_heat_consumer(net2, juncs[2], juncs[3], treturn_k=tout1, qext_w=qext[1])
    pandapipes.pipeflow(net2, mode="bidirectional", iter=24, use_numba=use_numba)

    pandapipes.create_heat_consumer(net3, juncs[1], juncs[4], controlled_mdot_kg_per_s=mdot[0], qext_w=qext[0])
    pandapipes.create_heat_consumer(net3, juncs[2], juncs[3], deltat_k=dt1, qext_w=qext[1])
    pandapipes.pipeflow(net3, mode="bidirectional", iter=5, use_numba=use_numba)

    pandapipes.create_heat_consumer(net4, juncs[1], juncs[4], controlled_mdot_kg_per_s=mdot[0], qext_w=qext[0])
    pandapipes.create_heat_consumer(net4, juncs[2], juncs[3], controlled_mdot_kg_per_s=mdot[1], treturn_k=tout1)
    pandapipes.pipeflow(net4, mode="bidirectional", iter=5, use_numba=use_numba)

    pandapipes.create_heat_consumer(net5, juncs[1], juncs[4], controlled_mdot_kg_per_s=mdot[0], qext_w=qext[0])
    pandapipes.create_heat_consumer(net5, juncs[2], juncs[3], controlled_mdot_kg_per_s=mdot[1], deltat_k=dt1)
    pandapipes.pipeflow(net5, mode="bidirectional", iter=5, use_numba=use_numba)

    assert np.allclose(net2.res_junction, net.res_junction)
    assert np.allclose(net2.res_pipe, net.res_pipe)
    assert np.allclose(net3.res_junction, net.res_junction)
    assert np.allclose(net3.res_pipe, net.res_pipe)
    assert np.allclose(net4.res_junction, net.res_junction)
    assert np.allclose(net4.res_pipe, net.res_pipe)
    assert np.allclose(net5.res_junction, net.res_junction)
    assert np.allclose(net5.res_pipe, net.res_pipe)


def test_heat_consumer_creation_not_allowed(simple_heat_net):
    net = copy.deepcopy(simple_heat_net)
    juncs = net.junction.index

    with pytest.raises(AttributeError):
        # check for less than 2 set parameters
        pandapipes.create_heat_consumer(net, juncs[1], juncs[4], controlled_mdot_kg_per_s=MDOT[0], qext_w=None,
                                        treturn_k=None)
    with pytest.raises(AttributeError):
        # check for less than 2 set parameters in some consumers
        pandapipes.create_heat_consumers(net, juncs[[1, 2]], juncs[[4, 3]], controlled_mdot_kg_per_s=MDOT,
                                         qext_w=[QEXT[0], None])
    with pytest.raises(AttributeError):
        # check for less than 2 set parameters in some consumers
        pandapipes.create_heat_consumers(net, juncs[[1, 2]], juncs[[4, 3]], qext_w=QEXT,
                                         controlled_mdot_kg_per_s=[MDOT[0], None])
    with pytest.raises(AttributeError):
        # check for less than 2 set parameters in some consumers
        pandapipes.create_heat_consumers(net, juncs[[1, 2]], juncs[[4, 3]], qext_w=QEXT,
                                         controlled_mdot_kg_per_s=None)


def test_heat_consumer_creation_not_allowed_2(simple_heat_net):
    net = copy.deepcopy(simple_heat_net)
    juncs = net.junction.index
    with pytest.raises(AttributeError):
        # check for more than 2 set parameters
        pandapipes.create_heat_consumer(net, juncs[1], juncs[4], controlled_mdot_kg_per_s=MDOT[0],
                                        qext_w=QEXT[0], treturn_k=390)
    with pytest.raises(AttributeError):
        # check for deltat_k and treturn_k given
        pandapipes.create_heat_consumer(net, juncs[1], juncs[4], deltat_k=20, treturn_k=390)

    with pytest.raises(AttributeError):
        # check for more than 2 set parameters
        pandapipes.create_heat_consumers(net, juncs[[1, 2]], juncs[[4, 3]], deltat_k=[30, 40],
                                         treturn_k=[390, 385])
    with pytest.raises(AttributeError):
        # check for less than 2 set parameters in some consumers
        pandapipes.create_heat_consumers(net, juncs[[1, 2]], juncs[[4, 3]], controlled_mdot_kg_per_s=MDOT,
                                         deltat_k=[20, None])
    with pytest.raises(AttributeError):
        # check for less than 2 set parameters in some consumers
        pandapipes.create_heat_consumers(net, juncs[[1, 2]], juncs[[4, 3]], qext_w=QEXT, deltat_k=[20, None])
    with pytest.raises(AttributeError):
        # check for less than 2 set parameters in some consumers
        pandapipes.create_heat_consumers(net, juncs[[1, 2]], juncs[[4, 3]], controlled_mdot_kg_per_s=MDOT,
                                         treturn_k=[390, None])
    with pytest.raises(AttributeError):
        # check for less than 2 set parameters in some consumers
        pandapipes.create_heat_consumers(net, juncs[[1, 2]], juncs[[4, 3]], qext_w=QEXT, treturn_k=[390, None])
    with pytest.raises(AttributeError):
        # check for less than 2 set parameters in all consumers
        pandapipes.create_heat_consumers(net, juncs[[1, 2]], juncs[[4, 3]], qext_w=QEXT, treturn_k=None,
                                         controlled_mdot_kg_per_s=None)
    with pytest.raises(AttributeError):
        # check for deltat_k and treturn_k given
        pandapipes.create_heat_consumers(net, juncs[[1, 2]], juncs[[4, 3]], deltat_k=[30, 40],
                                         treturn_k=[390, 385])
    with pytest.raises(AttributeError):
        # check for deltat_k and treturn_k given as single values
        pandapipes.create_heat_consumers(net, juncs[[1, 2]], juncs[[4, 3]], deltat_k=30, treturn_k=390)


def test_heat_consumer_qext_zero():
    net = pandapipes.create_empty_network("net", add_stdtypes=False, fluid="water")

    juncs = pandapipes.create_junctions(net, 6, pn_bar=5, tfluid_k=286, system=["flow"] * 3 + ["return"] * 3)
    pandapipes.create_pipes_from_parameters(net, juncs[[0, 1, 3, 4]], juncs[[1, 2, 4, 5]], k_mm=0.1, length_km=1,
                                            inner_diameter_mm=102.2, system=["flow"] * 2 + ["return"] * 2, u_w_per_m2k=10,
                                            text_k=273.15)
    pandapipes.create_circ_pump_const_pressure(net, juncs[-1], juncs[0], 5, 2, 300, type='pt')

    pandapipes.create_heat_consumer(net, juncs[1], juncs[4], treturn_k=263.4459264973806, qext_w=0)
    pandapipes.create_heat_consumer(net, juncs[2], juncs[3], controlled_mdot_kg_per_s=1, qext_w=7500)

    pandapipes.pipeflow(net, mode="bidirectional")

    assert net.res_junction.at[juncs[4], 't_k'] != 263.4459264973806


def test_heat_consumer_qe_tr_degenerate_ignores_stale_mdot():
    """Regression test: HeatConsumer.register_hydraulic_equations's QE_TR branch used to compute
    the branch's own load (and, via an unrelated numpy view-aliasing accident in the node-balance
    load construction a few lines further down, the pit's real MDOTINIT too) straight from
    whatever mass flow happened to already be sitting in the pit for a degenerate row (t_out >=
    t_in, or qext_w == 0 - no valid mdot = qext/(cp*(t_in-t_out)) exists there). A prior working
    version reset MDOTINIT to 0 for exactly these rows before using it.

    A plain end-to-end pipeflow() can't exercise this: every hydraulics-mode run starts from a
    fresh, zero-filled pit (create_empty_pit()), and qext_w == 0 is a static, per-row property
    that's already degenerate on iteration 0 - MDOTINIT never gets the chance to become nonzero
    before the degenerate branch first runs. This test pokes a stale mass flow into the pit
    directly instead, to stand in for what a still-converging Newton iteration (or the t_out >=
    t_in half of the same condition, which - unlike qext_w == 0 - genuinely can flip mid-solve as
    temperatures evolve) would otherwise have already accumulated in that pit slot by the time
    this row goes degenerate."""
    from pandapipes.idx_branch import IdxBranch
    from pandapipes.pf.pipeflow_setup import get_lookup
    from pandapipes.pf.system_index import ComponentRegistry, HydraulicSystemIndex, HydVarEq
    from pandapipes.component_models.heat_consumer_component import HeatConsumer

    net = pandapipes.create_empty_network("net", add_stdtypes=False, fluid="water")
    juncs = pandapipes.create_junctions(net, 6, pn_bar=5, tfluid_k=286, system=["flow"] * 3 + ["return"] * 3)
    pandapipes.create_pipes_from_parameters(net, juncs[[0, 1, 3, 4]], juncs[[1, 2, 4, 5]], k_mm=0.1, length_km=1,
                                            inner_diameter_mm=102.2, system=["flow"] * 2 + ["return"] * 2, u_w_per_m2k=10,
                                            text_k=273.15)
    pandapipes.create_circ_pump_const_pressure(net, juncs[-1], juncs[0], 5, 2, 300, type='pt')
    pandapipes.create_heat_consumer(net, juncs[1], juncs[4], treturn_k=263.4459264973806, qext_w=0)
    pandapipes.create_heat_consumer(net, juncs[2], juncs[3], controlled_mdot_kg_per_s=1, qext_w=7500)

    # one plain hydraulics run just to populate the lookups/_active_pit structures this internal
    # API needs - the degenerate consumer converges to MDOTINIT == 0 here, same as always
    pandapipes.pipeflow(net, mode="hydraulics", max_iter_hyd=10)

    f, _ = get_lookup(net, "branch", "from_to_active_hydraulics")[HeatConsumer.table_name()]
    branch_pit = net["_active_pit"]["branch"]
    node_pit = net["_active_pit"]["node"]
    assert branch_pit[f, IdxBranch.QEXT] == 0  # confirms row f is the degenerate (qext_w=0) consumer

    stale_mdot = 5.0
    branch_pit[f, IdxBranch.MDOTINIT] = stale_mdot

    sys_idx = HydraulicSystemIndex(node_pit, branch_pit)
    registry = ComponentRegistry()
    HeatConsumer.register_hydraulic_equations(net, branch_pit, node_pit, sys_idx, registry)

    branch_eq_row = sys_idx.idx(HydVarEq.BRANCH, np.array([f], dtype=np.int32))[0]
    load = next(eq.load_data[eq.load_rows == branch_eq_row][0]
               for eq in registry.normal if np.any(eq.load_rows == branch_eq_row))

    assert load == 0.0
    assert branch_pit[f, IdxBranch.MDOTINIT] == 0.0

def test_heat_consumer_result_extraction():
    net = pandapipes.create_empty_network("net", add_stdtypes=False, fluid="water")

    juncs = pandapipes.create_junctions(net, 6, pn_bar=5, tfluid_k=286, system=["flow"] * 3 + ["return"] * 3)
    pandapipes.create_pipes_from_parameters(net, juncs[[0, 1, 3, 4]], juncs[[1, 2, 4, 5]], k_mm=0.1, length_km=1,
                                            inner_diameter_mm=102.2, system=["flow"] * 2 + ["return"] * 2, u_w_per_m2k=10,
                                            text_k=273.15)
    pandapipes.create_circ_pump_const_pressure(net, juncs[-1], juncs[0], 5, 2, 300, type='pt')
    pandapipes.create_heat_consumer(net, juncs[1], juncs[4], treturn_k=263.4459264973806, qext_w=7500)
    pandapipes.create_heat_consumer(net, juncs[2], juncs[3], controlled_mdot_kg_per_s=1, qext_w=7500)

    # create not connected pipe to test for active inactive missmatch
    pandapipes.create_junctions(net, 2, pn_bar=5, tfluid_k=286)
    pandapipes.create_pipe_from_parameters(net, 6, 7, k_mm=0.1, length_km=1,
                                           inner_diameter_mm=102.2, u_w_per_m2k=10, text_k=273.15)

    pandapipes.pipeflow(net, mode="bidirectional", iter=13)

    #hydraulics only to check for lookup heat transfer error
    pandapipes.pipeflow(net, iter=3)


if __name__ == '__main__':
    pytest.main([__file__])
