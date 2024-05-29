# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
import copy

import pytest

import pandapipes
import numpy as np
import pytest


MDOT = [3, 2]
QEXT = [150000, 75000]


@pytest.fixture(scope="module")
def simple_heat_net():
    net = pandapipes.create_empty_network("net", add_stdtypes=False, fluid="water")

    juncs = pandapipes.create_junctions(net, 6, pn_bar=5, tfluid_k=283.15,
                                        system=["flow"] * 3 + ["return"] * 3)
    pandapipes.create_pipes_from_parameters(
        net, juncs[[0, 1, 3, 4]], juncs[[1, 2, 4, 5]], k_mm=0.1, length_km=1, diameter_m=0.1022,
        system=["flow"] * 2 + ["return"] * 2, alpha_w_per_m2k=10, text_k=273.15
    )
    pandapipes.create_circ_pump_const_pressure(net, juncs[-1], juncs[0], 5, 2, 400, type='pt')
    return net


def test_heat_consumer_equivalence(simple_heat_net):
    net = copy.deepcopy(simple_heat_net)
    net2 = copy.deepcopy(simple_heat_net)
    juncs = net.junction.index

    pandapipes.create_heat_consumer(net, juncs[1], juncs[4], 0.1022,
                                    controlled_mdot_kg_per_s=MDOT[0], qext_w=QEXT[0])
    pandapipes.create_heat_consumer(net, juncs[2], juncs[3], 0.1022,
                                    controlled_mdot_kg_per_s=MDOT[1], qext_w=QEXT[1])
    pandapipes.pipeflow(net, mode='sequential')

    j_mid = pandapipes.create_junctions(net2, 2, pn_bar=5, tfluid_k=283.15)
    pandapipes.create_flow_controls(net2, juncs[[1, 2]], j_mid, MDOT, diameter_m=0.1022)
    pandapipes.create_heat_exchangers(net2, j_mid, juncs[[4, 3]], 0.1022, QEXT)
    pandapipes.pipeflow(net2, mode='sequential')

    assert np.allclose(net.res_junction.values, net2.res_junction.iloc[:-2, :].values)


def test_heat_consumer_equivalence_bulk(simple_heat_net):
    net = copy.deepcopy(simple_heat_net)
    net2 = copy.deepcopy(simple_heat_net)
    juncs = net.junction.index

    pandapipes.create_heat_consumers(net, juncs[[1, 2]], juncs[[4, 3]], 0.1022,
                                     controlled_mdot_kg_per_s=MDOT, qext_w=QEXT)
    pandapipes.pipeflow(net, mode='sequential')

    j_mid = pandapipes.create_junctions(net2, 2, pn_bar=5, tfluid_k=283.15)
    pandapipes.create_flow_controls(net2, juncs[[1, 2]], j_mid, MDOT, diameter_m=0.1022)
    pandapipes.create_heat_exchangers(net2, j_mid, juncs[[4, 3]], 0.1022, QEXT)
    pandapipes.pipeflow(net2, mode='sequential')

    assert np.allclose(net.res_junction.values, net2.res_junction.iloc[:-2, :].values)

def test_heat_consumer_equivalence2():
    net = pandapipes.create_empty_network("net", add_stdtypes=False, fluid="water")

    mdot = [3, 3]
    qext = [150000, 75000]

    juncs = pandapipes.create_junctions(net, 6, pn_bar=5, tfluid_k=286,
                                        system=["flow"] * 3 + ["return"] * 3)
    pandapipes.create_pipes_from_parameters(
        net, juncs[[0, 1, 3, 4]], juncs[[1, 2, 4, 5]], k_mm=0.1, length_km=1, diameter_m=0.1022,
        system=["flow"] * 2 + ["return"] * 2, alpha_w_per_m2k=10, text_k=273.15
    )
    pandapipes.create_circ_pump_const_pressure(net, juncs[-1], juncs[0], 5, 2, 300, type='pt')

    net2 = copy.deepcopy(net)
    net3 = copy.deepcopy(net)
    net4 = copy.deepcopy(net)
    net5 = copy.deepcopy(net)

    pandapipes.create_heat_consumer(net, juncs[1], juncs[4], 0.1022, controlled_mdot_kg_per_s=mdot[0],
                                    qext_w=qext[0])
    pandapipes.create_heat_consumer(net, juncs[2], juncs[3], 0.1022, controlled_mdot_kg_per_s=mdot[1],
                                    qext_w=qext[1])
    pandapipes.pipeflow(net, mode="bidirectional")

    pandapipes.create_heat_consumer(net2, juncs[1], juncs[4], 0.1022, controlled_mdot_kg_per_s=mdot[0],
                                    qext_w=qext[0])
    pandapipes.create_heat_consumer(net2, juncs[2], juncs[3], 0.1022, treturn_k= 285.4461399735642,
                                    qext_w=qext[1])
    pandapipes.pipeflow(net2, mode="bidirectional", iter=20, alpha=0.65)

    pandapipes.create_heat_consumer(net3, juncs[1], juncs[4], 0.1022, controlled_mdot_kg_per_s=mdot[0],
                                    qext_w=qext[0])
    pandapipes.create_heat_consumer(net3, juncs[2], juncs[3], 0.1022, deltat_k=5.9673377831196035,
                                    qext_w=qext[1])
    pandapipes.pipeflow(net3, mode="bidirectional")

    pandapipes.create_heat_consumer(net4, juncs[1], juncs[4], 0.1022, controlled_mdot_kg_per_s=mdot[0],
                                    qext_w=qext[0])
    pandapipes.create_heat_consumer(net4, juncs[2], juncs[3], 0.1022, controlled_mdot_kg_per_s=mdot[1],
                                    treturn_k= 285.4461399735642)
    pandapipes.pipeflow(net4, mode="bidirectional")

    pandapipes.create_heat_consumer(net5, juncs[1], juncs[4], 0.1022, controlled_mdot_kg_per_s=mdot[0],
                                    qext_w=qext[0])
    pandapipes.create_heat_consumer(net5, juncs[2], juncs[3], 0.1022, controlled_mdot_kg_per_s=mdot[1],
                                    deltat_k=5.9673377831196035)
    pandapipes.pipeflow(net5, mode="bidirectional")

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
        pandapipes.create_heat_consumer(net, juncs[1], juncs[4], 0.1022,
                                        controlled_mdot_kg_per_s=MDOT[0], qext_w=None,
                                        treturn_k=None)
    with pytest.raises(AttributeError):
        # check for less than 2 set parameters in some consumers
        pandapipes.create_heat_consumers(net, juncs[[1, 2]], juncs[[4, 3]], 0.1022,
                                         controlled_mdot_kg_per_s=MDOT, qext_w=[QEXT[0], None])
    with pytest.raises(AttributeError):
        # check for less than 2 set parameters in some consumers
        pandapipes.create_heat_consumers(net, juncs[[1, 2]], juncs[[4, 3]], 0.1022, qext_w=QEXT,
                                         controlled_mdot_kg_per_s=[MDOT[0], None])
    with pytest.raises(AttributeError):
        # check for less than 2 set parameters in some consumers
        pandapipes.create_heat_consumers(net, juncs[[1, 2]], juncs[[4, 3]], 0.1022, qext_w=QEXT,
                                         controlled_mdot_kg_per_s=None)


@pytest.mark.xfail(reason="Can only be tested once models for deltat_k and treturn_k"
                          " are implemented for heat consumers.")
def test_heat_consumer_creation_not_allowed_2(simple_heat_net):
    net = copy.deepcopy(simple_heat_net)
    juncs = net.junction.index
    with pytest.raises(AttributeError):
        # check for more than 2 set parameters
        pandapipes.create_heat_consumer(net, juncs[1], juncs[4], 0.1022,
                                        controlled_mdot_kg_per_s=MDOT[0], qext_w=QEXT[0],
                                        treturn_k=390)
    with pytest.raises(AttributeError):
        # check for deltat_k and treturn_k given
        pandapipes.create_heat_consumer(net, juncs[1], juncs[4], 0.1022, deltat_k=20, treturn_k=390)

    with pytest.raises(AttributeError):
        # check for more than 2 set parameters
        pandapipes.create_heat_consumers(net, juncs[[1, 2]], juncs[[4, 3]], 0.1022,
                                         deltat_k=[30, 40], treturn_k=[390, 385])
    with pytest.raises(AttributeError):
        # check for less than 2 set parameters in some consumers
        pandapipes.create_heat_consumers(net, juncs[[1, 2]], juncs[[4, 3]], 0.1022,
                                         controlled_mdot_kg_per_s=MDOT, deltat_k=[20, None])
    with pytest.raises(AttributeError):
        # check for less than 2 set parameters in some consumers
        pandapipes.create_heat_consumers(net, juncs[[1, 2]], juncs[[4, 3]], 0.1022, qext_w=QEXT,
                                         deltat_k=[20, None])
    with pytest.raises(AttributeError):
        # check for less than 2 set parameters in some consumers
        pandapipes.create_heat_consumers(net, juncs[[1, 2]], juncs[[4, 3]], 0.1022,
                                         controlled_mdot_kg_per_s=MDOT, treturn_k=[390, None])
    with pytest.raises(AttributeError):
        # check for less than 2 set parameters in some consumers
        pandapipes.create_heat_consumers(net, juncs[[1, 2]], juncs[[4, 3]], 0.1022, qext_w=QEXT,
                                         treturn_k=[390, None])
    with pytest.raises(AttributeError):
        # check for less than 2 set parameters in all consumers
        pandapipes.create_heat_consumers(net, juncs[[1, 2]], juncs[[4, 3]], 0.1022, qext_w=QEXT,
                                         treturn_k=None, controlled_mdot_kg_per_s=None)
    with pytest.raises(AttributeError):
        # check for deltat_k and treturn_k given
        pandapipes.create_heat_consumers(net, juncs[[1, 2]], juncs[[4, 3]], 0.1022,
                                         deltat_k=[30, 40], treturn_k=[390, 385])
    with pytest.raises(AttributeError):
        # check for deltat_k and treturn_k given as single values
        pandapipes.create_heat_consumers(net, juncs[[1, 2]], juncs[[4, 3]], 0.1022,
                                         deltat_k=30, treturn_k=390)


if __name__ == '__main__':
    pytest.main([__file__])
