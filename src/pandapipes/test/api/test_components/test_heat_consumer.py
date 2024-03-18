# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
import copy

import pytest

import pandapipes
import pandas as pd
import numpy as np


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
    pandapipes.pipeflow(net, mode="all")

    j_mid = pandapipes.create_junctions(net2, 2, pn_bar=5, tfluid_k=283.15)
    pandapipes.create_flow_controls(net2, juncs[[1, 2]], j_mid, MDOT, diameter_m=0.1022)
    pandapipes.create_heat_exchangers(net2, j_mid, juncs[[4, 3]], 0.1022, QEXT)
    pandapipes.pipeflow(net2, mode="all")

    assert np.allclose(net.res_junction.values, net2.res_junction.iloc[:-2, :].values)


def test_heat_consumer_equivalence_bulk(simple_heat_net):
    net = copy.deepcopy(simple_heat_net)
    net2 = copy.deepcopy(simple_heat_net)
    juncs = net.junction.index

    pandapipes.create_heat_consumers(net, juncs[[1, 2]], juncs[[4, 3]], 0.1022,
                                     controlled_mdot_kg_per_s=MDOT, qext_w=QEXT)
    pandapipes.pipeflow(net, mode="all")

    j_mid = pandapipes.create_junctions(net2, 2, pn_bar=5, tfluid_k=283.15)
    pandapipes.create_flow_controls(net2, juncs[[1, 2]], j_mid, MDOT, diameter_m=0.1022)
    pandapipes.create_heat_exchangers(net2, j_mid, juncs[[4, 3]], 0.1022, QEXT)
    pandapipes.pipeflow(net2, mode="all")

    assert np.allclose(net.res_junction.values, net2.res_junction.iloc[:-2, :].values)


def test_heat_consumer_not_implemented_model(simple_heat_net):
    net = copy.deepcopy(simple_heat_net)
    juncs = net.junction.index

    with pytest.raises(NotImplementedError):
        pandapipes.create_heat_consumer(net, juncs[1], juncs[4], 0.1022,
                                        controlled_mdot_kg_per_s=MDOT[0], deltat_k=20)
    with pytest.raises(NotImplementedError):
        pandapipes.create_heat_consumer(net, juncs[1], juncs[4], 0.1022, qext_w=QEXT[0],
                                        deltat_k=20)
    with pytest.raises(NotImplementedError):
        pandapipes.create_heat_consumer(net, juncs[1], juncs[4], 0.1022,
                                        controlled_mdot_kg_per_s=MDOT[0], treturn_k=390)
    with pytest.raises(NotImplementedError):
        pandapipes.create_heat_consumer(net, juncs[1], juncs[4], 0.1022, qext_w=QEXT[0],
                                        treturn_k=390)

    with pytest.raises(NotImplementedError):
        pandapipes.create_heat_consumers(net, juncs[[1, 2]], juncs[[4, 3]], 0.1022,
                                         controlled_mdot_kg_per_s=MDOT, deltat_k=[20, None])
    with pytest.raises(NotImplementedError):
        pandapipes.create_heat_consumers(net, juncs[[1, 2]], juncs[[4, 3]], 0.1022, qext_w=QEXT,
                                         deltat_k=[20, None])
    with pytest.raises(NotImplementedError):
        pandapipes.create_heat_consumers(net, juncs[[1, 2]], juncs[[4, 3]], 0.1022,
                                         controlled_mdot_kg_per_s=MDOT, treturn_k=[390, None])
    with pytest.raises(NotImplementedError):
        pandapipes.create_heat_consumers(net, juncs[[1, 2]], juncs[[4, 3]], 0.1022, qext_w=QEXT,
                                         treturn_k=[390, None])


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
    pd.set_option("display.width", 1000)
    pd.set_option("display.max_columns", 45)
    pd.set_option("display.max_colwidth", 100)
    pd.set_option("display.max_rows", 200)
