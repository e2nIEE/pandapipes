# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
import copy

import pandapipes
import pandas as pd
import numpy as np


def test_heat_consumer_equivalence():
    net = pandapipes.create_empty_network("net", add_stdtypes=False, fluid="water")

    mdot = [3, 2]
    qext = [150000, 75000]

    juncs = pandapipes.create_junctions(net, 6, pn_bar=5, tfluid_k=283.15,
                                        system=["flow"] * 3 + ["return"] * 3)
    pandapipes.create_pipes_from_parameters(
        net, juncs[[0, 1, 3, 4]], juncs[[1, 2, 4, 5]], k_mm=0.1, length_km=1, diameter_m=0.1022,
        system=["flow"] * 2 + ["return"] * 2, alpha_w_per_m2k=10, text_k=273.15
    )
    pandapipes.create_circ_pump_const_pressure(net, juncs[-1], juncs[0], 5, 2, 300, type='pt')

    net2 = copy.deepcopy(net)

    pandapipes.create_heat_consumer(net, juncs[1], juncs[4], 0.1022, controlled_mdot_kg_per_s=mdot[0],
                                    qext_w=qext[0])
    pandapipes.create_heat_consumer(net, juncs[2], juncs[3], 0.1022, controlled_mdot_kg_per_s=mdot[1],
                                    qext_w=qext[1])
    pandapipes.pipeflow(net, mode="all")

    j_mid = pandapipes.create_junctions(net2, 2, pn_bar=5, tfluid_k=283.15)
    pandapipes.create_flow_controls(net2, juncs[[1, 2]], j_mid, mdot, diameter_m=0.1022)
    pandapipes.create_heat_exchangers(net2, j_mid, juncs[[4, 3]], 0.1022, qext)
    pandapipes.pipeflow(net2, mode="all")

    assert np.allclose(net.res_junction.values, net2.res_junction.iloc[:-2, :].values)


if __name__ == '__main__':
    pd.set_option("display.width", 1000)
    pd.set_option("display.max_columns", 45)
    pd.set_option("display.max_colwidth", 100)
    pd.set_option("display.max_rows", 200)
