# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.


import pandapipes.plotting as plot
import numpy as np
import pandapipes as pp

def test_pressure_profile_to_junction_geodata():
    net = pp.create_empty_network(fluid="lgas")
    j1 = pp.create_junction(net, pn_bar=1.05, tfluid_k=293.15)
    j2 = pp.create_junction(net, pn_bar=1.05, tfluid_k=293.15)
    j3 = pp.create_junction(net, pn_bar=1.05, tfluid_k=293.15)
    j4 = pp.create_junction(net, pn_bar=1.05, tfluid_k=293.15)

    pp.create_ext_grid(net, junction=j1, p_bar=1.1, t_k=293.15)
    pp.create_sink(net, junction=j4, mdot_kg_per_s=0.01)

    pp.create_pipe_from_parameters(net, from_junction=j1, to_junction=j2,
                                          length_km=0.2, diameter_m=0.05)

    pp.create_pipe_from_parameters(net, from_junction=j2, to_junction=j3,
                                          length_km=0.3, diameter_m=0.05)
    pp.create_pipe_from_parameters(net, from_junction=j3, to_junction=j4,
                                          length_km=0.4, diameter_m=0.05)

    pp.pipeflow(net)

    jgd = plot.pressure_profile_to_junction_geodata(net)

    assert jgd.y.loc[jgd.index].equals(net.res_junction.p_bar.loc[jgd.index])
    assert np.isclose(jgd.x.loc[jgd.index].values, [0.0, 0.2, 0.5, 0.9]).all()

    #add parallel pipe to test meshed topology
    pp.create_pipe_from_parameters(net, from_junction=j1, to_junction=j4,
                                          length_km=0.2, diameter_m=0.05,
                                          name="Pipe 1")
    pp.pipeflow(net)
    jgd = plot.pressure_profile_to_junction_geodata(net)
    assert jgd.y.loc[jgd.index].equals(net.res_junction.p_bar.loc[jgd.index])
    assert np.isclose(jgd.x.loc[jgd.index].values, [0.0, 0.2, 0.5, 0.2]).all()