# Copyright (c) 2020-2025 by Fraunhofer Institute for Energy Economics
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

    max_iter_hyd = 3
    pp.pipeflow(net, max_iter_hyd=max_iter_hyd)

    jgd = plot.pressure_profile_to_junction_geodata(net)

    assert jgd.y.loc[jgd.index].equals(net.res_junction.p_bar.loc[jgd.index])
    assert np.isclose(jgd.x.loc[jgd.index].values, [0.0, 0.2, 0.5, 0.9]).all()

    #add parallel pipe to test meshed topology
    pp.create_pipe_from_parameters(net, from_junction=j1, to_junction=j4,
                                          length_km=0.2, diameter_m=0.05,
                                          name="Pipe 1")
    max_iter_hyd = 4
    pp.pipeflow(net, max_iter_hyd=max_iter_hyd)
    jgd = plot.pressure_profile_to_junction_geodata(net)
    assert jgd.y.loc[jgd.index].equals(net.res_junction.p_bar.loc[jgd.index])
    assert np.isclose(jgd.x.loc[jgd.index].values, [0.0, 0.2, 0.5, 0.2]).all()


# TODO: Why is this test not on develop? What does it do?
def test_results_gas_velocity_entries():
    net = pp.create_empty_network(fluid="lgas")
    j1 = pp.create_junction(net, pn_bar=1.05, tfluid_k=293.15, name="Junction 1")
    j2 = pp.create_junction(net, pn_bar=1.05, tfluid_k=293.15, name="Junction 2")
    j3 = pp.create_junction(net, pn_bar=1.05, tfluid_k=293.15, name="Junction 3")
    j4 = pp.create_junction(net, pn_bar=1.05, tfluid_k=293.15, name="Junction 4")


    #create junction elements
    ext_grid = pp.create_ext_grid(net, junction=j1, p_bar=2, t_k=293.15, name="Grid Connection")

    sink = pp.create_sink(net, junction=j3, mdot_kg_per_s=0.04, name="Sink")
    sink = pp.create_sink(net, junction=j4, mdot_kg_per_s=0.05, name="Sink")

    pp.create_pipe_from_parameters(net, j1, j2, 1, 75e-3, k_mm=.1, sections=3, alpha_w_per_m2k=5, text_k=293.15)
    pp.create_pipe_from_parameters(net, j2, j3, 1, 60e-3, k_mm=.1, sections=4, alpha_w_per_m2k=5, text_k=293.15)
    pp.create_pipe_from_parameters(net, j2, j4, 1, 60e-3, k_mm=.1, sections=6, alpha_w_per_m2k=5, text_k=293.15)


    pp.pipeflow(net)

    assert(net.res_pipe.loc[0, "v_from_m_per_s"] <= net.res_pipe.loc[0, "v_to_m_per_s"])
    assert(net.res_pipe.loc[1, "v_from_m_per_s"] <= net.res_pipe.loc[1, "v_to_m_per_s"])
    assert(net.res_pipe.loc[2, "v_from_m_per_s"] <= net.res_pipe.loc[2, "v_to_m_per_s"])


    cont = net.pipe.loc[0, "from_junction"]
    net.pipe.loc[0, "from_junction"] = net.pipe.loc[0, "to_junction"]
    net.pipe.loc[0, "to_junction"] = cont

    cont = net.pipe.loc[1, "from_junction"]
    net.pipe.loc[1, "from_junction"] = net.pipe.loc[1, "to_junction"]
    net.pipe.loc[1, "to_junction"] = cont

    pp.pipeflow(net)

    assert(net.res_pipe.loc[0, "v_from_m_per_s"] <= net.res_pipe.loc[0, "v_to_m_per_s"])
    assert(net.res_pipe.loc[1, "v_from_m_per_s"] <= net.res_pipe.loc[1, "v_to_m_per_s"])
