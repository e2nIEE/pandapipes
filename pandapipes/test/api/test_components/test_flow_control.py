import numpy as np
import pytest

import pandapipes


@pytest.mark.parametrize("use_numba", [True, False])
def test_flow_control_simple(use_numba):
    net = pandapipes.create_empty_network("net", add_stdtypes=True, fluid="water")

    j = pandapipes.create_junctions(net, 7, pn_bar=5, tfluid_k=360)
    j1, j2, j3, j4, j5, j6, j7 = j

    p12, p25, p47, p64 = pandapipes.create_pipes_from_parameters(
        net, [j1, j2, j4, j6], [j2, j5, j7, j4], 0.2, 0.1, k_mm=0.1, alpha_w_per_m2k=20.,
        text_k=280)

    pandapipes.create_heat_exchanger(net, j3, j4, 0.1, 50000, 1)
    pandapipes.create_heat_exchanger(net, j5, j6, 0.1, 50000, 1)

    pandapipes.create_flow_control(net, j2, j3, 2, 0.1)

    pandapipes.create_ext_grid(net, j1, p_bar=5, t_k=360, type="pt")

    pandapipes.create_sink(net, j7, 3)

    pandapipes.pipeflow(net, mode="all", use_numba=use_numba)

    assert np.allclose(net.res_pipe.loc[[p12, p47, p25, p64], "mdot_from_kg_per_s"], [3, 3, 1, 1])
    assert np.allclose(net.res_pipe.loc[[p12, p47, p25, p64], "mdot_from_kg_per_s"], [3, 3, 1, 1])
    assert np.isclose(net.res_flow_control.at[0, "mdot_from_kg_per_s"], 2)
