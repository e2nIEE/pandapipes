import pandapipes
import pytest
import copy
import numpy as np


@pytest.fixture
def create_test_net():
    """

    :return:
    :rtype:
    """
    net = pandapipes.create_empty_network(fluid="hgas")
    j1 = pandapipes.create_junction(net, 1, 293.15, index=1)
    j2 = pandapipes.create_junction(net, 1, 293.15, index=2)
    j3 = pandapipes.create_junction(net, 1, 293.15, index=4)
    j4 = pandapipes.create_junction(net, 1, 293.15, index=5)
    j5 = pandapipes.create_junction(net, 1, 293.15, index=6)
    j6 = pandapipes.create_junction(net, 1, 293.15, index=7)

    pandapipes.create_ext_grid(net, j2, 1, 285.15, type="pt")
    pandapipes.create_ext_grid(net, j3, 1, 285.15, type="pt")
    pandapipes.create_ext_grid(net, j5, 1, 285.15, type="t")
    pandapipes.create_ext_grid(net, j1, 1, 285.15, type="pt")
    pandapipes.create_ext_grid(net, j1, 1, 285.15, type="pt")

    pandapipes.create_pipe_from_parameters(net, j1, j4, 0.1, 0.1)
    pandapipes.create_pipe_from_parameters(net, j2, j5, 0.1, 0.1)
    pandapipes.create_pipe_from_parameters(net, j3, j6, 0.1, 0.1)

    pandapipes.create_sink(net, j4, mdot_kg_per_s=0.1)
    pandapipes.create_sink(net, j5, mdot_kg_per_s=0.1)
    pandapipes.create_sink(net, j6, mdot_kg_per_s=0.1)
    pandapipes.create_sink(net, j2, mdot_kg_per_s=0.02)

    return net


def test_ext_grid_sorting(create_test_net):
    net = copy.deepcopy(create_test_net)
    pandapipes.pipeflow(net)

    assert np.isclose(net.res_ext_grid.at[0, "mdot_kg_per_s"], -0.12, atol=1e-12, rtol=1e-12)
    assert np.isclose(net.res_ext_grid.at[1, "mdot_kg_per_s"], -0.1, atol=1e-12, rtol=1e-12)
    assert np.isnan(net.res_ext_grid.at[2, "mdot_kg_per_s"])
    assert np.isclose(net.res_ext_grid.at[3, "mdot_kg_per_s"], -0.05, atol=1e-12, rtol=1e-12)
    assert np.isclose(net.res_ext_grid.at[4, "mdot_kg_per_s"], -0.05, atol=1e-12, rtol=1e-12)


if __name__ == '__main__':
    pytest.main([r'pandapipes/test/pipeflow_internals/test_ext_grid.py'])
