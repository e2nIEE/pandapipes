import pandapipes
import pytest
import numpy as np

@pytest.mark.parametrize("use_numba", [True, False])
def test_throttle_line(use_numba):
    net = pandapipes.create_empty_network(fluid='hgas')
    j1, j2 = pandapipes.create_junctions(net, 2, 3, 300)
    pandapipes.create_ext_grid(net, j1, 2)
    pandapipes.create_sink(net, j2, 4)
    pandapipes.create_throttle_valve(net, j1, j2, j2, 3, loss_coefficient=0.5)
    pandapipes.pipeflow(net, use_numba=use_numba)
    assert np.isclose(net.res_junction.at[j2, 'p_bar'], 1.986245)


@pytest.mark.parametrize("use_numba", [True, False])
def test_throttle_reverse(use_numba):
    net = pandapipes.create_empty_network(fluid='hgas')
    j1, j2 = pandapipes.create_junctions(net, 2, 3, 300)
    pandapipes.create_ext_grid(net, j1, 5)
    pandapipes.create_source(net, j2, 4)
    pandapipes.create_throttle_valve(net, j1, j2, j2, 3, loss_coefficient=0.5)
    pandapipes.pipeflow(net, use_numba=use_numba)
    assert np.isnan(net.res_junction.at[j2, 'p_bar'])


@pytest.mark.parametrize("use_numba", [True, False])
def test_throttle_default(use_numba):
    net = pandapipes.create_empty_network(fluid='hgas')
    j1, j2 = pandapipes.create_junctions(net, 2, 3, 300)
    pandapipes.create_ext_grid(net, j1, 5)
    pandapipes.create_sink(net, j2, 4)
    pandapipes.create_throttle_valve(net, j1, j2, j2, 3, loss_coefficient=0.5)
    pandapipes.pipeflow(net, use_numba=use_numba)
    assert np.isclose(net.res_junction.at[j2, 'p_bar'], 3)


@pytest.mark.parametrize("use_numba", [True, False])
def test_throttle_max_mdot(use_numba):
    net = pandapipes.create_empty_network(fluid='water')
    j1, j2, j3, j4 = pandapipes.create_junctions(net, 4, 3, 300)
    pandapipes.create_circ_pump_const_pressure(net, j4, j1,  10, 8)
    pandapipes.create_pipes_from_parameters(net, [j1, j2, j3], [j2, j4,  j4], 0.1, 0.1)
    pandapipes.create_throttle_valve(net, j2, j3, j3, 3, loss_coefficient=0.5, max_mdot_kg_per_s=10)
    pandapipes.pipeflow(net, use_numba=use_numba)
    assert np.isclose(net.res_throttle_valve.mdot_from_kg_per_s.values, 10)


if __name__ == '__main__':
    n = pytest.main(["test_throttle_valve.py"])
