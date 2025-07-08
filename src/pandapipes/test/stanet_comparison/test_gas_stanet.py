# Copyright (c) 2020-2025 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
import pytest

import pandapipes.networks.simple_gas_networks as nw
from pandapipes.pipeflow import logger as pf_logger
from pandapipes.test.stanet_comparison.pipeflow_stanet_comparison import pipeflow_stanet_comparison

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
pf_logger.setLevel(logging.WARNING)


# ---------- TEST AREA: combined networks ----------
# parallel_N
@pytest.mark.parametrize("use_numba", [True, False])
def test_case_3parallel_n(use_numba, log_results=False):
    """

    :param log_results:
    :type log_results:
    :return:
    :rtype:
    """
    net = nw.gas_3parallel(method="n")
    max_iter_hyd = 4 if use_numba else 4
    p_diff, v_diff_abs = pipeflow_stanet_comparison(net, log_results,
                                                    max_iter_hyd=max_iter_hyd,
                                                    use_numba=use_numba)
    assert np.all(p_diff < 1e-4)
    assert np.all(v_diff_abs < 1e-2)


# parallel_PC
@pytest.mark.parametrize("use_numba", [True, False])
def test_case_combined_3parallel_pc(use_numba, log_results=False):
    """

    :param log_results:
    :type log_results:
    :return:
    :rtype:
    """
    net = nw.gas_3parallel(method="pc")
    max_iter_hyd = 5 if use_numba else 5
    p_diff, v_diff_abs = pipeflow_stanet_comparison(net, log_results, max_iter_hyd=max_iter_hyd,
                                                    friction_model="colebrook",
                                                    use_numba=use_numba)
    assert np.all(p_diff < 1e-4)
    assert np.all(v_diff_abs < 1e-2)


# ---------- TEST AREA: meshed networks ----------
# square_N
@pytest.mark.parametrize("use_numba", [True, False])
def test_case_square_n(use_numba, log_results=False):
    """

    :param log_results:
    :type log_results:
    :return:
    :rtype:
    """
    net = nw.gas_meshed_square(method="n")
    max_iter_hyd = 3 if use_numba else 3
    p_diff, v_diff_abs = pipeflow_stanet_comparison(net, log_results,
                                                    max_iter_hyd=max_iter_hyd,
                                                    use_numba=use_numba)
    assert np.all(p_diff < 1e-4)
    assert np.all(v_diff_abs < 1e-2)


# square_PC
@pytest.mark.parametrize("use_numba", [True, False])
def test_case_square_pc(use_numba, log_results=False):
    """

    :param log_results:
    :type log_results:
    :return:
    :rtype:
    """
    net = nw.gas_meshed_square(method="pc")
    max_iter_hyd = 5 if use_numba else 5
    p_diff, v_diff_abs = pipeflow_stanet_comparison(net, log_results,
                                                    max_iter_hyd=max_iter_hyd,
                                                    friction_model="colebrook",
                                                    use_numba=use_numba)
    assert np.all(p_diff < 2e-3)
    assert np.all(v_diff_abs < 3e-2)


# delta_PC
@pytest.mark.parametrize("use_numba", [True, False])
def test_case_meshed_delta_pc(use_numba, log_results=False):
    """

    :param log_results:
    :type log_results:
    :return:
    :rtype:
    """
    net = nw.gas_meshed_delta()
    max_iter_hyd = 13 if use_numba else 13
    p_diff, v_diff_abs = pipeflow_stanet_comparison(net, log_results,
                                                    max_iter_hyd=max_iter_hyd,
                                                    friction_model="colebrook",
                                                    use_numba=use_numba)
    assert np.all(p_diff < 2e-3)
    assert np.all(v_diff_abs < 3e-2)


# pumps_N
@pytest.mark.parametrize("use_numba", [True, False])
def test_case_meshed_pumps(use_numba, log_results=False):
    """
    # TODO: check the big differences in v
    :param log_results:
    :type log_results:
    :return:
    :rtype:
    """
    net = nw.gas_meshed_pumps()
    max_iter_hyd = 15 if use_numba else 15
    p_diff, v_diff_abs = pipeflow_stanet_comparison(net, log_results,
                                                    max_iter_hyd=max_iter_hyd,
                                                    use_numba=use_numba,
                                                    calc_compression_power=False)
    assert np.all(p_diff < 2e-3)
    assert np.all(v_diff_abs < 3e-2)


# two_valves_N
@pytest.mark.parametrize("use_numba", [True, False])
def test_case_meshed_2valves_n(use_numba, log_results=False):
    net = nw.gas_meshed_two_valves(method="n")
    max_iter_hyd = 5 if use_numba else 5
    p_diff, v_diff_abs = pipeflow_stanet_comparison(
        net, log_results,
        max_iter_hyd=max_iter_hyd,
        use_numba=use_numba)
    assert np.all(p_diff < 1e-3)
    assert np.all(v_diff_abs < 1e-3)


# two_valves_PC
@pytest.mark.parametrize("use_numba", [True, False])
def test_case_meshed_2valves_pc(use_numba, log_results=False):
    net = nw.gas_meshed_two_valves(method="pc")
    max_iter_hyd = 5 if use_numba else 5
    p_diff, v_diff_abs = pipeflow_stanet_comparison(net, log_results,
                                                    max_iter_hyd=max_iter_hyd,
                                                    friction_model="colebrook",
                                                    use_numba=use_numba)
    assert np.all(p_diff < 1e-4)
    assert np.all(v_diff_abs < 1e-3)


# ---------- TEST AREA: one pipe ----------
# pipe_1_N
@pytest.mark.parametrize("use_numba", [True, False])
def test_case_one_pipe1_n(use_numba, log_results=False):
    """

    :param log_results:
    :type log_results:
    :return:
    :rtype:
    """
    net = nw.gas_one_pipe1(method="n")
    max_iter_hyd = 5 if use_numba else 5
    p_diff, v_diff_abs = pipeflow_stanet_comparison(net, log_results,
                                                    max_iter_hyd=max_iter_hyd,
                                                    use_numba=use_numba)
    assert np.all(p_diff < 1e-4)
    assert np.all(v_diff_abs < 1e-2)


# pipe_1_PC
@pytest.mark.parametrize("use_numba", [True, False])
def test_case_one_pipe1_pc(use_numba, log_results=False):
    """

    :param log_results:
    :type log_results:
    :return:
    :rtype:
    """
    net = nw.gas_one_pipe1(method="pc")
    max_iter_hyd = 5 if use_numba else 5
    p_diff, v_diff_abs = pipeflow_stanet_comparison(net, log_results,
                                                    max_iter_hyd=max_iter_hyd,
                                                    friction_model="colebrook",
                                                    use_numba=use_numba)
    assert np.all(p_diff < 1e-4)
    assert np.all(v_diff_abs < 1e-2)


# pipe_2_N
@pytest.mark.parametrize("use_numba", [True, False])
def test_case_one_pipe2_n(use_numba, log_results=False):
    """

    :param log_results:
    :type log_results:
    :return:
    :rtype:
    """
    net = nw.gas_one_pipe2(method="n")
    max_iter_hyd = 4 if use_numba else 4
    p_diff, v_diff_abs = pipeflow_stanet_comparison(net, log_results,
                                                    max_iter_hyd=max_iter_hyd,
                                                    use_numba=use_numba)
    assert np.all(p_diff < 2e-3)
    assert np.all(v_diff_abs < 3e-2)


# pipe_2_PC
@pytest.mark.parametrize("use_numba", [True, False])
def test_case_one_pipe2_pc(use_numba, log_results=False):
    """

    :param log_results:
    :type log_results:
    :return:
    :rtype:
    """
    net = nw.gas_one_pipe2(method="pc")
    max_iter_hyd = 5 if use_numba else 5
    p_diff, v_diff_abs = pipeflow_stanet_comparison(net, log_results,
                                                    max_iter_hyd=max_iter_hyd,
                                                    friction_model="colebrook",
                                                    use_numba=use_numba)
    assert np.all(p_diff < 1e-4)
    assert np.all(v_diff_abs < 1e-2)


# ---------- TEST AREA: strand net ----------
# two_pipes_N
@pytest.mark.parametrize("use_numba", [True, False])
def test_case_strand_2pipes_n(use_numba, log_results=False):
    """

    :param log_results:
    :type log_results:
    :return:
    :rtype:
    """
    net = nw.gas_strand_2pipes(method="n")
    max_iter_hyd = 3 if use_numba else 3
    p_diff, v_diff_abs = pipeflow_stanet_comparison(net, log_results,
                                                    max_iter_hyd=max_iter_hyd,
                                                    use_numba=use_numba)
    assert np.all(p_diff < 1e-4)
    assert np.all(v_diff_abs < 1e-2)


# two_pipes_PC
@pytest.mark.parametrize("use_numba", [True, False])
def test_case_strand_2pipes_pc(use_numba, log_results=False):
    """

    :param log_results:
    :type log_results:
    :return:
    :rtype:
    """
    net = nw.gas_strand_2pipes(method="pc")
    max_iter_hyd = 3 if use_numba else 3
    p_diff, v_diff_abs = pipeflow_stanet_comparison(net, log_results,
                                                    max_iter_hyd=max_iter_hyd,
                                                    friction_model="colebrook",
                                                    use_numba=use_numba)
    assert np.all(p_diff < 1e-4)
    assert np.all(v_diff_abs < 1e-2)

# pump_N
@pytest.mark.parametrize("use_numba", [True, False])
def test_case_strand_pump(use_numba, log_results=False):
    """

    :param log_results:
    :type log_results:
    :return:
    :rtype:
    """
    net = nw.gas_strand_pump()
    max_iter_hyd = 4 if use_numba else 4
    p_diff, v_diff_abs = pipeflow_stanet_comparison(net, log_results,
                                                    max_iter_hyd=max_iter_hyd,
                                                    use_numba=use_numba,
                                                    calc_compression_power=False)
    assert np.all(p_diff < 1e-2)
    assert np.all(v_diff_abs < 3e-2)


# ---------- TEST AREA: t_cross ----------
# t-cross1_N
@pytest.mark.parametrize("use_numba", [True, False])
def test_case_tcross1_n(use_numba, log_results=False):
    """

    :param log_results:
    :type log_results:
    :return:
    :rtype:
    """
    net = nw.gas_tcross1(method="n")
    max_iter_hyd = 3 if use_numba else 3
    p_diff, v_diff_abs = pipeflow_stanet_comparison(net, log_results,
                                                    max_iter_hyd=max_iter_hyd,
                                                    use_numba=use_numba)
    assert np.all(p_diff < 1e-4)
    assert np.all(v_diff_abs < 1e-2)


# t-cross1_PC
@pytest.mark.parametrize("use_numba", [True, False])
def test_case_tcross1_pc(use_numba, log_results=False):
    """

    :param log_results:
    :type log_results:
    :return:
    :rtype:
    """
    net = nw.gas_tcross1(method="pc")
    max_iter_hyd = 3 if use_numba else 3
    p_diff, v_diff_abs = pipeflow_stanet_comparison(net, log_results,
                                                    max_iter_hyd=max_iter_hyd,
                                                    friction_model="colebrook",
                                                    use_numba=use_numba)
    assert np.all(p_diff < 1e-4)
    assert np.all(v_diff_abs < 1e-2)


# t-cross2_N
@pytest.mark.parametrize("use_numba", [True, False])
def test_case_tcross2_n(use_numba, log_results=False):
    """

    :param log_results:
    :type log_results:
    :return:
    :rtype:
    """
    net = nw.gas_tcross2(method="n")
    max_iter_hyd = 3 if use_numba else 3
    p_diff, v_diff_abs = pipeflow_stanet_comparison(net, log_results,
                                                    max_iter_hyd=max_iter_hyd,
                                                    use_numba=use_numba)
    assert np.all(p_diff < 1e-4)
    assert np.all(v_diff_abs < 1e-2)


# t-cross2_PC
@pytest.mark.parametrize("use_numba", [True, False])
def test_case_tcross2_pc(use_numba, log_results=False):
    """

    :param log_results:
    :type log_results:
    :return:
    :rtype:
    """
    net = nw.gas_tcross2(method="pc")
    max_iter_hyd = 3 if use_numba else 3
    p_diff, v_diff_abs = pipeflow_stanet_comparison(net, log_results,
                                                    max_iter_hyd=max_iter_hyd,
                                                    friction_model="colebrook",
                                                    use_numba=use_numba)
    assert np.all(p_diff < 1e-4)
    assert np.all(v_diff_abs < 1e-2)


# ---------- TEST AREA: two pressure junctions ----------
# H-net_N
@pytest.mark.parametrize("use_numba", [True, False])
def test_case_2eg_hnet_n(use_numba, log_results=False):
    """

    :param log_results:
    :type log_results:
    :return:
    :rtype:
    """
    net = nw.gas_2eg_hnet(method="n")
    max_iter_hyd = 5 if use_numba else 5
    p_diff, v_diff_abs = pipeflow_stanet_comparison(net, log_results,
                                                    max_iter_hyd=max_iter_hyd,
                                                    use_numba=use_numba)
    assert np.all(p_diff < 1e-4)
    assert np.all(v_diff_abs < 1e-2)


# H-net_PC
@pytest.mark.parametrize("use_numba", [True, False])
def test_case_2eg_hnet_pc(use_numba, log_results=False):
    """

    :param log_results:
    :type log_results:
    :return:
    :rtype:
    """
    net = nw.gas_2eg_hnet(method="pc")
    max_iter_hyd = 5 if use_numba else 5
    p_diff, v_diff_abs = pipeflow_stanet_comparison(net, log_results,
                                                    max_iter_hyd=max_iter_hyd,
                                                    friction_model="colebrook",
                                                    use_numba=use_numba)
    assert np.all(p_diff < 1e-4)
    assert np.all(v_diff_abs < 1e-2)

# # ----------------------------------------
# def test_case12(log_results=False):
#     # Geschwindigkeitsabweichungen sind relativ groß, was am unterschiedlichen p_mean liegt.
#     # Eine genaue Ursache für die Abweichungen konnte ich sonst nicht weiter ausmachen.
#     # Druck und Volumenstrom stimmen aber überein, von daher ist es ok, wenn wir die größere
#     # Abweichung akzeptieren, denke ich
#     p_diff, v_diff_abs = test_pipeflow_analytic_comparison(0)
#     assert np.all(p_diff < 0.01)
#     assert np.all(v_diff_abs < 0.4)
# # ----------------------------------------


if __name__ == "__main__":
    pytest.main(['test_gas_stanet.py'])
