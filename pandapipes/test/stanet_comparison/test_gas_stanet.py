# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
import pandapipes.networks.simple_gas_networks as nw
import pytest
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
    p_diff, v_diff_abs = pipeflow_stanet_comparison(net, log_results, use_numba=use_numba)
    assert np.all(p_diff < 0.002)
    assert np.all(v_diff_abs < 0.03)


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
    p_diff, v_diff_abs = pipeflow_stanet_comparison(net, log_results, friction_model="colebrook",
                                                    use_numba=use_numba)
    assert np.all(p_diff < 0.002)
    assert np.all(v_diff_abs < 0.03)


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
    p_diff, v_diff_abs = pipeflow_stanet_comparison(net, log_results, use_numba=use_numba)
    assert np.all(p_diff < 0.002)
    assert np.all(v_diff_abs < 0.03)


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
    p_diff, v_diff_abs = pipeflow_stanet_comparison(net, log_results, friction_model="colebrook",
                                                    use_numba=use_numba)
    assert np.all(p_diff < 0.002)
    assert np.all(v_diff_abs < 0.03)


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
    p_diff, v_diff_abs = pipeflow_stanet_comparison(net, log_results, friction_model="colebrook",
                                                    use_numba=use_numba)
    assert np.all(p_diff < 0.002)
    assert np.all(v_diff_abs < 0.03)


# pumps_N
@pytest.mark.parametrize("use_numba", [True, False])
def test_case_meshed_pumps(use_numba, log_results=False):
    """

    :param log_results:
    :type log_results:
    :return:
    :rtype:
    """
    net = nw.gas_meshed_pumps()
    p_diff, v_diff_abs = pipeflow_stanet_comparison(net, log_results, use_numba=use_numba,
                                                    calc_compression_power=False)
    assert np.all(p_diff < 0.002)
    assert np.all(v_diff_abs < 0.03)


# two_valves_N
@pytest.mark.parametrize("use_numba", [True, False])
def test_case_meshed_2valves_n(use_numba, log_results=False):
    net = nw.gas_meshed_two_valves(method="n")
    p_diff, v_diff_abs = pipeflow_stanet_comparison(net, log_results, use_numba=use_numba)
    assert np.all(p_diff < 0.001)
    assert np.all(v_diff_abs < 0.001)


# two_valves_PC
@pytest.mark.parametrize("use_numba", [True, False])
def test_case_meshed_2valves_pc(use_numba, log_results=False):
    net = nw.gas_meshed_two_valves(method="pc")
    p_diff, v_diff_abs = pipeflow_stanet_comparison(net, log_results, friction_model="colebrook",
                                                    use_numba=use_numba)
    assert np.all(p_diff < 0.001)
    assert np.all(v_diff_abs < 0.001)


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
    p_diff, v_diff_abs = pipeflow_stanet_comparison(net, log_results, use_numba=use_numba)
    assert np.all(p_diff < 0.002)
    assert np.all(v_diff_abs < 0.03)


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
    p_diff, v_diff_abs = pipeflow_stanet_comparison(net, log_results, friction_model="colebrook",
                                                    use_numba=use_numba)
    assert np.all(p_diff < 0.002)
    assert np.all(v_diff_abs < 0.03)


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
    p_diff, v_diff_abs = pipeflow_stanet_comparison(net, log_results, use_numba=use_numba)
    assert np.all(p_diff < 0.002)
    assert np.all(v_diff_abs < 0.03)


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
    p_diff, v_diff_abs = pipeflow_stanet_comparison(net, log_results, friction_model="colebrook",
                                                    use_numba=use_numba)
    assert np.all(p_diff < 0.002)
    assert np.all(v_diff_abs < 0.03)


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
    p_diff, v_diff_abs = pipeflow_stanet_comparison(net, log_results, use_numba=use_numba)
    assert np.all(p_diff < 0.002)
    assert np.all(v_diff_abs < 0.03)


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
    p_diff, v_diff_abs = pipeflow_stanet_comparison(net, log_results, friction_model="colebrook",
                                                    use_numba=use_numba)
    assert np.all(p_diff < 0.002)
    assert np.all(v_diff_abs < 0.03)


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
    p_diff, v_diff_abs = pipeflow_stanet_comparison(net, log_results, use_numba=use_numba,
                                                    calc_compression_power=False)
    assert np.all(p_diff < 0.01)
    assert np.all(v_diff_abs < 0.03)


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
    p_diff, v_diff_abs = pipeflow_stanet_comparison(net, log_results, use_numba=use_numba)
    assert np.all(p_diff < 0.002)
    assert np.all(v_diff_abs < 0.03)


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
    p_diff, v_diff_abs = pipeflow_stanet_comparison(net, log_results, friction_model="colebrook",
                                                    use_numba=use_numba)
    assert np.all(p_diff < 0.002)
    assert np.all(v_diff_abs < 0.03)


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
    p_diff, v_diff_abs = pipeflow_stanet_comparison(net, log_results, use_numba=use_numba)
    assert np.all(p_diff < 0.002)
    assert np.all(v_diff_abs < 0.03)


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
    p_diff, v_diff_abs = pipeflow_stanet_comparison(net, log_results, friction_model="colebrook",
                                                    use_numba=use_numba)
    assert np.all(p_diff < 0.002)
    assert np.all(v_diff_abs < 0.03)


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
    p_diff, v_diff_abs = pipeflow_stanet_comparison(net, log_results, use_numba=use_numba)
    assert np.all(p_diff < 0.002)
    assert np.all(v_diff_abs < 0.03)


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
    p_diff, v_diff_abs = pipeflow_stanet_comparison(net, log_results, friction_model="colebrook",
                                                    use_numba=use_numba)
    assert np.all(p_diff < 0.002)
    assert np.all(v_diff_abs < 0.03)


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
