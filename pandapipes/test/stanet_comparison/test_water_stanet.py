# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
import pandapipes.networks.simple_water_networks as nw
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
# district_N
@pytest.mark.parametrize("use_numba", [True, False])
def test_case_district_grid_n(use_numba, log_results=False):
    """

    :param log_results:
    :type log_results:
    :return:
    :rtype:
    """
    net = nw.water_district_grid(method="n")
    p_diff, v_diff_abs = pipeflow_stanet_comparison(net, log_results, use_numba=use_numba)
    assert np.all(p_diff < 0.002)
    assert np.all(v_diff_abs < 0.03)


# district_PC
@pytest.mark.parametrize("use_numba", [True, False])
def test_case_district_grid_pc(use_numba, log_results=False):
    """

    :param log_results:
    :type log_results:
    :return:
    :rtype:
    """
    net = nw.water_district_grid(method="pc")
    p_diff, v_diff_abs = pipeflow_stanet_comparison(net, log_results, use_numba=use_numba,
                                                    friction_model="colebrook")
    assert np.all(p_diff < 0.03)
    assert np.all(v_diff_abs < 0.03)


# ---------- TEST AREA: meshed networks ----------
# pumps_N
@pytest.mark.parametrize("use_numba", [True, False])
def test_case_pumps_n(use_numba, log_results=False):
    """

    :param log_results:
    :type log_results:
    :return:
    :rtype:
    """
    net = nw.water_meshed_pumps(results_from="stanet")
    p_diff, v_diff_abs = pipeflow_stanet_comparison(net, log_results, use_numba=use_numba)
    assert np.all(p_diff < 0.002)
    assert np.all(v_diff_abs < 0.03)


# delta_N
@pytest.mark.parametrize("use_numba", [True, False])
def test_case_delta_n(use_numba, log_results=False):
    """

    :param log_results:
    :type log_results:
    :return:
    :rtype:
    """
    net = nw.water_meshed_delta(results_from="stanet")
    p_diff, v_diff_abs = pipeflow_stanet_comparison(net, log_results, use_numba=use_numba)
    assert np.all(p_diff < 0.002)
    assert np.all(v_diff_abs < 0.03)


# two_valves_N
@pytest.mark.parametrize("use_numba", [True, False])
def test_case_meshed_2valves_n(use_numba, log_results=False):
    net = nw.water_meshed_2valves(method="n", results_from="stanet")
    p_diff, v_diff_abs = pipeflow_stanet_comparison(net, log_results, use_numba=use_numba)
    assert np.all(p_diff < 0.001)
    assert np.all(v_diff_abs < 0.001)


# two_valves_PC
@pytest.mark.parametrize("use_numba", [True, False])
def test_case_meshed_2valves_pc(use_numba, log_results=False):
    net = nw.water_meshed_2valves(method="pc", results_from="stanet")
    p_diff, v_diff_abs = pipeflow_stanet_comparison(net, log_results, use_numba=use_numba,
                                                    friction_model="colebrook")
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
    net = nw.water_one_pipe1(method="n", results_from="stanet")
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
    net = nw.water_one_pipe1(method="pc", results_from="stanet")
    p_diff, v_diff_abs = pipeflow_stanet_comparison(net, log_results, use_numba=use_numba,
                                                    friction_model="colebrook")
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
    net = nw.water_one_pipe2(method="n", results_from="stanet")
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
    net = nw.water_one_pipe2(method="pc", results_from="stanet")
    p_diff, v_diff_abs = pipeflow_stanet_comparison(net, log_results, use_numba=use_numba,
                                                    friction_model="colebrook")
    assert np.all(p_diff < 0.002)
    assert np.all(v_diff_abs < 0.03)


# pipe_3_N
@pytest.mark.parametrize("use_numba", [True, False])
def test_case_one_pipe3_n(use_numba, log_results=False):
    """

    :param log_results:
    :type log_results:
    :return:
    :rtype:
    """
    net = nw.water_one_pipe3(method="n", results_from="stanet")
    p_diff, v_diff_abs = pipeflow_stanet_comparison(net, log_results, use_numba=use_numba)
    assert np.all(p_diff < 0.002)
    assert np.all(v_diff_abs < 0.03)


# pipe_3_PC
@pytest.mark.parametrize("use_numba", [True, False])
def test_case_one_pipe3_pc(use_numba, log_results=False):
    """

    :param log_results:
    :type log_results:
    :return:
    :rtype:
    """
    net = nw.water_one_pipe3(method="pc", results_from="stanet")
    p_diff, v_diff_abs = pipeflow_stanet_comparison(net, log_results, use_numba=use_numba,
                                                    friction_model="colebrook")
    assert np.all(p_diff < 0.002)
    assert np.all(v_diff_abs < 0.03)


# ---------- TEST AREA: strand net ----------
# strand_net_N
@pytest.mark.parametrize("use_numba", [True, False])
def test_case_simple_strand_net_n(use_numba, log_results=False):
    """

    :param log_results:
    :type log_results:
    :return:
    :rtype:
    """
    net = nw.water_simple_strand_net(method="n", results_from="stanet")
    p_diff, v_diff_abs = pipeflow_stanet_comparison(net, log_results, use_numba=use_numba)
    assert np.all(p_diff < 0.002)
    assert np.all(v_diff_abs < 0.03)


# strand_net_PC
@pytest.mark.parametrize("use_numba", [True, False])
def test_case_simple_strand_net_pc(use_numba, log_results=False):
    """

    :param log_results:
    :type log_results:
    :return:
    :rtype:
    """
    net = nw.water_simple_strand_net(method="pc", results_from="stanet")
    p_diff, v_diff_abs = pipeflow_stanet_comparison(net, log_results, use_numba=use_numba,
                                                    friction_model="colebrook")
    assert np.all(p_diff < 0.01)
    assert np.all(v_diff_abs < 0.03)


# two_pipes_N
@pytest.mark.parametrize("use_numba", [True, False])
def test_case_two_pipes_n(use_numba, log_results=False):
    """

    :param log_results:
    :type log_results:
    :return:
    :rtype:
    """
    net = nw.water_strand_2pipes(method="n", results_from="stanet")
    p_diff, v_diff_abs = pipeflow_stanet_comparison(net, log_results, use_numba=use_numba)
    assert np.all(p_diff < 0.002)
    assert np.all(v_diff_abs < 0.03)


# two_pipes_PC
@pytest.mark.parametrize("use_numba", [True, False])
def test_case_two_pipes_pc(use_numba, log_results=False):
    """

    :param log_results:
    :type log_results:
    :return:
    :rtype:
    """
    net = nw.water_strand_2pipes(method="pc", results_from="stanet")
    p_diff, v_diff_abs = pipeflow_stanet_comparison(net, log_results, use_numba=use_numba,
                                                    friction_model="colebrook")
    assert np.all(p_diff < 0.002)
    assert np.all(v_diff_abs < 0.03)


# cross_PC
@pytest.mark.parametrize("use_numba", [True, False])
def test_case_cross_pc(use_numba, log_results=False):
    """

    :param log_results:
    :type log_results:
    :return:
    :rtype:
    """
    net = nw.water_strand_cross(results_from="stanet")
    p_diff, v_diff_abs = pipeflow_stanet_comparison(net, log_results, use_numba=use_numba,
                                                    friction_model="colebrook")
    assert np.all(p_diff < 0.002)
    assert np.all(v_diff_abs < 0.03)


# pump_N
@pytest.mark.parametrize("use_numba", [True, False])
def test_case_pump_n(use_numba, log_results=False):
    """

    :param log_results:
    :type log_results:
    :return:
    :rtype:
    """
    net = nw.water_strand_pump()
    p_diff, v_diff_abs = pipeflow_stanet_comparison(net, log_results, use_numba=use_numba)
    assert np.all(p_diff < 0.002)
    assert np.all(v_diff_abs < 0.03)


# ---------- TEST AREA: t_cross ----------
# t-cross_N
@pytest.mark.parametrize("use_numba", [True, False])
def test_case_tcross_n(use_numba, log_results=False):
    """

    :param log_results:
    :type log_results:
    :return:
    :rtype:
    """
    net = nw.water_tcross(method="n", results_from="stanet")
    p_diff, v_diff_abs = pipeflow_stanet_comparison(net, log_results, use_numba=use_numba)
    assert np.all(p_diff < 0.002)
    assert np.all(v_diff_abs < 0.03)


# t-cross_PC
@pytest.mark.parametrize("use_numba", [True, False])
def test_case_tcross_pc(use_numba, log_results=False):
    """

    :param log_results:
    :type log_results:
    :return:
    :rtype:
    """
    net = nw.water_tcross(method="pc", results_from="stanet")
    p_diff, v_diff_abs = pipeflow_stanet_comparison(net, log_results, use_numba=use_numba,
                                                    friction_model="colebrook")
    assert np.all(p_diff < 0.002)
    assert np.all(v_diff_abs < 0.03)


# ---------- TEST AREA: two pressure junctions ----------
# two_pipes_N
@pytest.mark.parametrize("use_numba", [True, False])
def test_case_2eg_two_pipes_n(use_numba, log_results=False):
    """

    :param log_results:
    :type log_results:
    :return:
    :rtype:
    """
    net = nw.water_2eg_two_pipes(method="n", results_from="stanet")
    p_diff, v_diff_abs = pipeflow_stanet_comparison(net, log_results, use_numba=use_numba)
    assert np.all(p_diff < 0.002)
    assert np.all(v_diff_abs < 0.03)


# two_pipes_PC
@pytest.mark.parametrize("use_numba", [True, False])
def test_case_2eg_two_pipes_pc(use_numba, log_results=False):
    """

    :param log_results:
    :type log_results:
    :return:
    :rtype:
    """
    net = nw.water_2eg_two_pipes(method="pc", results_from="stanet")
    p_diff, v_diff_abs = pipeflow_stanet_comparison(net, log_results, use_numba=use_numba,
                                                    friction_model="colebrook")
    assert np.all(p_diff < 0.002)
    assert np.all(v_diff_abs < 0.03)


if __name__ == "__main__":
    pytest.main([r'pandapipes/test/stanet_comparison/test_water_stanet.py'])
