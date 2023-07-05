# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import os
import pytest
import numpy as np
import pandapipes.networks.simple_water_networks as nw
from pandapipes.pipeflow import logger as pf_logger
from pandapipes.test.openmodelica_comparison.pipeflow_openmodelica_comparison \
    import pipeflow_openmodelica_comparison

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
pf_logger.setLevel(logging.WARNING)


# ---------- TEST AREA: combined networks ----------
# mixed_net
@pytest.mark.parametrize("use_numba", [True, False])
def test_case_combined_mixed_pc(use_numba, log_results=False):
    net = nw.water_combined_mixed()
    p_diff, v_diff_abs = pipeflow_openmodelica_comparison(net, log_results, use_numba=use_numba)
    assert np.all(p_diff < 0.01)
    assert np.all(v_diff_abs < 0.05)


@pytest.mark.parametrize("use_numba", [True, False])
def test_case_combined_mixed_sj(use_numba, log_results=False):
    net = nw.water_combined_mixed(method="swamee-jain")
    p_diff, v_diff_abs = pipeflow_openmodelica_comparison(net, log_results, use_numba=use_numba)
    assert np.all(p_diff < 0.01)
    assert np.all(v_diff_abs < 0.05)


# versatility
@pytest.mark.parametrize("use_numba", [True, False])
def test_case_combined_versatility_pc(use_numba, log_results=False):
    net = nw.water_combined_versatility()
    p_diff, v_diff_abs = pipeflow_openmodelica_comparison(net, log_results, use_numba=use_numba)
    assert np.all(p_diff < 0.04)  # only in one place the comparison for 0.01 is not correct
    assert np.all(v_diff_abs < 0.05)


@pytest.mark.parametrize("use_numba", [True, False])
def test_case_combined_versatility_sj(use_numba, log_results=False):
    net = nw.water_combined_versatility(method="swamee-jain")
    p_diff, v_diff_abs = pipeflow_openmodelica_comparison(net, log_results, use_numba=use_numba)
    assert np.all(p_diff < 0.04)  # only in one place the comparison for 0.01 is not correct
    assert np.all(v_diff_abs < 0.05)


# ---------- TEST AREA: meshed networks ----------
# delta
@pytest.mark.parametrize("use_numba", [True, False])
def test_case_meshed_delta_pc(use_numba, log_results=False):
    net = nw.water_meshed_delta()
    p_diff, v_diff_abs = pipeflow_openmodelica_comparison(net, log_results, use_numba=use_numba)
    assert np.all(p_diff < 0.01)
    assert np.all(v_diff_abs < 0.05)


@pytest.mark.parametrize("use_numba", [True, False])
def test_case_meshed_delta_sj(use_numba, log_results=False):
    net = nw.water_meshed_delta(method="swamee-jain")
    p_diff, v_diff_abs = pipeflow_openmodelica_comparison(net, log_results, use_numba=use_numba)
    assert np.all(p_diff < 0.01)
    assert np.all(v_diff_abs < 0.05)


# two_valves
@pytest.mark.parametrize("use_numba", [True, False])
def test_case_meshed_2valves_pc(use_numba, log_results=False):
    net = nw.water_meshed_2valves()
    p_diff, v_diff_abs = pipeflow_openmodelica_comparison(net, log_results, use_numba=use_numba)
    assert np.all(p_diff < 0.01)
    assert np.all(v_diff_abs < 0.05)


@pytest.mark.parametrize("use_numba", [True, False])
def test_case_meshed_2valves_sj(use_numba, log_results=False):
    net = nw.water_meshed_2valves(method="swamee-jain")
    p_diff, v_diff_abs = pipeflow_openmodelica_comparison(net, log_results, use_numba=use_numba)
    assert np.all(p_diff < 0.01)
    assert np.all(v_diff_abs < 0.05)


# pumps
@pytest.mark.parametrize("use_numba", [True, False])
def test_case_meshed_pumps_pc(use_numba, log_results=False):
    net = nw.water_meshed_pumps()
    p_diff, v_diff_abs = pipeflow_openmodelica_comparison(net, log_results, use_numba=use_numba)
    assert np.all(p_diff < 0.02)  # in two places the comparison for 0.01 is not correct
    assert np.all(v_diff_abs < 0.05)


@pytest.mark.parametrize("use_numba", [True, False])
def test_case_meshed_pumps_sj(use_numba, log_results=False):
    net = nw.water_meshed_pumps(method="swamee-jain")
    p_diff, v_diff_abs = pipeflow_openmodelica_comparison(net, log_results, use_numba=use_numba)
    assert np.all(p_diff < 0.02)  # in two places the comparison for 0.01 is not correct
    assert np.all(v_diff_abs < 0.05)


# heights
@pytest.mark.parametrize("use_numba", [True, False])
def test_case_meshed_heights_pc(use_numba, log_results=False):
    net = nw.water_meshed_heights()
    p_diff, v_diff_abs = pipeflow_openmodelica_comparison(net, log_results, use_numba=use_numba)
    assert np.all(p_diff < 0.01)
    assert np.all(v_diff_abs < 0.05)


@pytest.mark.parametrize("use_numba", [True, False])
def test_case_meshed_heights_sj(use_numba, log_results=False):
    net = nw.water_meshed_heights(method="swamee-jain")
    p_diff, v_diff_abs = pipeflow_openmodelica_comparison(net, log_results, use_numba=use_numba)
    assert np.all(p_diff < 0.01)
    assert np.all(v_diff_abs < 0.05)


# ---------- TEST AREA: one pipe ----------
# pipe_1
@pytest.mark.parametrize("use_numba", [True, False])
def test_case_one_pipe_1_pc(use_numba, log_results=False):
    net = nw.water_one_pipe1()
    p_diff, v_diff_abs = pipeflow_openmodelica_comparison(net, log_results, use_numba=use_numba)
    assert np.all(p_diff < 0.01)
    assert np.all(v_diff_abs < 0.05)


@pytest.mark.parametrize("use_numba", [True, False])
def test_case_one_pipe_1_sj(use_numba, log_results=False):
    net = nw.water_one_pipe1(method="swamee-jain")
    p_diff, v_diff_abs = pipeflow_openmodelica_comparison(net, log_results, use_numba=use_numba)
    assert np.all(p_diff < 0.01)
    assert np.all(v_diff_abs < 0.05)


# pipe_2
@pytest.mark.parametrize("use_numba", [True, False])
def test_case_one_pipe_2_pc(use_numba, log_results=False):
    net = nw.water_one_pipe2()
    p_diff, v_diff_abs = pipeflow_openmodelica_comparison(net, log_results, use_numba=use_numba)
    assert np.all(p_diff < 0.01)
    assert np.all(v_diff_abs < 0.05)


@pytest.mark.parametrize("use_numba", [True, False])
def test_case_one_pipe_2_sj(use_numba, log_results=False):
    net = nw.water_one_pipe2(method="swamee-jain")
    p_diff, v_diff_abs = pipeflow_openmodelica_comparison(net, log_results, use_numba=use_numba)
    assert np.all(p_diff < 0.01)
    assert np.all(v_diff_abs < 0.05)


# pipe_3
@pytest.mark.parametrize("use_numba", [True, False])
def test_case_one_pipe_3_pc(use_numba, log_results=False):
    net = nw.water_one_pipe3()
    p_diff, v_diff_abs = pipeflow_openmodelica_comparison(net, log_results, use_numba=use_numba)
    assert np.all(p_diff < 0.01)
    assert np.all(v_diff_abs < 0.05)


@pytest.mark.parametrize("use_numba", [True, False])
def test_case_one_pipe_3_sj(use_numba, log_results=False):
    net = nw.water_one_pipe3(method="swamee-jain")
    p_diff, v_diff_abs = pipeflow_openmodelica_comparison(net, log_results, use_numba=use_numba)
    assert np.all(p_diff < 0.01)
    assert np.all(v_diff_abs < 0.05)


# ---------- TEST AREA: strand net ----------
# cross_3ext
@pytest.mark.parametrize("use_numba", [True, False])
def test_case_strand_net_cross3ext_pc(use_numba, log_results=False):
    net = nw.water_strand_cross()
    p_diff, v_diff_abs = pipeflow_openmodelica_comparison(net, log_results, use_numba=use_numba)
    assert np.all(p_diff < 0.01)
    assert np.all(v_diff_abs < 0.05)


@pytest.mark.parametrize("use_numba", [True, False])
def test_case_strand_net_cross3ext_sj(use_numba, log_results=False):
    net = nw.water_strand_cross(method="swamee-jain")
    p_diff, v_diff_abs = pipeflow_openmodelica_comparison(net, log_results, use_numba=use_numba)
    assert np.all(p_diff < 0.01)
    assert np.all(v_diff_abs < 0.05)


# strand_net
@pytest.mark.parametrize("use_numba", [True, False])
def test_case_strand_net_pc(use_numba, log_results=False):
    net = nw.water_simple_strand_net()
    p_diff, v_diff_abs = pipeflow_openmodelica_comparison(net, log_results, use_numba=use_numba)
    assert np.all(p_diff < 0.01)
    assert np.all(v_diff_abs < 0.05)


@pytest.mark.parametrize("use_numba", [True, False])
def test_case_strand_net_sj(use_numba, log_results=False):
    net = nw.water_simple_strand_net(method="swamee-jain")
    p_diff, v_diff_abs = pipeflow_openmodelica_comparison(net, log_results, use_numba=use_numba)
    assert np.all(p_diff < 0.01)
    assert np.all(v_diff_abs < 0.05)


# two_pipes
@pytest.mark.parametrize("use_numba", [True, False])
def test_case_strand_net_2pipes_pc(use_numba, log_results=False):
    net = nw.water_strand_2pipes()
    p_diff, v_diff_abs = pipeflow_openmodelica_comparison(net, log_results, use_numba=use_numba)
    assert np.all(p_diff < 0.01)
    assert np.all(v_diff_abs < 0.05)


@pytest.mark.parametrize("use_numba", [True, False])
def test_case_strand_net_2pipes_sj(use_numba, log_results=False):
    net = nw.water_strand_2pipes(method="swamee-jain")
    p_diff, v_diff_abs = pipeflow_openmodelica_comparison(net, log_results, use_numba=use_numba)
    assert np.all(p_diff < 0.01)
    assert np.all(v_diff_abs < 0.05)


# two_pumps
@pytest.mark.parametrize("use_numba", [True, False])
def test_case_strand_net_2pumps_pc(use_numba, log_results=False):
    net = nw.water_strand_net_2pumps()
    p_diff, v_diff_abs = pipeflow_openmodelica_comparison(net, log_results, use_numba=use_numba)
    assert np.all(p_diff < 0.01)
    assert np.all(v_diff_abs < 0.05)


@pytest.mark.parametrize("use_numba", [True, False])
def test_case_strand_net_2pumps_sj(use_numba, log_results=False):
    net = nw.water_strand_net_2pumps(method="swamee-jain")
    p_diff, v_diff_abs = pipeflow_openmodelica_comparison(net, log_results, use_numba=use_numba)
    assert np.all(p_diff < 0.01)
    assert np.all(v_diff_abs < 0.05)


# ---------- TEST AREA: t_cross ----------
# t_cross
@pytest.mark.parametrize("use_numba", [True, False])
def test_case_tcross_pc(use_numba, log_results=False):
    net = nw.water_tcross()
    p_diff, v_diff_abs = pipeflow_openmodelica_comparison(net, log_results, use_numba=use_numba)
    assert np.all(p_diff < 0.01)
    assert np.all(v_diff_abs < 0.05)


@pytest.mark.parametrize("use_numba", [True, False])
def test_case_tcross_sj(use_numba, log_results=False):
    net = nw.water_tcross(method="swamee-jain")
    p_diff, v_diff_abs = pipeflow_openmodelica_comparison(net, log_results, use_numba=use_numba)
    assert np.all(p_diff < 0.02)  # 2 values are greater than 0.01
    assert np.all(v_diff_abs < 0.05)


# valves
@pytest.mark.parametrize("use_numba", [True, False])
def test_case_tcross_valves_pc(use_numba, log_results=False):
    net = nw.water_tcross_valves()
    p_diff, v_diff_abs = pipeflow_openmodelica_comparison(net, log_results, use_numba=use_numba)
    assert np.all(p_diff < 0.4)  # only in one place the comparison for 0.01 is not correct
    assert np.all(v_diff_abs < 0.05)


@pytest.mark.parametrize("use_numba", [True, False])
def test_case_tcross_valves_sj(use_numba, log_results=False):
    net = nw.water_tcross_valves(method="swamee-jain")
    p_diff, v_diff_abs = pipeflow_openmodelica_comparison(net, log_results, use_numba=use_numba)
    assert np.all(p_diff < 0.4)  # only in one place the comparison for 0.01 is not correct
    assert np.all(v_diff_abs < 0.05)


# ---------- TEST AREA: two pressure junctions ----------
# two_pipes
@pytest.mark.parametrize("use_numba", [True, False])
def test_case_2eg_two_pipes_pc(use_numba, log_results=False):
    net = nw.water_2eg_two_pipes()
    p_diff, v_diff_abs = pipeflow_openmodelica_comparison(net, log_results, use_numba=use_numba)
    assert np.all(p_diff < 0.01)
    assert np.all(v_diff_abs < 0.05)


@pytest.mark.parametrize("use_numba", [True, False])
def test_case_2eg_two_pipes_sj(use_numba, log_results=False):
    net = nw.water_2eg_two_pipes(method="swamee-jain")
    p_diff, v_diff_abs = pipeflow_openmodelica_comparison(net, log_results, use_numba=use_numba)
    assert np.all(p_diff < 0.01)
    assert np.all(v_diff_abs < 0.05)


if __name__ == "__main__":
    pytest.main([os.path.join(os.path.dirname(__file__), "test_water_openmodelica.py")])
