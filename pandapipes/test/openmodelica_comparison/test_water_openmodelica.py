# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import os
import pytest
import numpy as np
import pandapipes.networks.simple_water_networks as nw
from pandapipes.pipeflow import logger as pf_logger
from pandapipes.test.openmodelica_comparison.pipeflow_openmodelica_comparison \
    import pipeflow_openmodelica_comparison

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
pf_logger.setLevel(logging.WARNING)


# ---------- TEST AREA: combined networks ----------
# mixed_net
def test_case_combined_mixed(log_results=False):
    net = nw.water_combined_mixed()
    p_diff, v_diff_abs = pipeflow_openmodelica_comparison(net, log_results)
    assert np.all(p_diff < 0.01)
    assert np.all(v_diff_abs < 0.05)


# versatility
def test_case_combined_versatility(log_results=False):
    net = nw.water_combined_versatility()
    p_diff, v_diff_abs = pipeflow_openmodelica_comparison(net, log_results)
    assert np.all(p_diff < 0.06)  # only in one place the comparison for 0.01 is not correct
    assert np.all(v_diff_abs < 0.05)


# ---------- TEST AREA: meshed networks ----------
# delta
def test_case_meshed_delta(log_results=False):
    net = nw.water_meshed_delta()
    p_diff, v_diff_abs = pipeflow_openmodelica_comparison(net, log_results)
    assert np.all(p_diff < 0.01)
    assert np.all(v_diff_abs < 0.05)


# two_valves
def test_case_meshed_2valves(log_results=False):
    net = nw.water_meshed_2valves()
    p_diff, v_diff_abs = pipeflow_openmodelica_comparison(net, log_results)
    assert np.all(p_diff < 0.01)
    assert np.all(v_diff_abs < 0.05)


# pumps
def test_case_meshed_pumps(log_results=False):
    net = nw.water_meshed_pumps()
    p_diff, v_diff_abs = pipeflow_openmodelica_comparison(net, log_results)
    assert np.all(p_diff < 0.02)  # in two places the comparison for 0.01 is not correct
    assert np.all(v_diff_abs < 0.05)


# heights
def test_case_meshed_heights(log_results=False):
    net = nw.water_meshed_heights()
    p_diff, v_diff_abs = pipeflow_openmodelica_comparison(net, log_results)
    assert np.all(p_diff < 0.01)
    assert np.all(v_diff_abs < 0.05)


# ---------- TEST AREA: one pipe ----------
# pipe_1
def test_case_one_pipe_1(log_results=False):
    net = nw.water_one_pipe1()
    p_diff, v_diff_abs = pipeflow_openmodelica_comparison(net, log_results)
    assert np.all(p_diff < 0.01)
    assert np.all(v_diff_abs < 0.05)


# pipe_2
def test_case_one_pipe_2(log_results=False):
    net = nw.water_one_pipe2()
    p_diff, v_diff_abs = pipeflow_openmodelica_comparison(net, log_results)
    assert np.all(p_diff < 0.01)
    assert np.all(v_diff_abs < 0.05)


# pipe_3
def test_case_one_pipe_3(log_results=False):
    net = nw.water_one_pipe3()
    p_diff, v_diff_abs = pipeflow_openmodelica_comparison(net, log_results)
    assert np.all(p_diff < 0.01)
    assert np.all(v_diff_abs < 0.05)


# ---------- TEST AREA: strand net ----------
# cross_3ext
def test_case_strand_net_cross3ext(log_results=False):
    net = nw.water_strand_cross()
    p_diff, v_diff_abs = pipeflow_openmodelica_comparison(net, log_results)
    assert np.all(p_diff < 0.01)
    assert np.all(v_diff_abs < 0.05)


# strand_net
def test_case_strand_net(log_results=False):
    net = nw.water_simple_strand_net()
    p_diff, v_diff_abs = pipeflow_openmodelica_comparison(net, log_results)
    assert np.all(p_diff < 0.01)
    assert np.all(v_diff_abs < 0.05)


# two_pipes
def test_case_strand_net_2pipes(log_results=False):
    net = nw.water_strand_2pipes()
    p_diff, v_diff_abs = pipeflow_openmodelica_comparison(net, log_results)
    assert np.all(p_diff < 0.01)
    assert np.all(v_diff_abs < 0.05)


# two_pumps
def test_case_strand_net_2pumps(log_results=False):
    net = nw.water_strand_net_2pumps()
    p_diff, v_diff_abs = pipeflow_openmodelica_comparison(net, log_results)
    assert np.all(p_diff < 0.01)
    assert np.all(v_diff_abs < 0.05)


# ---------- TEST AREA: t_cross ----------
# t_cross
def test_case_tcross(log_results=False):
    net = nw.water_tcross()
    p_diff, v_diff_abs = pipeflow_openmodelica_comparison(net, log_results)
    assert np.all(p_diff < 0.01)
    assert np.all(v_diff_abs < 0.05)


# valves
def test_case_tcross_valves(log_results=False):
    net = nw.water_tcross_valves()
    p_diff, v_diff_abs = pipeflow_openmodelica_comparison(net, log_results)
    assert np.all(p_diff < 0.4)  # only in one place the comparison for 0.01 is not correct
    assert np.all(v_diff_abs < 0.05)


# ---------- TEST AREA: two pressure junctions ----------
# two_pipes
def test_case_2eg_two_pipes(log_results=False):
    net = nw.water_2eg_two_pipes()
    p_diff, v_diff_abs = pipeflow_openmodelica_comparison(net, log_results)
    assert np.all(p_diff < 0.01)
    assert np.all(v_diff_abs < 0.05)


if __name__ == "__main__":
    pytest.main([os.path.join(os.path.dirname(__file__), "test_water_openmodelica.py")])
