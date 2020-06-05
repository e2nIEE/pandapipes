# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import os
import pytest
import numpy as np
import pandapipes.networks.simple_heat_transfer_networks as nw
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


def test_case_delta(log_results=False):
    net = nw.heat_transfer_delta()
    p_diff, v_diff_abs, T_diff_mean = pipeflow_openmodelica_comparison(net, log_results, mode="all")
    assert np.all(T_diff_mean < 0.007)
    assert np.all(p_diff < 0.05)
    assert np.all(v_diff_abs < 0.05)

def test_case_delta_2sinks(log_results=False):
    net = nw.heat_transfer_delta_2sinks()
    p_diff, v_diff_abs, T_diff_mean = pipeflow_openmodelica_comparison(net, log_results, mode="all")
    assert np.all(T_diff_mean < 0.007)
    assert np.all(p_diff < 0.02)
    assert np.all(v_diff_abs < 0.05)

def test_case_heights(log_results=False):
    net = nw.heat_transfer_heights()
    p_diff, v_diff_abs, T_diff_mean = pipeflow_openmodelica_comparison(net, log_results, mode="all")
    assert np.all(T_diff_mean < 0.0065)
    assert np.all(p_diff < 0.02)
    assert np.all(v_diff_abs < 0.05)

def test_case_one_pipe(log_results=False):
    net = nw.heat_transfer_one_pipe()
    p_diff, v_diff_abs, T_diff_mean = pipeflow_openmodelica_comparison(net, log_results, mode="all")
    assert np.all(T_diff_mean < 0.004)
    assert np.all(p_diff < 0.01)
    assert np.all(v_diff_abs < 0.05)

def test_case_one_source(log_results=False):
    net = nw.heat_transfer_one_source()
    p_diff, v_diff_abs, T_diff_mean = pipeflow_openmodelica_comparison(net, log_results, mode="all")
    assert np.all(T_diff_mean < 0.04)
    assert np.all(p_diff < 0.01)
    assert np.all(v_diff_abs < 0.05)

def test_case_section_variation(log_results=False):
    net = nw.heat_transfer_section_variation()
    p_diff, v_diff_abs, T_diff_mean = pipeflow_openmodelica_comparison(net, log_results, mode="all")
    assert np.all(T_diff_mean < 0.03) # all values of T_diff_mean are zero except one with about 0.025
    assert np.all(p_diff < 0.022)
    assert np.all(v_diff_abs < 0.05)

def test_case_t_cross(log_results=False):
    net = nw.heat_transfer_t_cross()
    p_diff, v_diff_abs, T_diff_mean = pipeflow_openmodelica_comparison(net, log_results, mode="all")
    assert np.all(T_diff_mean < 0.007)
    assert np.all(p_diff < 0.01)
    assert np.all(v_diff_abs < 0.05)

def test_case_two_pipes(log_results=False):
    net = nw.heat_transfer_two_pipes()
    p_diff, v_diff_abs, T_diff_mean = pipeflow_openmodelica_comparison(net, log_results, mode="all")
    assert np.all(T_diff_mean < 0.004)
    assert np.all(p_diff < 0.01)
    assert np.all(v_diff_abs < 0.05)

if __name__ == "__main__":
    pytest.main([os.path.join(os.path.dirname(__file__), "test_heat_transfer_openmodelica.py")])