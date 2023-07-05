# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import os
import pytest
import numpy as np
import pandapipes.networks.simple_heat_transfer_networks as nw
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


@pytest.mark.parametrize("use_numba", [True, False])
def test_case_delta(use_numba, log_results=False):
    net = nw.heat_transfer_delta()
    p_diff, v_diff_abs, temp_diff_mean = pipeflow_openmodelica_comparison(
        net, log_results, mode="all", use_numba=use_numba)
    assert np.all(temp_diff_mean < 0.007)
    assert np.all(p_diff < 0.05)
    assert np.all(v_diff_abs < 0.05)


@pytest.mark.parametrize("use_numba", [True, False])
def test_case_delta_2sinks(use_numba, log_results=False):
    net = nw.heat_transfer_delta_2sinks()
    p_diff, v_diff_abs, temp_diff_mean = pipeflow_openmodelica_comparison(
        net, log_results, mode="all", use_numba=use_numba)
    assert np.all(temp_diff_mean < 0.007)
    assert np.all(p_diff < 0.02)
    assert np.all(v_diff_abs < 0.05)


@pytest.mark.parametrize("use_numba", [True, False])
def test_case_heights(use_numba, log_results=False):
    net = nw.heat_transfer_heights()
    p_diff, v_diff_abs, temp_diff_mean = pipeflow_openmodelica_comparison(
        net, log_results, mode="all", use_numba=use_numba)
    assert np.all(temp_diff_mean < 0.0065)
    assert np.all(p_diff < 0.02)
    assert np.all(v_diff_abs < 0.05)


@pytest.mark.parametrize("use_numba", [True, False])
def test_case_one_pipe(use_numba, log_results=False):
    net = nw.heat_transfer_one_pipe()
    p_diff, v_diff_abs, temp_diff_mean = pipeflow_openmodelica_comparison(
        net, log_results, mode="all", use_numba=use_numba)
    assert np.all(temp_diff_mean < 0.004)
    assert np.all(p_diff < 0.01)
    assert np.all(v_diff_abs < 0.05)


@pytest.mark.parametrize("use_numba", [True, False])
def test_case_one_source(use_numba, log_results=False):
    net = nw.heat_transfer_one_source()
    p_diff, v_diff_abs, temp_diff_mean = pipeflow_openmodelica_comparison(
        net, log_results, mode="all", use_numba=use_numba)
    assert np.all(temp_diff_mean < 0.04)
    assert np.all(p_diff < 0.01)
    assert np.all(v_diff_abs < 0.05)


@pytest.mark.parametrize("use_numba", [True, False])
def test_case_section_variation(use_numba, log_results=False):
    net = nw.heat_transfer_section_variation()
    p_diff, v_diff_abs, temp_diff_mean = pipeflow_openmodelica_comparison(
        net, log_results, mode="all", use_numba=use_numba)
    # all values of temp_diff_mean are zero except one with about 0.025
    assert np.all(temp_diff_mean < 0.03)
    assert np.all(p_diff < 0.022)
    assert np.all(v_diff_abs < 0.05)


@pytest.mark.parametrize("use_numba", [True, False])
def test_case_t_cross(use_numba, log_results=False):
    net = nw.heat_transfer_t_cross()
    p_diff, v_diff_abs, temp_diff_mean = pipeflow_openmodelica_comparison(
        net, log_results, mode="all", use_numba=use_numba)
    assert np.all(temp_diff_mean < 0.007)
    assert np.all(p_diff < 0.01)
    assert np.all(v_diff_abs < 0.05)


@pytest.mark.parametrize("use_numba", [True, False])
def test_case_two_pipes(use_numba, log_results=False):
    net = nw.heat_transfer_two_pipes()
    p_diff, v_diff_abs, temp_diff_mean = pipeflow_openmodelica_comparison(
        net, log_results, mode="all", use_numba=use_numba)
    assert np.all(temp_diff_mean < 0.004)
    assert np.all(p_diff < 0.01)
    assert np.all(v_diff_abs < 0.05)


if __name__ == "__main__":
    pytest.main([os.path.join(os.path.dirname(__file__), "test_heat_transfer_openmodelica.py")])
