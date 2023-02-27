# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
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
pf_logger.setLevel(logging.WARNING)


@pytest.mark.parametrize("use_numba", [True, False])
def test_update(use_numba, log_results=False):
    """

    :param log_results:
    :type log_results:
    :return:
    :rtype:
    """
    # before: gas_case3.json
    net = nw.gas_one_pipe1()
    p_diff, v_diff_abs = pipeflow_stanet_comparison(net, log_results, use_numba=use_numba,
                                                    only_update_hydraulic_matrix=True)
    assert np.all(p_diff < 0.01)
    assert np.all(v_diff_abs < 0.05)


if __name__ == "__main__":
    pytest.main([r'pandapipes/test/pipeflow_internals/test_update_matrix.py'])
