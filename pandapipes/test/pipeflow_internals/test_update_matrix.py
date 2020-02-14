# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
import pandapipes.networks.simple_gas_networks as nw
import pytest
from pandapipes.pipeflow import logger as pf_logger
from pandapipes.test.stanet_comparison.pipeflow_stanet_comparison import pipeflow_stanet_comparison

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)
pf_logger.setLevel(logging.WARNING)


def test_update(log_results=False):
    """

    :param log_results:
    :type log_results:
    :return:
    :rtype:
    """
    # before: gas_case3.json
    net = nw.gas_one_pipe1()
    p_diff, v_diff_abs = pipeflow_stanet_comparison(net, log_results,
                                                    only_update_hydraulic_matrix=True)
    assert np.all(p_diff < 0.01)
    assert np.all(v_diff_abs < 0.05)


if __name__ == "__main__":
    pytest.main([r'pandapipes/test/pipeflow_internals/test_update_matrix.py'])
