# Copyright (c) 2020-2025 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import itertools

import pandapipes
import pandapipes.networks as nw
import pytest


@pytest.mark.parametrize("use_numba, friction_model",
                         itertools.product([True, False],
                                           ["nikuradse", "colebrook", "hofer", "swamee-jain"]))
def test_friction_model_hydraulic_only(use_numba, friction_model):
    """Runs a hydraulic pipeflow calculation for all available friction models. Fails if pipeflow does not converge.

    :return: None
    :rtype: None
    """
    net = nw.schutterwald()
    pandapipes.pipeflow(net, use_numba=use_numba, friction_model=friction_model, mode="hydraulics")

    assert net.converged
