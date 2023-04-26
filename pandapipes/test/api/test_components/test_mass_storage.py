# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
import pytest

import pandapipes


@pytest.mark.parametrize("use_numba", [True, False])
def test_mass_storage(use_numba):

    net = pandapipes.create_empty_network("net", add_stdtypes=True, fluid="water")

    j = pandapipes.create_junctions(net, 3, pn_bar=2, tfluid_k=283.15)
    j1, j2, j3 = j

    pandapipes.create_pipe_from_parameters(net, j1, j2, length_km=1, diameter_m=0.5)
    pandapipes.create_pipe_from_parameters(net, j2, j3, length_km=1, diameter_m=0.5)
    pandapipes.create_ext_grid(net, j1, 2, 283.15, type="p")
    pandapipes.create_mass_storage(net, j2, 0.1)
    pandapipes.create_mass_storage(net, j3, -0.2)

    pandapipes.pipeflow(net, use_numba=use_numba)
    assert np.isclose(net.res_ext_grid["mdot_kg_per_s"], 0.1)


if __name__ == '__main__':
    n = pytest.main(["test_mass_storage.py"])