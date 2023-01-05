# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

from pandapipes import pp_dir
import pandapipes as pp
import numpy as np
import os
import pytest
from pandapipes.io import convert_format
from itertools import product
from packaging import version

folder = os.path.join(pp_dir, "test", "api", "old_versions")
found_versions = sorted(set([file.split("_")[1].split(".json")[0] for _, _, files
                             in os.walk(folder) for file in files]))
numba_usage = [True, False]
test_params = list(product(found_versions, numba_usage))
# as of version 0.8.0, water and gas grids were separately created for convert_format testing
minimal_version_two_nets = "0.8.0"


@pytest.mark.slow
@pytest.mark.parametrize("pp_version, use_numba", test_params)
def test_convert_format(pp_version, use_numba):
    if version.parse(pp_version) >= version.parse(minimal_version_two_nets):
        names = ["_gas", "_water"]
    else:
        names = [""]
    for name in names:
        filename = os.path.join(folder, "example_%s%s.json" % (pp_version, name))
        if not os.path.isfile(filename):
            raise ValueError("File for %s grid of version %s does not exist" % (name, pp_version))
        try:
            net = pp.from_json(filename, convert=False)
        except:
            raise UserWarning("Can not load %s network saved in pandapipes version %s"
                              % (name, pp_version))
        p_bar_old = net.res_junction.p_bar.copy()
        v_mean_m_per_s_old = net.res_pipe.v_mean_m_per_s.copy()
        convert_format(net)
        try:
            pp.pipeflow(net, run_control="controller" in net and len(net.controller) > 0,
                        use_numba=use_numba)
        except:
            raise UserWarning("Can not run pipe flow in %s network "
                              "saved with pandapipes version %s" % (name, pp_version))
        p_res = np.allclose(p_bar_old.values, net.res_junction.p_bar.values)
        v_res = np.allclose(v_mean_m_per_s_old.values, net.res_pipe.v_mean_m_per_s.values)
        if not all([p_res, v_res]):
            raise UserWarning("Pipe flow results mismatch in %s grid"
                              "with pandapipes version %s" % (name, pp_version))


if __name__ == '__main__':
    pytest.main([__file__])
