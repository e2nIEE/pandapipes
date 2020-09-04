# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import os

import numpy as np
import pandapipes
import pytest
from pandapipes import pp_dir
from pandapipes.io import convert_format

folder = os.path.join(pp_dir, "test", "api", "old_versions")
found_versions = [file.split("_")[1].split(".json")[0] for _, _, files
                  in os.walk(folder) for file in files]


@pytest.mark.slow
@pytest.mark.parametrize("found_version", found_versions)
def test_convert_format(found_version):
    filename = os.path.join(folder, "example_%s.json" % found_version)
    if not os.path.isfile(filename):
        raise ValueError("File for version %s does not exist" % found_version)
    try:
        net = pandapipes.from_json(filename, convert=False)
    except:
        raise UserWarning("Can not load network saved in pandapipes version %s" % found_version)
    p_bar_old = net.res_junction.p_bar.copy()
    v_mean_m_per_s_old = net.res_pipe.v_mean_m_per_s.copy()
    convert_format(net)
    try:
        pandapipes.pipeflow(net, run_control="controller" in net and len(net.controller) > 0)
    except:
        raise UserWarning("Can not run pipe flow in network "
                          "saved with pandapipes version %s" % found_version)
    p_res = np.allclose(p_bar_old.values, net.res_junction.p_bar.values)
    v_res = np.allclose(v_mean_m_per_s_old.values, net.res_pipe.v_mean_m_per_s.values)
    if not all([p_res, v_res]):
        raise UserWarning("Pipe flow results mismatch "
                          "with pandapipes version %s" % found_version)


if __name__ == '__main__':
    pytest.main([__file__])
