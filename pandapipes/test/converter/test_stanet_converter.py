# Copyright (c) 2020-2022 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np

import pandapipes
from pandapipes.converter.stanet.stanet2pandapipes import stanet_to_pandapipes
from pandapipes.pipeflow import logger as pf_logger
from pandapipes.test import test_path

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
pf_logger.setLevel(logging.WARNING)


test_file_folder = os.path.join(test_path, "converter", "converter_test_files")


def test_mini_schutterwald():
    """

    :param log_results:
    :type log_results:
    :return:
    :rtype:
    """
    schutterwald_path = os.path.join(test_file_folder, "Schutterwald_mini.CSV")
    net = stanet_to_pandapipes(schutterwald_path, add_layers=False)
    pandapipes.pipeflow(net)

    res_p_pp = net.res_junction.p_bar
    res_p_stanet = net.junction.p_stanet
    p_diff = (res_p_pp - res_p_stanet).abs()
    assert np.all(p_diff < 1e-4)

    res_v_pp = net.res_pipe.v_mean_m_per_s
    res_v_stanet = net.pipe.v_stanet
    v_diff = (res_v_pp - res_v_stanet).abs()

    section_counts = net.pipe.groupby("stanet_id").stanet_id.count()
    several_sections = net.pipe.apply(lambda pip: section_counts[pip.stanet_id] > 1, axis=1)
    assert np.all(v_diff.loc[~several_sections] < 1e-4)
    assert np.all(v_diff.loc[several_sections] < 0.02)
