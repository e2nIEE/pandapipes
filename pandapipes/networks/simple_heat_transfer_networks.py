# Copyright (c) 2020-2021 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import os
from pandapipes.io.file_io import from_json
from pandapipes import pp_dir
try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)

heat_tranfer_modelica_path = os.path.join(pp_dir, "networks", "network_files",
                                   "openmodelica_test_networks", "heat_transfer_cases")

def heat_transfer_delta():
    """

    :return: net - OpenModelica network converted to a pandapipes network
    :rtype: pandapipesNet

    :Example:
        >>> pandapipes.networks.simple_heat_transfer_networks.heat_transfer_delta()

    """
    return from_json(os.path.join(heat_tranfer_modelica_path, "delta.json"))


def heat_transfer_delta_2sinks():
    """

    :return: net - OpenModelica network converted to a pandapipes network
    :rtype: pandapipesNet

    :Example:
        >>> pandapipes.networks.simple_heat_transfer_networks.heat_transfer_delta_2sinks()

    """
    return from_json(os.path.join(heat_tranfer_modelica_path, "delta_2sinks.json"))


def heat_transfer_heights():
    """

    :return: net - OpenModelica network converted to a pandapipes network
    :rtype: pandapipesNet

    :Example:
        >>> pandapipes.networks.simple_heat_transfer_networks.heat_transfer_heights()

    """
    return from_json(os.path.join(heat_tranfer_modelica_path, "heights.json"))


def heat_transfer_one_pipe():
    """

    :return: net - OpenModelica network converted to a pandapipes network
    :rtype: pandapipesNet

    :Example:
        >>> pandapipes.networks.simple_heat_transfer_networks.heat_transfer_one_pipe()

    """
    return from_json(os.path.join(heat_tranfer_modelica_path, "one_pipe.json"))


def heat_transfer_one_source():
    """

    :return: net - OpenModelica network converted to a pandapipes network
    :rtype: pandapipesNet

    :Example:
        >>> pandapipes.networks.simple_heat_transfer_networks.heat_transfer_one_source()

    """
    return from_json(os.path.join(heat_tranfer_modelica_path, "one_source.json"))


def heat_transfer_section_variation():
    """

    :return: net - OpenModelica network converted to a pandapipes network
    :rtype: pandapipesNet

    :Example:
        >>> pandapipes.networks.simple_heat_transfer_networks.heat_transfer_section_variation()

    """
    return from_json(os.path.join(heat_tranfer_modelica_path, "section_variation.json"))


def heat_transfer_t_cross():
    """

    :return: net - OpenModelica network converted to a pandapipes network
    :rtype: pandapipesNet

    :Example:
        >>> pandapipes.networks.simple_heat_transfer_networks.heat_transfer_t_cross()

    """
    return from_json(os.path.join(heat_tranfer_modelica_path, "t_cross.json"))


def heat_transfer_two_pipes():
    """

    :return: net - OpenModelica network converted to a pandapipes network
    :rtype: pandapipesNet

    :Example:
        >>> pandapipes.networks.simple_heat_transfer_networks.heat_transfer_two_pipes()

    """
    return from_json(os.path.join(heat_tranfer_modelica_path, "two_pipes.json"))
