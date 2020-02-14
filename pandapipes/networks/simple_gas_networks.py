# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import os
from pandapipes.io.file_io import from_json
from pandapipes import pp_dir
from pandapipes.networks.nw_aux import log_result_upon_loading

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)
gas_stanet_path = os.path.join(pp_dir, "networks", "simple_test_networks", "stanet_test_networks",
                               "gas_cases")


# -------------- combined networks --------------
def gas_3parallel(method="nikuradse"):
    """

    :param method: which results should be loaded: nikuradse or prandtl-colebrook
    :type method: str, default "nikuradse"
    :return: net - STANET network converted to a pandapipes network
    :rtype: pandapipesNet

    :Example:
        >>> pandapipes.networks.simple_gas_networks.gas_3parallel(method="n")

    """
    log_result_upon_loading(logger, method=method, converter="stanet")
    net_name = "parallel_N.json" if method.lower() in ["nikuradse", "n"] else "parallel_PC.json"
    return from_json(os.path.join(gas_stanet_path, "combined_networks", net_name))


def gas_versatility():
    """

    :return: net - STANET network converted to a pandapipes network (method = "prandtl-colebrook")
    :rtype: pandapipesNet

    :Example:
        >>> pandapipes.networks.simple_gas_networks.gas_versatility()

    """
    log_result_upon_loading(logger, method="pc", converter="stanet")
    net_name = "versatility_PC.json"
    return from_json(os.path.join(gas_stanet_path, "combined_networks", net_name))


# -------------- meshed networks --------------
def gas_meshed_delta():
    """

    :return: net - STANET network converted to a pandapipes network (method = "prandtl-colebrook")
    :rtype: pandapipesNet

    :Example:
        >>> pandapipes.networks.simple_gas_networks.gas_meshed_delta()

    """
    log_result_upon_loading(logger, method="pc", converter="stanet")
    return from_json(os.path.join(gas_stanet_path, "meshed_networks", "delta_PC.json"))


def gas_meshed_pumps():
    """

    :return: net - STANET network converted to a pandapipes network (method = "nikuradse")
    :rtype: pandapipesNet

    :Example:
        >>> pandapipes.networks.simple_gas_networks.gas_meshed_pumps()

    """
    log_result_upon_loading(logger, method="n", converter="stanet")
    return from_json(os.path.join(gas_stanet_path, "meshed_networks", "pumps_N.json"))


def gas_meshed_square(method="nikuradse"):
    """

    :param method: which results should be loaded: nikuradse or prandtl-colebrook
    :type method: str, default "nikuradse"
    :return: net - STANET network converted to a pandapipes network
    :rtype: pandapipesNet

    :Example:
        >>> pandapipes.networks.simple_gas_networks.gas_meshed_square(method="n")

    """
    log_result_upon_loading(logger, method=method, converter="stanet")
    net_name = "square_N.json" if method.lower() in ["nikuradse", "n"] else "square_PC.json"
    return from_json(os.path.join(gas_stanet_path, "meshed_networks", net_name))


def gas_meshed_two_valves(method="nikuradse"):
    """

    :param method: which results should be loaded: nikuradse or prandtl-colebrook
    :type method: str, default "nikuradse"
    :return: net - STANET network converted to a pandapipes network
    :rtype: pandapipesNet

    :Example:
        >>> pandapipes.networks.simple_gas_networks.gas_meshed_two_valves()

    """
    log_result_upon_loading(logger, method=method, converter="stanet")
    net_name = "two_valves_N.json" if method.lower() in ["nikuradse", "n"] else "two_valves_PC.json"
    return from_json(os.path.join(gas_stanet_path, "meshed_networks", net_name))


# -------------- one pipe --------------
def gas_one_pipe1(method="nikuradse"):
    """

    :param method: which results should be loaded: nikuradse or prandtl-colebrook
    :type method: str, default "nikuradse"
    :return: net - STANET network converted to a pandapipes network
    :rtype: pandapipesNet

    :Example:
        >>> pandapipes.networks.simple_gas_networks.gas_one_pipe1(method="pc")

    """
    log_result_upon_loading(logger, method=method, converter="stanet")
    net_name = "pipe_1_N.json" if method.lower() in ["nikuradse", "n"] else "pipe_1_PC.json"
    return from_json(os.path.join(gas_stanet_path, "one_pipe", net_name))


def gas_one_pipe2(method="nikuradse"):
    """

    :param method: which results should be loaded: nikuradse or prandtl-colebrook
    :type method: str, default "nikuradse"
    :return: net - STANET network converted to a pandapipes network
    :rtype: pandapipesNet

    :Example:
        >>> pandapipes.networks.simple_gas_networks.gas_one_pipe2()

    """
    log_result_upon_loading(logger, method=method, converter="stanet")
    net_name = "pipe_2_N.json" if method.lower() in ["nikuradse", "n"] else "pipe_2_PC.json"
    return from_json(os.path.join(gas_stanet_path, "one_pipe", net_name))


# -------------- strand net --------------
def gas_strand_2pipes(method="nikuradse"):
    """

    :param method: which results should be loaded: nikuradse or prandtl-colebrook
    :type method: str, default "nikuradse"
    :return: net - STANET network converted to a pandapipes network
    :rtype: pandapipesNet

    :Example:
        >>> pandapipes.networks.simple_gas_networks.gas_strand_2pipes()

    """
    log_result_upon_loading(logger, method=method, converter="stanet")
    net_name = "two_pipes_N.json" if method.lower() in ["nikuradse", "n"] else "two_pipes_PC.json"
    return from_json(os.path.join(gas_stanet_path, "strand_net", net_name))


def gas_strand_pump():
    """

    :return: net - STANET network converted to a pandapipes network  (method = "nikuradse")
    :rtype: pandapipesNet

    :Example:
        >>> pandapipes.networks.simple_gas_networks.gas_meshed_pump()

    """
    log_result_upon_loading(logger, method="n", converter="stanet")
    return from_json(os.path.join(gas_stanet_path, "strand_net", "pump_N.json"))


# -------------- t_cross --------------
def gas_tcross1(method="nikuradse"):
    """

    :param method: if results_from = "stanet", which results should be loaded: nikuradse or prandtl-colebrook
    :type method: str, default "nikuradse"
    :return: net - STANET network converted to a pandapipes network
    :rtype: pandapipesNet

    :Example:
        >>> pandapipes.networks.simple_gas_networks.gas_tcross1(method="pc")

    """
    log_result_upon_loading(logger, method=method, converter="stanet")
    net_name = "t_cross1_N.json" if method.lower() in ["nikuradse", "n"] else "t_cross1_PC.json"
    return from_json(os.path.join(gas_stanet_path, "t_cross", net_name))


def gas_tcross2(method="nikuradse"):
    """

    :param method: if results_from = "stanet", which results should be loaded: nikuradse or prandtl-colebrook
    :type method: str, default "nikuradse"
    :return: net - STANET network converted to a pandapipes network
    :rtype: pandapipesNet

    :Example:
        >>> pandapipes.networks.simple_gas_networks.gas_tcross2(method="pc")

    """
    log_result_upon_loading(logger, method=method, converter="stanet")
    net_name = "t_cross2_N.json" if method.lower() in ["nikuradse", "n"] else "t_cross2_PC.json"
    return from_json(os.path.join(gas_stanet_path, "t_cross", net_name))


# -------------- two pressure junctions --------------
def gas_2eg_hnet(method="nikuradse"):
    """

    :param method: if results_from = "stanet", which results should be loaded: nikuradse or prandtl-colebrook
    :type method: str, default "nikuradse"
    :return: net - STANET network converted to a pandapipes network
    :rtype: pandapipesNet

    :Example:
        >>> pandapipes.networks.simple_gas_networks.gas_2eg_hnet(method="n")

    """
    log_result_upon_loading(logger, method=method, converter="stanet")
    net_name = "H_net_N.json" if method.lower() in ["nikuradse", "n"] else "H_net_PC.json"
    return from_json(os.path.join(gas_stanet_path, "two_pressure_junctions", net_name))
