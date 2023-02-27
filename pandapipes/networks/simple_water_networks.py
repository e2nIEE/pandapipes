# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import os
from pandapipes.io.file_io import from_json
from pandapipes import pp_dir
from pandapipes.networks.nw_aux import log_result_upon_loading

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)
water_stanet_path = os.path.join(pp_dir, "networks", "network_files", "stanet_test_networks",
                                 "water_cases")
water_modelica_colebrook_path = os.path.join(pp_dir, "networks", "network_files",
                                   "openmodelica_test_networks", "water_cases_colebrook")

water_modelica_swamee_path = os.path.join(pp_dir, "networks", "network_files",
                                   "openmodelica_test_networks", "water_cases_swamee-jain")


# -------------- combined networks --------------
def water_district_grid(method="nikuradse"):
    """

    :param method: Which results should be loaded: nikuradse or prandtl-colebrook
    :type method: str, default "nikuradse"
    :return: net - STANET network converted to a pandapipes network
    :rtype: pandapipesNet

    :Example:
        >>> pandapipes.networks.simple_water_networks.water_district_grid(method="pc")

    """
    method_str = log_result_upon_loading(logger, method=method, converter="stanet")
    net_name = "district_N.json" if method_str == "Nikuradse" else "district_PC.json"
    return from_json(os.path.join(water_stanet_path, "combined_networks", net_name))


def water_combined_mixed(method="colebrook"):
    """

    :param method: Which results should be loaded: prandtl-colebrook or swamee-jain
    :type method: str, default "colebrook"
    :return: net - OpenModelica network converted to a pandapipes network
    :rtype: pandapipesNet

    :Example:
        >>> pandapipes.networks.simple_water_networks.water_combined_mixed(method="swamee-jain")

    """
    method_str = log_result_upon_loading(logger, method=method, converter="openmodelica")
    if method_str == "Prandtl-Colebrook":
        return from_json(os.path.join(water_modelica_colebrook_path, "combined_networks", "mixed_net.json"))
    return from_json(os.path.join(water_modelica_swamee_path, "combined_networks", "mixed_net.json"))


def water_combined_versatility(results_from="openmodelica", method="colebrook"):
    """

    :param results_from: Which converted net should be loaded: openmodelica or stanet
    :type results_from: str, default "openmodelica"
    :param method: results_from = "stanet" -> nikuradse or prandtl-colebrook, results_from = "openmodelica" -> prandtl-colebrook or swamee-jain
    :type method: str, default "colebrook"
    :return: net - STANET resp. OpenModelica network converted to a pandapipes network
    :rtype: pandapipesNet

    :Example:
        >>> pandapipes.networks.simple_water_networks.water_combined_versatility(method="")

    """
    method_str = log_result_upon_loading(logger, method=method, converter=results_from)
    if results_from.lower() == "stanet":
        net_name = "versatility_N.json" if method_str == "Nikuradse" else "versatility_PC.json"
        return from_json(os.path.join(water_stanet_path, "combined_networks", net_name))

    if method_str == "Prandtl-Colebrook":
        return from_json(os.path.join(water_modelica_colebrook_path, "combined_networks", "versatility.json"))
    return from_json(os.path.join(water_modelica_swamee_path, "combined_networks", "versatility.json"))


# -------------- meshed networks --------------
def water_meshed_delta(results_from="openmodelica", method="colebrook"):
    """

    :param results_from: Which converted net should be loaded: openmodelica or stanet
    :type results_from: str, default "openmodelica"
    :param method: results_from = "stanet" -> nikuradse, results_from = "openmodelica" -> prandtl-colebrook or swamee-jain
    :type method: str, default "colebrook"
    :return: net - STANET resp. OpenModelica network converted to a pandapipes network
    :rtype: pandapipesNet

    :Example:
        >>> pandapipes.networks.simple_water_networks.water_meshed_delta(results_from="stanet")

    """
    method = "n" if results_from.lower() == "stanet" else method

    method_str = log_result_upon_loading(logger, method=method, converter=results_from)
    if results_from.lower() == "stanet":
        return from_json(os.path.join(water_stanet_path, "meshed_networks", "delta_N.json"))

    if method_str == "Prandtl-Colebrook":
        return from_json(os.path.join(water_modelica_colebrook_path, "meshed_networks", "delta.json"))
    return from_json(os.path.join(water_modelica_swamee_path, "meshed_networks", "delta.json"))


def water_meshed_pumps(results_from="openmodelica", method="colebrook"):
    """

    :param results_from: Which converted net should be loaded: openmodelica or stanet
    :type results_from: str, default "openmodelica"
    :param method: results_from = "stanet" -> nikuradse or prandtl-colebrook, results_from = "openmodelica" -> prandtl-colebrook or swamee-jain
    :type method: str, default "colebrook"
    :return: net - STANET resp. OpenModelica network converted to a pandapipes network
    :rtype: pandapipesNet

    :Example:
        >>> pandapipes.networks.simple_water_networks.water_meshed_pumps(method="swamee")

    """
    method = "n" if results_from.lower() == "stanet" else method

    method_str = log_result_upon_loading(logger, method=method, converter=results_from)
    if results_from.lower() == "stanet":
        return from_json(os.path.join(water_stanet_path, "meshed_networks", "pumps_N.json"))

    if method_str == "Prandtl-Colebrook":
        return from_json(os.path.join(water_modelica_colebrook_path, "meshed_networks", "pumps.json"))
    return from_json(os.path.join(water_modelica_swamee_path, "meshed_networks", "pumps.json"))

def water_meshed_heights(method="colebrook"):

    """
    :param method: which results should be loaded: prandtl-colebrook or swamee-jain
    :type method: str, default "colebrook"
    :return: net - OpenModelica network converted to a pandapipes network
    :rtype: pandapipesNet

    :Example:
        >>> pandapipes.networks.simple_water_networks.water_meshed_heights()

    """
    method_str = log_result_upon_loading(logger, method=method, converter="openmodelica")
    if method_str == "Prandtl-Colebrook":
        return from_json(os.path.join(water_modelica_colebrook_path, "meshed_networks", "heights.json"))
    return from_json(os.path.join(water_modelica_swamee_path, "meshed_networks", "heights.json"))


def water_meshed_2valves(results_from="openmodelica", method="colebrook"):
    """

    :param results_from: Which converted net should be loaded: openmodelica or stanet
    :type results_from: str, default "openmodelica"
    :param method: results_from = "stanet" -> nikuradse or prandtl-colebrook, results_from = "openmodelica" -> prandtl-colebrook or swamee-jain
    :type method: str, default "colebrook"
    :return: net - STANET resp. OpenModelica network converted to a pandapipes network
    :rtype: pandapipesNet

    :Example:
        >>> pandapipes.networks.simple_water_networks.water_meshed_2valves(method="swamee-jain")

    """
    method_str = log_result_upon_loading(logger, method=method, converter=results_from)

    if results_from.lower() == "stanet":
        net_name = "two_valves_N.json" if method_str == "Nikuradse" else "two_valves_PC.json"
        return from_json(os.path.join(water_stanet_path, "meshed_networks", net_name))

    if method_str == "Prandtl-Colebrook":
        return from_json(os.path.join(water_modelica_colebrook_path, "meshed_networks", "two_valves.json"))
    return from_json(os.path.join(water_modelica_swamee_path, "meshed_networks", "two_valves.json"))


# -------------- one pipe --------------
def water_one_pipe1(results_from="openmodelica", method="colebrook"):
    """

    :param results_from: Which converted net should be loaded: openmodelica or stanet
    :type results_from: str, default "openmodelica"
    :param method: results_from = "stanet" -> nikuradse or prandtl-colebrook, results_from = "openmodelica" -> prandtl-colebrook or swamee-jain
    :type method: str, default "colebrook"
    :return: net - STANET resp. OpenModelica network converted to a pandapipes network
    :rtype: pandapipesNet

    :Example:
        >>> pandapipes.networks.simple_water_networks.water_one_pipe1(method="pc", results_from="stanet")

    """
    method_str = log_result_upon_loading(logger, method=method, converter=results_from)

    if results_from.lower() == "stanet":
        net_name = "pipe_1_N.json" if method_str == "Nikuradse" else "pipe_1_PC.json"
        return from_json(os.path.join(water_stanet_path, "one_pipe", net_name))

    if method_str == "Prandtl-Colebrook":
        return from_json(os.path.join(water_modelica_colebrook_path, "one_pipe", "pipe_1.json"))
    return from_json(os.path.join(water_modelica_swamee_path, "one_pipe", "pipe_1.json"))


def water_one_pipe2(results_from="openmodelica", method="colebrook"):
    """

    :param results_from: Which converted net should be loaded: openmodelica or stanet
    :type results_from: str, default "openmodelica"
    :param method: results_from = "stanet" -> nikuradse or prandtl-colebrook, results_from = "openmodelica" -> prandtl-colebrook or swamee-jain
    :type method: str, default "colebrook"
    :return: net - STANET resp. OpenModelica network converted to a pandapipes network
    :rtype: pandapipesNet

    :Example:
        >>> pandapipes.networks.simple_water_networks.water_one_pipe2()

    """
    method_str = log_result_upon_loading(logger, method=method, converter=results_from)

    if results_from.lower() == "stanet":
        net_name = "pipe_2_N.json" if method_str == "Nikuradse" else "pipe_2_PC.json"
        return from_json(os.path.join(water_stanet_path, "one_pipe", net_name))

    if method_str == "Prandtl-Colebrook":
        return from_json(os.path.join(water_modelica_colebrook_path, "one_pipe", "pipe_2.json"))
    return from_json(os.path.join(water_modelica_swamee_path, "one_pipe", "pipe_2.json"))


def water_one_pipe3(results_from="openmodelica", method="colebrook"):
    """

    :param results_from: Which converted net should be loaded: openmodelica or stanet
    :type results_from: str, default "openmodelica"
    :param method: results_from = "stanet" -> nikuradse or prandtl-colebrook, results_from = "openmodelica" -> prandtl-colebrook or swamee-jain
    :type method: str, default "colebrook"
    :return: net - STANET resp. OpenModelica network converted to a pandapipes network
    :rtype: pandapipesNet

    :Example:
        >>> pandapipes.networks.simple_water_networks.water_one_pipe3(method="")

    """
    method_str = log_result_upon_loading(logger, method=method, converter=results_from)

    if results_from.lower() == "stanet":
        net_name = "pipe_3_N.json" if method_str == "Nikuradse" else "pipe_3_PC.json"
        return from_json(os.path.join(water_stanet_path, "one_pipe", net_name))

    if method_str == "Prandtl-Colebrook":
        return from_json(os.path.join(water_modelica_colebrook_path, "one_pipe", "pipe_3.json"))
    return from_json(os.path.join(water_modelica_swamee_path, "one_pipe", "pipe_3.json"))


# -------------- strand net --------------
def water_simple_strand_net( results_from="openmodelica", method="colebrook"):
    """

    :param results_from: Which converted net should be loaded: openmodelica or stanet
    :type results_from: str, default "openmodelica"
    :param method: results_from = "stanet" -> nikuradse or prandtl-colebrook, results_from = "openmodelica" -> prandtl-colebrook or swamee-jain
    :type method: str, default "colebrook"
    :return: net - STANET resp. OpenModelica network converted to a pandapipes network
    :rtype: pandapipesNet

    :Example:
        >>> pandapipes.networks.simple_water_networks.water_simple_strand_net()

    """
    method_str = log_result_upon_loading(logger, method=method, converter=results_from)

    if results_from.lower() == "stanet":
        net_name = "strand_net_N.json" if method_str == "Nikuradse" else "strand_net_PC.json"
        return from_json(os.path.join(water_stanet_path, "strand_net", net_name))

    if method_str == "Prandtl-Colebrook":
        return from_json(os.path.join(water_modelica_colebrook_path, "strand_net", "strand_net.json"))
    return from_json(os.path.join(water_modelica_swamee_path, "strand_net", "strand_net.json"))


def water_strand_2pipes(results_from="openmodelica", method="colebrook"):
    """

    :param results_from: Which converted net should be loaded: openmodelica or stanet
    :type results_from: str, default "openmodelica"
    :param method: results_from = "stanet" -> nikuradse or prandtl-colebrook, results_from = "openmodelica" -> prandtl-colebrook or swamee-jain
    :type method: str, default "colebrook"
    :return: net - STANET resp. OpenModelica network converted to a pandapipes network
    :rtype: pandapipesNet

    :Example:
        >>> pandapipes.networks.simple_water_networks.water_strand_2pipes()

    """
    method_str = log_result_upon_loading(logger, method=method, converter=results_from)

    if results_from.lower() == "stanet":
        net_name = "two_pipes_N.json" if method_str == "Nikuradse" else "two_pipes_PC.json"
        return from_json(os.path.join(water_stanet_path, "strand_net", net_name))

    if method_str == "Prandtl-Colebrook":
        return from_json(os.path.join(water_modelica_colebrook_path, "strand_net", "two_pipes.json"))
    return from_json(os.path.join(water_modelica_swamee_path, "strand_net", "two_pipes.json"))


def water_strand_cross(results_from="openmodelica", method="colebrook"):
    """

    :param results_from: Which converted net should be loaded: openmodelica or stanet
    :type results_from: str, default "openmodelica"
    :param method: results_from = "stanet" -> prandtl-colebrook, results_from = "openmodelica" -> prandtl-colebrook or swamee-jain
    :type method: str, default "colebrook"
    :return: net - STANET resp. OpenModelica network converted to a pandapipes network (method = "prandtl-colebrook")
    :rtype: pandapipesNet

    :Example:
        >>> pandapipes.networks.simple_water_networks.water_strand_cross(results_from="stanet")

    """
    method_str = log_result_upon_loading(logger, method=method, converter=results_from)

    if results_from.lower() == "stanet":
        return from_json(os.path.join(water_stanet_path, "strand_net", "cross_PC.json"))

    if method_str == "Prandtl-Colebrook":
        return from_json(os.path.join(water_modelica_colebrook_path, "strand_net", "cross_3ext.json"))
    return from_json(os.path.join(water_modelica_swamee_path, "strand_net", "cross_3ext.json"))


def water_strand_net_2pumps(method="colebrook"):
    """
    :param method: Which results should be loaded: prandtl-colebrook or swamee-jain
    :type method: str, default "colebrook"
    :return: net - OpenModelica network converted to a pandapipes network
    :rtype: pandapipesNet

    :Example:
        >>> pandapipes.networks.simple_water_networks.water_strand_net_2pumps()

    """
    method_str = log_result_upon_loading(logger, method=method, converter="openmodelica")

    if method_str == "Prandtl-Colebrook":
        return from_json(os.path.join(water_modelica_colebrook_path, "strand_net", "two_pumps.json"))
    return from_json(os.path.join(water_modelica_swamee_path, "strand_net", "two_pumps.json"))


def water_strand_pump():
    """

    :return: net - STANET network converted to a pandapipes network  (method = "nikuradse")
    :rtype: pandapipesNet

    :Example:
        >>> pandapipes.networks.simple_water_networks.water_meshed_pump()

    """
    log_result_upon_loading(logger, method="n", converter="stanet")
    return from_json(os.path.join(water_stanet_path, "strand_net", "pump_N.json"))


# -------------- t_cross --------------
def water_tcross(results_from="openmodelica", method="colebrook"):
    """

    :param results_from: Which converted net should be loaded: openmodelica or stanet
    :type results_from: str, default "openmodelica"
    :param method: results_from = "stanet" -> prandtl-colebrook, results_from = "openmodelica" -> prandtl-colebrook or swamee-jain
    :type method: str, default "colebrook"
    :return: net - STANET resp. OpenModelica network converted to a pandapipes network
    :rtype: pandapipesNet

    :Example:
        >>> pandapipes.networks.simple_water_networks.water_tcross()

    """
    method_str = log_result_upon_loading(logger, method=method, converter=results_from)

    if results_from.lower() == "stanet":
        net_name = "t_cross_N.json" if method_str == "Nikuradse" else "t_cross_PC.json"
        return from_json(os.path.join(water_stanet_path, "t_cross", net_name))

    if method_str == "Prandtl-Colebrook":
        return from_json(os.path.join(water_modelica_colebrook_path, "t_cross", "t_cross.json"))
    return from_json(os.path.join(water_modelica_swamee_path, "t_cross", "t_cross.json"))


def water_tcross_valves(method="colebrook"):
    """
    :param method: Which results should be loaded: prandtl-colebrook or swamee-jain
    :type method: str, default "colebrook"
    :return: net - OpenModelica network converted to a pandapipes network
    :rtype: pandapipesNet

    :Example:
        >>> pandapipes.networks.simple_water_networks.water_tcross_valves()

    """
    method_str = log_result_upon_loading(logger, method=method, converter="openmodelica")

    if method_str == "Prandtl-Colebrook":
        return from_json(os.path.join(water_modelica_colebrook_path, "t_cross", "valves.json"))
    return from_json(os.path.join(water_modelica_swamee_path, "t_cross", "valves.json"))


# -------------- two pressure junctions --------------
def water_2eg_two_pipes(results_from="openmodelica", method="colebrook"):
    """

    :param results_from: Which converted net should be loaded: openmodelica or stanet
    :type results_from: str, default "openmodelica"
    :param method: results_from = "stanet" -> prandtl-colebrook, results_from = "openmodelica" -> prandtl-colebrook or swamee-jain
    :type method: str, default "colebrook"
    :return: net - STANET resp. OpenModelica network converted to a pandapipes network
    :rtype: pandapipesNet

    :Example:
        >>> pandapipes.networks.simple_water_networks.water_2eg_two_pipes()

    """
    method_str = log_result_upon_loading(logger, method=method, converter=results_from)

    if results_from.lower() == "stanet":
        net_name = "two_pipes_N.json" if method_str == "Nikuradse" else "two_pipes_PC.json"
        return from_json(os.path.join(water_stanet_path, "two_pressure_junctions", net_name))


    if method_str == "Prandtl-Colebrook":
        return from_json(os.path.join(water_modelica_colebrook_path, "two_pressure_junctions", "two_pipes.json"))
    return from_json(os.path.join(water_modelica_swamee_path, "two_pressure_junctions", "two_pipes.json"))

