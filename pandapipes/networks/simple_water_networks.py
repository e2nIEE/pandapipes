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
water_stanet_path = os.path.join(pp_dir, "networks", "simple_test_networks", "stanet_test_networks",
                                 "water_cases")
water_modelica_path = os.path.join(pp_dir, "networks", "simple_test_networks",
                                   "openmodelica_test_networks", "water_cases")


# -------------- combined networks --------------
def water_district_grid(method="nikuradse"):
    """

    :param method: which results should be loaded: nikuradse or prandtl-colebrook
    :type method: str, default "nikuradse"
    :return: net - STANET network converted to a pandapipes network
    :rtype: pandapipesNet

    :Example:
        >>> pandapipes.networks.simple_water_networks.water_district_grid(method="pc")

    """
    log_result_upon_loading(logger, method=method, converter="stanet")
    net_name = "district_N.json" if method.lower() in ["nikuradse", "n"] else "district_PC.json"
    return from_json(os.path.join(water_stanet_path, "combined_networks", net_name))


def water_combined_mixed():
    """

    :return: net - OpenModelica network converted to a pandapipes network
    :rtype: pandapipesNet

    :Example:
        >>> pandapipes.networks.simple_water_networks.water_combined_mixed()

    """
    log_result_upon_loading(logger, method="", converter="openmodelica")
    return from_json(os.path.join(water_modelica_path, "combined_networks", "mixed_net.json"))


def water_combined_versatility(results_from="openmodelica", method="n"):
    """

    :param results_from: which converted net should be loaded: openmodelica or stanet
    :type results_from: str, default "openmodelica"
    :param method: if results_from = "stanet", which results should be loaded: nikuradse or prandtl-colebrook
    :type method: str, default "nikuradse"
    :return: net - STANET resp. OpenModelica network converted to a pandapipes network
    :rtype: pandapipesNet

    :Example:
        >>> pandapipes.networks.simple_water_networks.water_combined_versatility(method="")

    """
    log_result_upon_loading(logger, method=method, converter=results_from)
    if results_from.lower() == "stanet":
        net_name = "versatility_N.json" if method.lower() in ["nikuradse", "n"] \
            else "versatility_PC.json"
        return from_json(os.path.join(water_stanet_path, "combined_networks", net_name))
    return from_json(os.path.join(water_modelica_path, "combined_networks", "versatility.json"))


# -------------- meshed networks --------------
def water_meshed_delta(results_from="openmodelica"):
    """

    :param results_from: which converted net should be loaded: openmodelica or stanet
    :type results_from: str, default "openmodelica"
    :return: net - STANET resp. OpenModelica network converted to a pandapipes network
    :rtype: pandapipesNet

    :Example:
        >>> pandapipes.networks.simple_water_networks.water_meshed_delta(results_from="stanet")

    """
    log_result_upon_loading(logger, method="n", converter=results_from)
    if results_from.lower() == "stanet":
        return from_json(os.path.join(water_stanet_path, "meshed_networks", "delta_N.json"))
    return from_json(os.path.join(water_modelica_path, "meshed_networks", "delta.json"))


def water_meshed_pumps(results_from="openmodelica"):
    """

    :param results_from: which converted net should be loaded: openmodelica or stanet
    :type results_from: str, default "openmodelica"
    :return: net - STANET resp. OpenModelica network converted to a pandapipes network
    :rtype: pandapipesNet

    :Example:
        >>> pandapipes.networks.simple_water_networks.water_meshed_pumps()

    """
    log_result_upon_loading(logger, method="n", converter=results_from)
    if results_from.lower() == "stanet":
        return from_json(os.path.join(water_stanet_path, "meshed_networks", "pumps_N.json"))
    return from_json(os.path.join(water_modelica_path, "meshed_networks", "pumps.json"))


def water_meshed_heights():
    """

    :return: net - OpenModelica network converted to a pandapipes network
    :rtype: pandapipesNet

    :Example:
        >>> pandapipes.networks.simple_water_networks.water_meshed_heights()

    """
    log_result_upon_loading(logger, method="", converter="openmodelica")
    return from_json(os.path.join(water_modelica_path, "meshed_networks", "heights.json"))


def water_meshed_2valves(method="nikuradse", results_from="openmodelica"):
    """

    :param results_from: which converted net should be loaded: openmodelica or stanet
    :type results_from: str, default "openmodelica"
    :param method: if results_from = "stanet", which results should be loaded: nikuradse or prandtl-colebrook
    :type method: str, default "nikuradse"
    :return: net - STANET resp. OpenModelica network converted to a pandapipes network
    :rtype: pandapipesNet

    :Example:
        >>> pandapipes.networks.simple_water_networks.water_meshed_2valves()

    """
    log_result_upon_loading(logger, method=method, converter=results_from)
    net_name = "two_valves_N.json" if method.lower() in ["nikuradse", "n"] else "two_valves_PC.json"
    if results_from.lower() == "stanet":
        return from_json(os.path.join(water_stanet_path, "meshed_networks", net_name))
    return from_json(os.path.join(water_modelica_path, "meshed_networks", "two_valves.json"))


# -------------- one pipe --------------
def water_one_pipe1(method="nikuradse", results_from="openmodelica"):
    """

    :param results_from: which converted net should be loaded: openmodelica or stanet
    :type results_from: str, default "openmodelica"
    :param method: if results_from = "stanet", which results should be loaded: nikuradse or prandtl-colebrook
    :type method: str, default "nikuradse"
    :return: net - STANET resp. OpenModelica network converted to a pandapipes network
    :rtype: pandapipesNet

    :Example:
        >>> pandapipes.networks.simple_water_networks.water_one_pipe1(method="pc", results_from="stanet")

    """
    log_result_upon_loading(logger, method=method, converter=results_from)
    net_name = "pipe_1_N.json" if method.lower() in ["nikuradse", "n"] else "pipe_1_PC.json"
    if results_from.lower() == "stanet":
        return from_json(os.path.join(water_stanet_path, "one_pipe", net_name))
    return from_json(os.path.join(water_modelica_path, "one_pipe", "pipe_1.json"))


def water_one_pipe2(method="nikuradse", results_from="openmodelica"):
    """

    :param results_from: which converted net should be loaded: openmodelica or stanet
    :type results_from: str, default "openmodelica"
    :param method: if results_from = "stanet", which results should be loaded: nikuradse or prandtl-colebrook
    :type method: str, default "nikuradse"
    :return: net - STANET resp. OpenModelica network converted to a pandapipes network
    :rtype: pandapipesNet

    :Example:
        >>> pandapipes.networks.simple_water_networks.water_one_pipe2()

    """
    log_result_upon_loading(logger, method=method, converter=results_from)
    net_name = "pipe_2_N.json" if method.lower() in ["nikuradse", "n"] else "pipe_2_PC.json"
    if results_from.lower() == "stanet":
        return from_json(os.path.join(water_stanet_path, "one_pipe", net_name))
    return from_json(os.path.join(water_modelica_path, "one_pipe", "pipe_2.json"))


def water_one_pipe3(method="nikuradse", results_from="openmodelica"):
    """

    :param results_from: which converted net should be loaded: openmodelica or stanet
    :type results_from: str, default "openmodelica"
    :param method: if results_from = "stanet", which results should be loaded: nikuradse or prandtl-colebrook
    :type method: str, default "nikuradse"
    :return: net - STANET resp. OpenModelica network converted to a pandapipes network
    :rtype: pandapipesNet

    :Example:
        >>> pandapipes.networks.simple_water_networks.water_one_pipe3(method="")

    """
    log_result_upon_loading(logger, method=method, converter=results_from)
    net_name = "pipe_3_N.json" if method.lower() in ["nikuradse", "n"] else "pipe_3_PC.json"
    if results_from.lower() == "stanet":
        return from_json(os.path.join(water_stanet_path, "one_pipe", net_name))
    return from_json(os.path.join(water_modelica_path, "one_pipe", "pipe_3.json"))


# -------------- strand net --------------
def water_simple_strand_net(method="nikuradse", results_from="openmodelica"):
    """

    :param results_from: which converted net should be loaded: openmodelica or stanet
    :type results_from: str, default "openmodelica"
    :param method: if results_from = "stanet", which results should be loaded: nikuradse or prandtl-colebrook
    :type method: str, default "nikuradse"
    :return: net - STANET resp. OpenModelica network converted to a pandapipes network
    :rtype: pandapipesNet

    :Example:
        >>> pandapipes.networks.simple_water_networks.water_simple_strand_net()

    """
    log_result_upon_loading(logger, method=method, converter=results_from)
    net_name = "strand_net_N.json" if method.lower() in ["nikuradse", "n"] else "strand_net_PC.json"
    if results_from.lower() == "stanet":
        return from_json(os.path.join(water_stanet_path, "strand_net", net_name))
    return from_json(os.path.join(water_modelica_path, "strand_net", "strand_net.json"))


def water_strand_2pipes(method="nikuradse", results_from="openmodelica"):
    """

    :param results_from: which converted net should be loaded: openmodelica or stanet
    :type results_from: str, default "openmodelica"
    :param method: if results_from = "stanet", which results should be loaded: nikuradse or prandtl-colebrook
    :type method: str, default "nikuradse"
    :return: net - STANET resp. OpenModelica network converted to a pandapipes network
    :rtype: pandapipesNet

    :Example:
        >>> pandapipes.networks.simple_water_networks.water_strand_2pipes()

    """
    log_result_upon_loading(logger, method=method, converter=results_from)
    net_name = "two_pipes_N.json" if method.lower() in ["nikuradse", "n"] else "two_pipes_PC.json"
    if results_from.lower() == "stanet":
        return from_json(os.path.join(water_stanet_path, "strand_net", net_name))
    return from_json(os.path.join(water_modelica_path, "strand_net", "two_pipes.json"))


def water_strand_cross(results_from="openmodelica"):
    """

    :param results_from: which converted net should be loaded: openmodelica or stanet
    :type results_from: str, default "openmodelica"
    :return: net - STANET resp. OpenModelica network converted to a pandapipes network (method = "prandtl-colebrook")
    :rtype: pandapipesNet

    :Example:
        >>> pandapipes.networks.simple_water_networks.water_strand_cross(results_from="stanet")

    """
    log_result_upon_loading(logger, method="pc", converter=results_from)
    if results_from.lower() == "stanet":
        return from_json(os.path.join(water_stanet_path, "strand_net", "cross_PC.json"))
    return from_json(os.path.join(water_modelica_path, "strand_net", "cross_3ext.json"))


def water_strand_net_2pumps():
    """

    :return: net - OpenModelica network converted to a pandapipes network
    :rtype: pandapipesNet

    :Example:
        >>> pandapipes.networks.simple_water_networks.water_strand_net_2pumps()

    """
    return from_json(os.path.join(water_modelica_path, "strand_net", "two_pumps.json"))


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
def water_tcross(method="nikuradse", results_from="openmodelica"):
    """

    :param results_from: which converted net should be loaded: openmodelica or stanet
    :type results_from: str, default "openmodelica"
    :param method: if results_from = "stanet", which results should be loaded: nikuradse or prandtl-colebrook
    :type method: str, default "nikuradse"
    :return: net - STANET resp. OpenModelica network converted to a pandapipes network
    :rtype: pandapipesNet

    :Example:
        >>> pandapipes.networks.simple_water_networks.water_tcross()

    """
    log_result_upon_loading(logger, method=method, converter=results_from)
    net_name = "t_cross_N.json" if method.lower() in ["nikuradse", "n"] else "t_cross_PC.json"
    if results_from.lower() == "stanet":
        return from_json(os.path.join(water_stanet_path, "t_cross", net_name))
    return from_json(os.path.join(water_modelica_path, "t_cross", "t_cross.json"))


def water_tcross_valves():
    """

    :return: net - OpenModelica network converted to a pandapipes network
    :rtype: pandapipesNet

    :Example:
        >>> pandapipes.networks.simple_water_networks.water_tcross_valves()

    """
    log_result_upon_loading(logger, method="", converter="openmodelica")
    return from_json(os.path.join(water_modelica_path, "t_cross", "valves.json"))


# -------------- two pressure junctions --------------
def water_2eg_two_pipes(method="nikuradse", results_from="openmodelica"):
    """

    :param results_from: which converted net should be loaded: openmodelica or stanet
    :type results_from: str, default "openmodelica"
    :param method: if results_from = "stanet", which results should be loaded: nikuradse or prandtl-colebrook
    :type method: str, default "nikuradse"
    :return: net - STANET resp. OpenModelica network converted to a pandapipes network
    :rtype: pandapipesNet

    :Example:
        >>> pandapipes.networks.simple_water_networks.water_2eg_two_pipes()

    """
    log_result_upon_loading(logger, method=method, converter=results_from)
    net_name = "two_pipes_N.json" if method.lower() in ["nikuradse", "n"] else "two_pipes_PC.json"
    if results_from.lower() == "stanet":
        return from_json(os.path.join(water_stanet_path, "two_pressure_junctions", net_name))
    return from_json(os.path.join(water_modelica_path, "two_pressure_junctions", "two_pipes.json"))
