# Copyright (c) 2020-2024 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

from pandapipes.properties.fluids import get_fluid
from pandapipes.utils.internals import get_net_option
from pandapipes.utils.result_extraction import get_basic_branch_results, get_branch_results_gas_numba, get_branch_results_gas
from pandapipes.component_models.component_registry import COMPONENT_REGISTRY


def extract_all_results(net, calculation_mode):
    """
    Extract results from branch pit and node pit and write them to the different tables of the net,\
    as defined by the component models.

    :param net: pandapipes net for which to extract results into net.res_xy
    :type net: pandapipesNet
    :param net: mode of the simulation (e.g. "hydraulics" or "heat" or "sequential" or "bidirectional")
    :type net: str
    :return: No output

    """
    branch_pit = net["_pit"]["branch"]
    node_pit = net["_pit"]["node"]
    branch_results = get_basic_branch_results(net, branch_pit, node_pit)
    if get_fluid(net).is_gas:
        if get_net_option(net, "use_numba"):
            v_gas_from, v_gas_to, v_gas_mean, p_abs_from, p_abs_to, p_abs_mean, normfactor_from, \
                normfactor_to, normfactor_mean = get_branch_results_gas_numba(
                net, branch_pit, node_pit, branch_results['from_nodes'], branch_results['to_nodes'],
                branch_results['v_mps'], branch_results['p_from'], branch_results['p_to'])
        else:
            v_gas_from, v_gas_to, v_gas_mean, p_abs_from, p_abs_to, p_abs_mean, normfactor_from, \
                normfactor_to, normfactor_mean = get_branch_results_gas(
                net, branch_pit, node_pit, branch_results['from_nodes'], branch_results['to_nodes'],
                branch_results['v_mps'], branch_results['p_from'], branch_results['p_to'])
        gas_branch_results = {
            "v_gas_from": v_gas_from, "v_gas_to": v_gas_to, "v_gas_mean": v_gas_mean,
            "p_from": branch_results['p_from'], "p_to": branch_results['p_to'], "p_abs_from": p_abs_from,
            "p_abs_to": p_abs_to, "p_abs_mean": p_abs_mean, "normfactor_from": normfactor_from,
            "normfactor_to": normfactor_to, "normfactor_mean": normfactor_mean
        }
        branch_results.update(gas_branch_results)
    for comp in net['component_list']:
        COMPONENT_REGISTRY[comp].extract_results(net, net["_options"], branch_results, calculation_mode)
