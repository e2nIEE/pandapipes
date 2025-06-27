import numpy as np

from pandapipes.constants import NORMAL_PRESSURE, NORMAL_TEMPERATURE
from pandapipes.idx_branch import (QEXT, ELEMENT_IDX, FROM_NODE, TO_NODE, MDOTINIT, RE, LAMBDA, PL,
                                   TOUTINIT, AREA, TEXT, LOSS_COEFFICIENT as LC)
from pandapipes.idx_node import TABLE_IDX as TABLE_IDX_NODE, PINIT, PAMB, TINIT as TINIT_NODE
from pandapipes.pf.internals_toolbox import _sum_by_group
from pandapipes.pf.pipeflow_setup import get_table_number, get_lookup, get_net_option
from pandapipes.properties.fluids import get_fluid
from pandapipes.properties.properties_toolbox import get_branch_real_density

try:
    from numba import jit
except ImportError:
    from pandapower.pf.no_numba import jit


def extract_all_results(net, calculation_mode):
    """
    Extract results from branch pit and node pit and write them to the different tables of the net,\
    as defined by the component models.

    :param net: pandapipes net for which to extract results into net.res_xy
    :type net: pandapipesNet
    :param calculation_mode: mode of the simulation (e.g. "hydraulics" or "heat" or "sequential" or "bidirectional")
    :type calculation_mode: str
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
        comp.extract_results(net, net["_options"], branch_results, calculation_mode)


def get_basic_branch_results(net, branch_pit, node_pit):
    from_nodes = branch_pit[:, FROM_NODE].astype(np.int32)
    to_nodes = branch_pit[:, TO_NODE].astype(np.int32)
    t0 = node_pit[from_nodes, TINIT_NODE]
    t1 = node_pit[to_nodes, TINIT_NODE]
    fluid = get_fluid(net)
    if fluid.is_gas:
        vf = branch_pit[:, MDOTINIT] / fluid.get_density(NORMAL_TEMPERATURE)
    else:
        vf = branch_pit[:, MDOTINIT] / get_branch_real_density(fluid, node_pit, branch_pit)
    v = vf / branch_pit[:, AREA]
    t_outlet = branch_pit[:, TOUTINIT]
    branch_results = {"v_mps": v, "mf_from": branch_pit[:, MDOTINIT], "mf_to": -branch_pit[:, MDOTINIT],
                      "vf": vf, "p_from": node_pit[from_nodes, PINIT], "p_to": node_pit[to_nodes, PINIT],
                      "from_nodes": from_nodes, "to_nodes": to_nodes,  "temp_from": t0, "temp_to": t1,
                      "reynolds": branch_pit[:, RE], "lambda": branch_pit[:, LAMBDA], "pl": branch_pit[:, PL],
                      "t_outlet": t_outlet, "qext": branch_pit[:, QEXT], "loss_coeff": branch_pit[:, LC],}
    return branch_results


def get_branch_results_gas(net, branch_pit, node_pit, from_nodes, to_nodes, v_mps, p_from, p_to):
    p_abs_from = node_pit[from_nodes, PAMB] + p_from
    p_abs_to = node_pit[to_nodes, PAMB] + p_to
    mask = ~np.isclose(p_abs_from, p_abs_to)
    p_abs_mean = np.empty_like(p_abs_to)
    p_abs_mean[~mask] = p_abs_from[~mask]
    p_abs_mean[mask] = 2 / 3 * (p_abs_from[mask] ** 3 - p_abs_to[mask] ** 3) \
                       / (p_abs_from[mask] ** 2 - p_abs_to[mask] ** 2)

    fluid = get_fluid(net)
    t_from = node_pit[from_nodes, TINIT_NODE]
    t_to = branch_pit[:, TOUTINIT]
    tm = (t_from + t_to) / 2
    numerator_from = NORMAL_PRESSURE * t_from / NORMAL_TEMPERATURE
    numerator_to = NORMAL_PRESSURE * t_to / NORMAL_TEMPERATURE
    numerator = NORMAL_PRESSURE * tm / NORMAL_TEMPERATURE

    normfactor_from = numerator_from * fluid.get_property("compressibility", p_abs_from) / p_abs_from
    normfactor_to = numerator_to * fluid.get_property("compressibility", p_abs_to) / p_abs_to
    normfactor_mean = numerator * fluid.get_property("compressibility", p_abs_mean) / p_abs_mean

    v_gas_from = v_mps * normfactor_from
    v_gas_to = v_mps * normfactor_to
    v_gas_mean = v_mps * normfactor_mean

    return v_gas_from, v_gas_to, v_gas_mean, p_abs_from, p_abs_to, p_abs_mean, normfactor_from, \
        normfactor_to, normfactor_mean


def get_branch_results_gas_numba(net, branch_pit, node_pit, from_nodes, to_nodes, v_mps, p_from,
                                 p_to):
    p_abs_from, p_abs_to, p_abs_mean = get_pressures_numba(node_pit, from_nodes, to_nodes, v_mps,
                                                           p_from, p_to)

    fluid = get_fluid(net)
    comp_from = fluid.get_property("compressibility", p_abs_from)
    comp_to = fluid.get_property("compressibility", p_abs_to)
    comp_mean = fluid.get_property("compressibility", p_abs_mean)

    v_gas_from, v_gas_to, v_gas_mean, normfactor_from, normfactor_to, normfactor_mean = \
        get_gas_vel_numba(node_pit, branch_pit, comp_from, comp_to, comp_mean, p_abs_from, p_abs_to,
                          p_abs_mean, v_mps)

    return v_gas_from, v_gas_to, v_gas_mean, p_abs_from, p_abs_to, p_abs_mean, normfactor_from, \
        normfactor_to, normfactor_mean


@jit(nopython=True)
def get_pressures_numba(node_pit, from_nodes, to_nodes, v_mps, p_from, p_to):
    p_abs_from, p_abs_to, p_abs_mean = [np.empty_like(v_mps) for _ in range(3)]

    for i in range(len(v_mps)):
        p_abs_from[i] = node_pit[from_nodes[i], PAMB] + p_from[i]
        p_abs_to[i] = node_pit[to_nodes[i], PAMB] + p_to[i]
        if np.less_equal(np.abs(p_abs_from[i] - p_abs_to[i]), 1e-8 + 1e-5 * abs(p_abs_to[i])):
            p_abs_mean[i] = p_abs_from[i]
        else:
            p_abs_mean[i] = np.divide(2 * (p_abs_from[i] ** 3 - p_abs_to[i] ** 3),
                                      3 * (p_abs_from[i] ** 2 - p_abs_to[i] ** 2))

    return p_abs_from, p_abs_to, p_abs_mean


@jit(nopython=True)
def get_gas_vel_numba(node_pit, branch_pit, comp_from, comp_to, comp_mean, p_abs_from, p_abs_to,
                      p_abs_mean, v_mps):
    v_gas_from, v_gas_to, v_gas_mean, normfactor_from, normfactor_to, normfactor_mean = \
        [np.empty_like(v_mps) for _ in range(6)]
    from_nodes = branch_pit[:, FROM_NODE].astype(np.int32)
    for i in range(len(v_mps)):
        t_from = node_pit[from_nodes[i], TINIT_NODE]
        t_to = branch_pit[i, TOUTINIT]
        tm = (t_from + t_to) / 2
        numerator_from = np.divide(NORMAL_PRESSURE * t_from, NORMAL_TEMPERATURE)
        numerator_to = np.divide(NORMAL_PRESSURE * t_to, NORMAL_TEMPERATURE)
        numerator = np.divide(NORMAL_PRESSURE * tm, NORMAL_TEMPERATURE)
        normfactor_from[i] = np.divide(numerator_from * comp_from[i], p_abs_from[i])
        normfactor_to[i] = np.divide(numerator_to * comp_to[i], p_abs_to[i])
        normfactor_mean[i] = np.divide(numerator * comp_mean[i], p_abs_mean[i])
        v_gas_from[i] = v_mps[i] * normfactor_from[i]
        v_gas_to[i] = v_mps[i] * normfactor_to[i]
        v_gas_mean[i] = v_mps[i] * normfactor_mean[i]

    return v_gas_from, v_gas_to, v_gas_mean, normfactor_from, normfactor_to, normfactor_mean


def extract_branch_results_with_internals(net, branch_results, table_name,
                                          res_nodes_from_hydraulics, res_nodes_from_heat,
                                          res_nodes_to_hydraulics, res_nodes_to_heat,
                                          res_mean_hydraulics, res_branch_ht, res_mean_heat, node_name,
                                          simulation_mode):
    # the result table to write results to
    res_table = net["res_" + table_name]

    # lookup for the component calling this function (where in branch_pit are entries for this
    # table?)
    f, t = get_lookup(net, "branch", "from_to")[table_name]

    branch_pit = net["_pit"]["branch"]
    # since the function _sum_by_group sorts the entries by an index (in this case the index of the
    # respective table), the placement of the indices mus be known to allocate the values correctly
    placement_table = np.argsort(net[table_name].index.values)
    idx_pit = branch_pit[f:t, ELEMENT_IDX]

    node_pit = net["_pit"]["node"]

    # the id of the external node table inside the node_pit (mostly this is "junction": 0)
    ext_node_tbl_idx = get_table_number(get_lookup(net, "node", "table"), node_name)

    for (result_mode, res_nodes_from, res_nodes_to, res_mean, res_branch) in [
        ("hydraulics", res_nodes_from_hydraulics, res_nodes_to_hydraulics, res_mean_hydraulics, []),
        ("heat", res_nodes_from_heat, res_nodes_to_heat, res_mean_heat, res_branch_ht)
    ]:
        if result_mode == "hydraulics" and simulation_mode == "heat":
            continue
        lookup_name = "hydraulics"
        if result_mode == "heat" and simulation_mode in ["heat", "sequential", "bidirectional"]:
            lookup_name = "heat_transfer"
        comp_connected = get_lookup(net, "branch", "active_" + lookup_name)[f:t]
        for (res_ext, node_name) in ((res_nodes_from, "from_nodes"), (res_nodes_to, "to_nodes")):
            if len(res_ext) == 0:
                continue
            # results that relate to the from_node --> in case of many internal nodes, only the
            # single from_node that is the exterior node (e.g. junction vs. internal pipe_node)
            # result has to be extracted from the node_pit
            end_nodes = branch_results[node_name][f:t]
            end_nodes_external = node_pit[end_nodes, TABLE_IDX_NODE] == ext_node_tbl_idx
            considered = end_nodes_external & comp_connected
            external_active = comp_connected[end_nodes_external]
            for res_name, entry in res_ext:
                res_table[res_name].values[external_active] = branch_results[entry][f:t][considered]
        if len(res_mean) > 0:
            # results that relate to the whole branch and shall be averaged (by summing up all
            # values and dividing by number of internal sections)
            use_numba = get_net_option(net, "use_numba")
            res = _sum_by_group(use_numba, idx_pit, np.ones_like(idx_pit),
                                comp_connected.astype(np.int32),
                                *[branch_results[rn[1]][f:t] for rn in res_mean])
            connected_ind = res[2] > 0.99
            num_internals = res[1][connected_ind]

            # hint: idx_pit[placement_table] should result in the indices as ordered in the table
            pt = placement_table[connected_ind]

            for i, (res_name, entry) in enumerate(res_mean_hydraulics):
                res_table[res_name].values[pt] = res[i + 3][connected_ind] / num_internals
        if len(res_branch) > 0:
            use_numba = get_net_option(net, "use_numba")
            _, sections, connected_sum = _sum_by_group(use_numba, idx_pit, np.ones_like(idx_pit),
                                comp_connected.astype(np.int32))
            connected_ind = connected_sum > 0.99
            indices_last_section = (np.cumsum(sections) - 1).astype(int)[connected_ind]
            # hint: idx_pit[placement_table] should result in the indices as ordered in the table
            pt = placement_table[connected_ind]

            for i, (res_name, entry) in enumerate(res_branch):
                res_table[res_name].values[pt] = branch_results[entry][indices_last_section]


def extract_branch_results_without_internals(net, branch_results, required_results_hydraulic,
                                             required_results_heat, table_name, simulation_mode):
    """
    Extract the results from the branch result array derived from the pit to the result table of the
    net (only for branch components without internal nodes). Here, we need to consider which results
    exist for hydraulic calculation and for heat transfer calculation (wrt. connectivity).

    :param net: The pandapipes net that the internal structure belongs to
    :type net: pandapipesNet
    :param branch_results: Important branch results from the internal pit structure
    :type branch_results: dict[np.ndarray]
    :param required_results_hydraulic: The entries that should be extracted for the respective \
        component for hydraulic calculation
    :type required_results_hydraulic: list[tuple]
    :param required_results_heat: The entries that should be extracted for the respective \
        component for heat transfer calculation
    :type required_results_heat: list[tuple]
    :param table_name: The name of the table that the results should be written to
    :type table_name: str
    :param simulation_mode: simulation mode (e.g. "hydraulics", "heat", "sequential", "bidirectional"); defines whether results from \
        hydraulic or temperature calculation are transferred
    :type simulation_mode: str
    :return: No output
    :rtype: None
    """
    res_table = net["res_" + table_name]
    f, t = get_lookup(net, "branch", "from_to")[table_name]

    # extract hydraulic results
    if simulation_mode in ["hydraulics", 'sequential', "bidirectional"]:
        # lookup for connected branch elements (hydraulic results)
        comp_connected_hyd = get_lookup(net, "branch", "active_hydraulics")[f:t]
        for res_name, entry in required_results_hydraulic:
            res_table[res_name].values[:][comp_connected_hyd] = \
                branch_results[entry][f:t][comp_connected_hyd]
        if simulation_mode == "hydraulics":
            for res_name, entry in required_results_heat:
                res_table[res_name].values[:][comp_connected_hyd] = \
                    branch_results[entry][f:t][comp_connected_hyd]

    # extract heat transfer results
    if simulation_mode in ["heat", 'sequential', "bidirectional"]:
        # lookup for connected branch elements (heat transfer results)
        comp_connected_ht = get_lookup(net, "branch", "active_heat_transfer")[f:t]
        for res_name, entry in required_results_heat:
            res_table[res_name].values[:][comp_connected_ht] = \
                branch_results[entry][f:t][comp_connected_ht]


def extract_results_active_pit(net, mode="hydraulics"):
    """
    Extract the pipeflow results from the internal pit structure ("_active_pit") to the general pit
    structure.

    :param net: The pandapipes net that the internal structure belongs to
    :type net: pandapipesNet
    :param mode: defines whether results from hydraulic or temperature calculation are transferred
    :type mode: str, default "hydraulics"
    :return: No output

    """
    nodes_connected = get_lookup(net, "node", "active_" + mode)
    branches_connected = get_lookup(net, "branch", "active_" + mode)
    result_node_col = PINIT if mode == "hydraulics" else TINIT_NODE
    not_affected_node_col = TINIT_NODE if mode == "hydraulics" else PINIT
    copied_node_cols = np.array([i for i in range(net["_pit"]["node"].shape[1])
                                 if i not in [not_affected_node_col]])
    rows_nodes = np.arange(net["_pit"]["node"].shape[0])[nodes_connected]

    result_branch_col = MDOTINIT if mode == "hydraulics" else TOUTINIT
    not_affected_branch_col = TOUTINIT if mode == "hydraulics" else MDOTINIT
    copied_branch_cols = np.array([i for i in range(net["_pit"]["branch"].shape[1])
                                   if i not in [FROM_NODE, TO_NODE,
                                                not_affected_branch_col]])
    rows_branches = np.arange(net["_pit"]["branch"].shape[0])[branches_connected]

    amb = get_net_option(net, 'ambient_temperature')

    net["_pit"]["node"][~nodes_connected, result_node_col] = np.nan if mode == "hydraulics" else amb
    net["_pit"]["node"][rows_nodes[:, np.newaxis], copied_node_cols[np.newaxis, :]] = \
        net["_active_pit"]["node"][:, copied_node_cols]
    net["_pit"]["branch"][~branches_connected, result_branch_col] = np.nan if mode == "hydraulics" else \
        net["_pit"]["branch"][~branches_connected, TEXT]
    net["_pit"]["branch"][rows_branches[:, np.newaxis], copied_branch_cols[np.newaxis, :]] = \
        net["_active_pit"]["branch"][:, copied_branch_cols]
