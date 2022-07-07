import numpy as np

from pandapipes.constants import NORMAL_PRESSURE, NORMAL_TEMPERATURE
from pandapipes.pf.internals_toolbox import _sum_by_group
from pandapipes.pf.pipeflow_setup import get_table_number, get_lookup, get_net_option
from pandapipes.properties.fluids import get_fluid, get_mixture_density, is_fluid_gas, get_mixture_compressibility

try:
    from numba import jit
except ImportError:
    from pandapower.pf.no_numba import jit


def extract_all_results(net, nodes_connected, branches_connected):
    """
    Extract results from branch pit and node pit and write them to the different tables of the net,\
    as defined by the component models.

    :param net: pandapipes net for which to extract results into net.res_xy
    :type net: pandapipesNet
    :return: No output

    """
    branch_pit = net["_pit"]["branch"]
    node_pit = net["_pit"]["node"]
    v_mps, mf, vf, from_nodes, to_nodes, temp_from, temp_to, reynolds, _lambda, p_from, p_to, pl = \
        get_basic_branch_results(net, branch_pit, node_pit)
    branch_results = {"v_mps": v_mps, "mf_from": mf, "mf_to": -mf, "vf": vf, "p_from": p_from,
                      "p_to": p_to, "from_nodes": from_nodes, "to_nodes": to_nodes,
                      "temp_from": temp_from, "temp_to": temp_to, "reynolds": reynolds,
                      "lambda": _lambda, "pl": pl}
    if is_fluid_gas(net):
        if get_net_option(net, "use_numba"):
            v_gas_from, v_gas_to, v_gas_mean, p_abs_from, p_abs_to, p_abs_mean, normfactor_from, \
            normfactor_to, normfactor_mean = get_branch_results_gas_numba(
                net, branch_pit, node_pit, from_nodes, to_nodes, v_mps, p_from, p_to)
        else:
            v_gas_from, v_gas_to, v_gas_mean, p_abs_from, p_abs_to, p_abs_mean, normfactor_from, \
            normfactor_to, normfactor_mean = get_branch_results_gas(
                net, branch_pit, node_pit, from_nodes, to_nodes, v_mps, p_from, p_to)
        gas_branch_results = {
            "v_gas_from": v_gas_from, "v_gas_to": v_gas_to, "v_gas_mean": v_gas_mean,
            "p_from": p_from, "p_to": p_to, "p_abs_from": p_abs_from, "p_abs_to": p_abs_to,
            "p_abs_mean": p_abs_mean, "normfactor_from": normfactor_from,
            "normfactor_to": normfactor_to, "normfactor_mean": normfactor_mean
        }
        branch_results.update(gas_branch_results)
    for comp in np.concatenate([net['node_element_list'], net['node_list'], net['branch_list']]):
        comp.extract_results(net, net["_options"], branch_results, nodes_connected,
                             branches_connected)


def get_basic_branch_results(net, branch_pit, node_pit):
    from_nodes = branch_pit[:, net['_idx_branch']['FROM_NODE']].astype(np.int32)
    to_nodes = branch_pit[:, net['_idx_branch']['TO_NODE']].astype(np.int32)

    t0 = node_pit[from_nodes, net['_idx_node']['TINIT']]
    t1 = node_pit[to_nodes, net['_idx_node']['TINIT']]
    mf = branch_pit[:, net['_idx_branch']['LOAD_VEC_NODES']]
    if len(net._fluid) == 1:
        density = get_fluid(net, net._fluid[0]).get_density((t0 + t1) / 2)
    else:
        w = get_lookup(net, 'branch', 'w')
        mass_fract = branch_pit[:, w]
        density = get_mixture_density(net, (t0 + t1) / 2, mass_fraction=mass_fract)
    vf = mf / density
    return branch_pit[:, net['_idx_branch']['VINIT']], mf, vf, from_nodes, to_nodes, t0, t1, \
           branch_pit[:, net['_idx_branch']['RE']], \
           branch_pit[:, net['_idx_branch']['LAMBDA']], \
           node_pit[from_nodes, net['_idx_node']['PINIT']], \
           node_pit[to_nodes, net['_idx_node']['PINIT']], \
           branch_pit[:, net['_idx_branch']['PL']]


def get_branch_results_gas(net, branch_pit, node_pit, from_nodes, to_nodes, v_mps, p_from, p_to):
    p_abs_from = node_pit[from_nodes, net['_idx_node']['PAMB']] + p_from
    p_abs_to = node_pit[to_nodes, net['_idx_node']['PAMB']] + p_to
    mask = ~np.isclose(p_abs_from, p_abs_to)
    p_abs_mean = np.empty_like(p_abs_to)
    p_abs_mean[~mask] = p_abs_from[~mask]
    p_abs_mean[mask] = 2 / 3 * (p_abs_from[mask] ** 3 - p_abs_to[mask] ** 3) \
                       / (p_abs_from[mask] ** 2 - p_abs_to[mask] ** 2)
    numerator = NORMAL_PRESSURE * branch_pit[:, net['_idx_branch']['TINIT']] / NORMAL_TEMPERATURE
    if len(net._fluid) == 1:
        normfactor_from = numerator * get_fluid(net, net._fluid[0]).get_property("compressibility", p_abs_from) / p_abs_from
        normfactor_to = numerator * get_fluid(net, net._fluid[0]).get_property("compressibility", p_abs_to) / p_abs_to
        normfactor_mean = numerator * get_fluid(net, net._fluid[0]).get_property("compressibility", p_abs_mean) / p_abs_mean
    else:
        w = get_lookup(net, 'branch', 'w')
        mass_fract = branch_pit[:, w]
        normfactor_from = numerator * get_mixture_compressibility(net, p_abs_from, mass_fract) / p_abs_from
        normfactor_to = numerator * get_mixture_compressibility(net, p_abs_to, mass_fract) / p_abs_to
        normfactor_mean = numerator * get_mixture_compressibility(net, p_abs_mean, mass_fract) / p_abs_mean

    v_gas_from = v_mps * normfactor_from
    v_gas_to = v_mps * normfactor_to
    v_gas_mean = v_mps * normfactor_mean

    return v_gas_from, v_gas_to, v_gas_mean, p_abs_from, p_abs_to, p_abs_mean, normfactor_from, \
           normfactor_to, normfactor_mean


def get_branch_results_gas_numba(net, branch_pit, node_pit, from_nodes, to_nodes, v_mps, p_from,
                                 p_to):
    pit_cols = [net['_idx_branch']['TINIT'], net['_idx_node']['PAMB']]

    p_abs_from, p_abs_to, p_abs_mean = get_pressures_numba(pit_cols[1], node_pit, from_nodes, to_nodes, v_mps,
                                                           p_from, p_to)
    if len(net._fluid) == 1:
        comp_from = get_fluid(net, net._fluid[0]).get_property("compressibility", p_abs_from)
        comp_to = get_fluid(net, net._fluid[0]).get_property("compressibility", p_abs_to)
        comp_mean = get_fluid(net, net._fluid[0]).get_property("compressibility", p_abs_mean)
    else:
        w = get_lookup(net, 'branch', 'w')
        mass_fract = branch_pit[:, w]
        comp_from = get_mixture_compressibility(net, p_abs_from, mass_fract)
        comp_to = get_mixture_compressibility(net, p_abs_to, mass_fract)
        comp_mean = get_mixture_compressibility(net, p_abs_mean, mass_fract)

    v_gas_from, v_gas_to, v_gas_mean, normfactor_from, normfactor_to, normfactor_mean = \
        get_gas_vel_numba(pit_cols[0], branch_pit, comp_from, comp_to, comp_mean, p_abs_from, p_abs_to,
                          p_abs_mean, v_mps)

    return v_gas_from, v_gas_to, v_gas_mean, p_abs_from, p_abs_to, p_abs_mean, normfactor_from, \
           normfactor_to, normfactor_mean


@jit(nopython=True)
def get_pressures_numba(pit_cols, node_pit, from_nodes, to_nodes, v_mps, p_from, p_to):
    p_abs_from, p_abs_to, p_abs_mean = [np.empty_like(v_mps) for _ in range(3)]

    for i in range(len(v_mps)):
        p_abs_from[i] = node_pit[from_nodes[i], pit_cols] + p_from[i]
        p_abs_to[i] = node_pit[to_nodes[i], pit_cols] + p_to[i]
        if np.less_equal(np.abs(p_abs_from[i] - p_abs_to[i]), 1e-8 + 1e-5 * abs(p_abs_to[i])):
            p_abs_mean[i] = p_abs_from[i]
        else:
            p_abs_mean[i] = np.divide(2 * (p_abs_from[i] ** 3 - p_abs_to[i] ** 3),
                                      3 * (p_abs_from[i] ** 2 - p_abs_to[i] ** 2))

    return p_abs_from, p_abs_to, p_abs_mean


@jit(nopython=True)
def get_gas_vel_numba(pit_cols, branch_pit, comp_from, comp_to, comp_mean, p_abs_from, p_abs_to, p_abs_mean,
                      v_mps):
    v_gas_from, v_gas_to, v_gas_mean, normfactor_from, normfactor_to, normfactor_mean = \
        [np.empty_like(v_mps) for _ in range(6)]

    for i in range(len(v_mps)):
        numerator = np.divide(NORMAL_PRESSURE * branch_pit[i, pit_cols], NORMAL_TEMPERATURE)
        normfactor_from[i] = np.divide(numerator * comp_from[i], p_abs_from[i])
        normfactor_to[i] = np.divide(numerator * comp_to[i], p_abs_to[i])
        normfactor_mean[i] = np.divide(numerator * comp_mean[i], p_abs_mean[i])
        v_gas_from[i] = v_mps[i] * normfactor_from[i]
        v_gas_to[i] = v_mps[i] * normfactor_to[i]
        v_gas_mean[i] = v_mps[i] * normfactor_mean[i]

    return v_gas_from, v_gas_to, v_gas_mean, normfactor_from, normfactor_to, normfactor_mean


def extract_branch_results_with_internals(net, branch_results, table_name, res_nodes_from,
                                          res_nodes_to, res_mean, node_name, branches_connected):
    # the result table to write results to
    res_table = net["res_" + table_name]

    # lookup for the component calling this function (where in branch_pit are entries for this
    # table?)
    f, t = get_lookup(net, "branch", "from_to")[table_name]

    branch_pit = net["_pit"]["branch"]
    # since the function _sum_by_group sorts the entries by an index (in this case the index of the
    # respective table), the placement of the indices mus be known to allocate the values correctly
    placement_table = np.argsort(net[table_name].index.values)
    idx_pit = branch_pit[f:t, net['_idx_branch']['ELEMENT_IDX']]
    comp_connected = branches_connected[f:t]

    node_pit = net["_pit"]["node"]

    # the id of the external node table inside the node_pit (mostly this is "junction": 0)
    ext_node_tbl_idx = get_table_number(get_lookup(net, "node", "table"), node_name)

    if len(res_nodes_from) > 0:
        # results that relate to the from_node --> in case of many internal nodes, only the single
        # from_node that is the exterior node (e.g. junction vs. internal pipe_node) result has to
        # be extracted from the node_pit
        from_nodes = branch_results["from_nodes"][f:t]
        from_nodes_external = node_pit[from_nodes, net['_idx_node']['TABLE_IDX']] == ext_node_tbl_idx
        considered = from_nodes_external & comp_connected
        external_active = comp_connected[from_nodes_external]
        for res_name, entry in res_nodes_from:
            res_table[res_name].values[external_active] = branch_results[entry][f:t][considered]

    if len(res_nodes_to) > 0:
        # results that relate to the to_node --> in case of many internal nodes, only the single
        # to_node that is the exterior node (e.g. junction vs. internal pipe_node) result has to
        # be extracted from the node_pit
        to_nodes = branch_results["to_nodes"][f:t]
        to_nodes_external = node_pit[to_nodes, net['_idx_node']['TABLE_IDX']] == ext_node_tbl_idx
        considered = to_nodes_external & comp_connected
        external_active = comp_connected[to_nodes_external]
        for res_name, entry in res_nodes_to:
            res_table[res_name].values[external_active] = branch_results[entry][f:t][considered]

    if len(res_mean) > 0:
        # results that relate to the whole branch and shall be averaged (by summing up all values
        # and dividing by number of internal sections)
        use_numba = get_net_option(net, "use_numba")
        res = _sum_by_group(use_numba, idx_pit, np.ones_like(idx_pit),
                            comp_connected.astype(np.int32),
                            *[branch_results[rn[1]][f:t] for rn in res_mean])
        connected_ind = res[2] > 0.99
        num_internals = res[1][connected_ind]
        # hint: idx_pit[placement_table] should result in the indices as ordered in the table
        placement_table = placement_table[connected_ind]

        for i, (res_name, entry) in enumerate(res_mean):
            res_table[res_name].values[placement_table] = res[i + 3][connected_ind] / num_internals


def extract_branch_results_without_internals(net, branch_results, required_results, table_name,
                                             branches_connected):
    res_table = net["res_" + table_name]
    f, t = get_lookup(net, "branch", "from_to")[table_name]
    comp_connected = branches_connected[f:t]

    for res_name, entry in required_results:
        res_table[res_name].values[:][comp_connected] = branch_results[entry][f:t][comp_connected]


def extract_results_active_pit(net, node_pit, branch_pit, nodes_connected, branches_connected):
    """
    Extract the pipeflow results from the internal pit structure ("_active_pit") to the general pit
    structure.

    :param net: The pandapipes net that the internal structure belongs to
    :type net: pandapipesNet
    :param node_pit: The internal structure node array
    :type node_pit: np.array
    :param branch_pit: The internal structure branch array
    :type branch_pit: np.array
    :param nodes_connected: A mask array stating which nodes are actually connected to the rest of\
            the net
    :type nodes_connected: np.array
    :param branches_connected: A mask array stating which branches are actually connected to the \
             rest of the net
    :type branches_connected: np.array
    :return: No output

    """
    all_nodes_connected = np.alltrue(nodes_connected)
    if not all_nodes_connected:
        node_pit[~nodes_connected, net['_idx_node']['PINIT']] = np.NaN
        node_pit[nodes_connected, :] = net["_active_pit"]["node"]
        cols_br = np.array([i for i in range(branch_pit.shape[1])
                            if i not in [net['_idx_branch']['FROM_NODE'],
                                         net['_idx_branch']['TO_NODE'],
                                         net['_idx_branch']['FROM_NODE_T'],
                                         net['_idx_branch']['TO_NODE_T']]])
    else:
        net["_pit"]["node"] = np.copy(net["_active_pit"]["node"])
        cols_br = None

    if not np.alltrue(branches_connected):
        branch_pit[~branches_connected, net['_idx_branch']['VINIT']] = np.NaN
        rows_active_br = np.where(branches_connected)[0]
        if all_nodes_connected:
            branch_pit[rows_active_br, :] = net["_active_pit"]["branch"][:, :]
        else:
            branch_pit[rows_active_br[:, np.newaxis], cols_br[np.newaxis, :]] = \
                net["_active_pit"]["branch"][:, cols_br]
    else:
        if all_nodes_connected:
            net["_pit"]["branch"] = np.copy(net["_active_pit"]["branch"])
        else:
            net["_pit"]["branch"][:, cols_br] = net["_active_pit"]["branch"][:, cols_br]
