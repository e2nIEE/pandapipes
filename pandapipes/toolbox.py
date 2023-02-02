# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
import copy
from collections.abc import Iterable

import numpy as np
import pandas as pd
from networkx import has_path
from pandapower.auxiliary import get_indices
from pandapower.toolbox import dataframes_equal, clear_result_tables

import pandapipes
from pandapipes.component_models.abstract_models.branch_models import BranchComponent
from pandapipes.component_models.abstract_models.node_element_models import NodeElementComponent
from pandapipes.create import create_empty_network
from pandapipes.idx_branch import TABLE_IDX as TABLE_IDX_BRANCH, ELEMENT_IDX as ELEMENT_IDX_BRANCH, \
    FROM_NODE as FROM_NODE_BRANCH, TO_NODE as TO_NODE_BRANCH, ACTIVE as ACTIVE_BRANCH, \
    LENGTH as LENGTH_BRANCH, D as D_BRANCH, AREA as AREA_BRANCH, RHO as RHO_BRANCH, \
    ETA as ETA_BRANCH, K as K_BRANCH, TINIT as TINIT_BRANCH, VINIT as VINIT_BRANCH, \
    RE as RE_BRANCH, LAMBDA as LAMBDA_BRANCH, JAC_DERIV_DV as JAC_DERIV_DV_BRANCH, \
    JAC_DERIV_DP as JAC_DERIV_DP_BRANCH, JAC_DERIV_DP1 as JAC_DERIV_DP1_BRANCH, \
    LOAD_VEC_BRANCHES as LOAD_VEC_BRANCHES_BRANCH, JAC_DERIV_DV_NODE as JAC_DERIV_DV_NODE_BRANCH, \
    LOAD_VEC_NODES as LOAD_VEC_NODES_BRANCH, LOSS_COEFFICIENT as LOSS_COEFFICIENT_BRANCH, \
    CP as CP_BRANCH, ALPHA as ALPHA_BRANCH, JAC_DERIV_DT as JAC_DERIV_DT_BRANCH, \
    JAC_DERIV_DT1 as JAC_DERIV_DT1_BRANCH, LOAD_VEC_BRANCHES_T as LOAD_VEC_BRANCHES_T_BRANCH, \
    T_OUT as T_OUT_BRANCH, JAC_DERIV_DT_NODE as JAC_DERIV_DT_NODE_BRANCH, \
    LOAD_VEC_NODES_T as LOAD_VEC_NODES_T_BRANCH, VINIT_T as VINIT_T_BRANCH, \
    FROM_NODE_T as FROM_NODE_T_BRANCH, TO_NODE_T as TO_NODE_T_BRANCH, QEXT as QEXT_BRANCH, \
    TEXT as TEXT_BRANCH, STD_TYPE as STD_TYPE_BRANCH, PL as PL_BRANCH, TL as TL_BRANCH, \
    BRANCH_TYPE as BRANCH_TYPE_BRANCH, PRESSURE_RATIO as PRESSURE_RATIO_BRANCH, branch_cols
from pandapipes.idx_node import TABLE_IDX as TABLE_IDX_NODE, ELEMENT_IDX as ELEMENT_IDX_NODE, \
    NODE_TYPE as NODE_TYPE_NODE, ACTIVE as ACTIVE_NODE, RHO as RHO_NODE, PINIT as PINIT_NODE, \
    LOAD as LOAD_NODE, HEIGHT as HEIGHT_NODE, TINIT as TINIT_NODE, PAMB as PAMB_NODE, \
    LOAD_T as LOAD_T_NODE, NODE_TYPE_T as NODE_TYPE_T_NODE, \
    EXT_GRID_OCCURENCE as EXT_GRID_OCCURENCE_NODE, \
    EXT_GRID_OCCURENCE_T as EXT_GRID_OCCURENCE_T_NODE, node_cols, \
    T as TYPE_T, P as TYPE_P, PC as TYPE_PC, NONE as TYPE_NONE, L as TYPE_L
from pandapipes.pandapipes_net import pandapipesNet
from pandapipes.topology import create_nxgraph

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def nets_equal(net1, net2, check_only_results=False, exclude_elms=None, **kwargs):
    """
    Compares the DataFrames of two networks.

    The networks are considered equal if they share the same keys and values, except of the 'et'
    (elapsed time) entry which differs depending on runtime conditions and entries stating with '_'.

    :param net1: first net for comparison
    :type net1: pandapipesNet
    :param net2: second net for comparison
    :type net2: pandapipesNet
    :param check_only_results:
    :type check_only_results: bool, default False
    :param exclude_elms: element types that should be skipped in the comparison
    :type exclude_elms: list of strings, default None
    :param kwargs: key word arguments
    :type kwargs:
    :return: True, if nets are equal
    :rtype: Bool
    """

    eq = isinstance(net1, pandapipesNet) and isinstance(net2, pandapipesNet)
    exclude_elms = [] if exclude_elms is None else list(exclude_elms)
    exclude_elms += ["res_" + ex for ex in exclude_elms]
    not_equal = []

    if eq:
        # for two networks make sure both have the same keys that do not start with "_"...
        net1_keys = [key for key in net1.keys() if not (key.startswith("_") or key in exclude_elms)]
        net2_keys = [key for key in net2.keys() if not (key.startswith("_") or key in exclude_elms)]
        keys_to_check = set(net1_keys) & set(net2_keys)
        key_difference = set(net1_keys) ^ set(net2_keys)

        if len(key_difference) > 0:
            logger.info("Networks entries mismatch at: %s" % key_difference)
            if not check_only_results:
                return False

        # ... and then iter through the keys, checking for equality for each table
        for df_name in list(keys_to_check):
            # skip 'et' (elapsed time) and entries starting with '_' (internal vars)
            if df_name != 'et' and not df_name.startswith("_"):
                if check_only_results and not df_name.startswith("res_"):
                    continue  # skip anything that is not a result table

                if isinstance(net1[df_name], pd.DataFrame) and isinstance(net2[df_name],
                                                                          pd.DataFrame):
                    frames_equal = dataframes_equal(net1[df_name], net2[df_name], **kwargs)
                    eq &= frames_equal

                    if not frames_equal:
                        not_equal.append(df_name)

    if len(not_equal) > 0:
        logger.info("Networks do not match in DataFrame(s): %s" % (', '.join(not_equal)))

    return eq


def element_junction_tuples(include_node_elements=True, include_branch_elements=True,
                            include_res_elements=False, net=None):
    """
    Utility function
    Provides the tuples of elements and corresponding columns for junctions they are connected to

    :param include_node_elements: whether tuples for junction elements e.g. sink, source, are \
           included
    :type include_node_elements: bool
    :param include_branch_elements: whether branch elements e.g. pipe, pumps, ... are included
    :type include_branch_elements: bool
    :param include_res_elements: whether to include result tables
    :type include_res_elements: bool
    :param net: pandapipes net from which to derive component names
    :type net: pandapipesNet
    :return: set of tuples with element names and column names
    :rtype: set
    """
    from pandapipes.component_models import Sink, Source, ExtGrid, Pipe, Valve, Pump, \
        CirculationPumpMass, CirculationPumpPressure, HeatExchanger, PressureControlComponent, \
        Compressor, FlowControlComponent
    from pandapipes.converter.stanet.valve_pipe_component import ValvePipe
    special_elements_junctions = [("press_control", "controlled_junction")]
    move_elements = {"n2b": [], "b2n": []}
    node_elements = []
    branch_elements = []
    if net is not None:
        all_tables = {comp.table_name(): comp for comp in net.component_list}
    else:
        comp_list = [Sink, Source, ExtGrid, Pipe, Valve, Pump, CirculationPumpMass,
                     CirculationPumpPressure, HeatExchanger, PressureControlComponent, Compressor,
                     FlowControlComponent, ValvePipe]
        all_tables = {comp.table_name(): comp for comp in comp_list}

    ejts = set()
    if include_node_elements:
        node_elements = [tbl for tbl, comp in all_tables.items() if
                         issubclass(comp, NodeElementComponent) and tbl not in move_elements["n2b"]]
        node_elements += [me for me in move_elements["b2n"] if me in all_tables.keys()]
        for elm in node_elements:
            ejts.update([(elm, "junction")])

    if include_branch_elements:
        branch_elements = [(tbl, comp.from_to_node_cols()) for tbl, comp in all_tables.items()
                           if issubclass(comp, BranchComponent) and tbl not in move_elements["b2n"]]
        branch_elements += [(me, all_tables[me].from_to_node_cols())
                            for me in move_elements["n2b"] if me in all_tables.keys()]
        for elm, from_to_cols in branch_elements:
            ejts.update([(elm, from_to_cols[0]), (elm, from_to_cols[1])])

    branch_tables = [br[0] for br in branch_elements]
    if include_res_elements:
        if net is not None:
            elements_without_res = [elm for elm in node_elements + branch_tables
                                    if "res_" + elm not in net]
        else:
            elements_without_res = []
        ejts.update(
            [("res_" + ejt[0], ejt[1]) for ejt in ejts if ejt[0] not in elements_without_res])
    ejts.update((el, jn) for el, jn in special_elements_junctions if el in node_elements
                or el in branch_tables)
    return ejts


def pp_elements(junction=True, include_node_elements=True, include_branch_elements=True,
                include_res_elements=False, net=None):
    """
    Provides a list of all pandapipes elements belonging to the desired element types. If a net is
    given, the elements are derived from the component list.

    :param junction: if True, return junction table name
    :type junction: bool, default True
    :param include_node_elements: if True, return node element table names
    :type include_node_elements: bool, default True
    :param include_branch_elements: if True, return branch element table names
    :type include_branch_elements: bool, default True
    :param include_res_elements: if True, return result table names for all the elements
    :type include_res_elements: bool, default False
    :param net: if a pandapipes network is given, the table names will be derived from its \
            component list
    :type net: pandapipesNet
    :return: pp_elms - set of table names for the desired element types
    :rtype: set
    """

    pp_elms = {"junction"} if junction else set()
    pp_elms |= set([el[0] for el in element_junction_tuples(
        include_node_elements, include_branch_elements, include_res_elements, net)])
    return pp_elms


def reindex_junctions(net, junction_lookup):
    """
    Changes the index of net.junction and considers the new junction indices in all other
    pandapipes element tables.

    :param net: pandapipes network
    :type net: pandapipesNet
    :param junction_lookup: the keys are the old junction indices, the values the new junction \
            indices
    :type junction_lookup: dict
    :return: junction_lookup - the finally reindexed junction lookup (with corrections if necessary)
    :rtype: dict
    """

    not_fitting_junction_lookup_keys = set(junction_lookup.keys()) - set(net.junction.index)
    if len(not_fitting_junction_lookup_keys):
        logger.error("These junction indices are unknown to net. Thus, they cannot be reindexed: " +
                     str(not_fitting_junction_lookup_keys))

    missing_junction_indices = sorted(set(net.junction.index) - set(junction_lookup.keys()))
    if len(missing_junction_indices):
        duplicates = set(missing_junction_indices).intersection(set(junction_lookup.values()))
        if len(duplicates):
            logger.error("The junctions %s are not listed in junction_lookup but their index is "
                         "used as a new index. This would result in duplicated junction indices."
                         % (str(duplicates)))
        else:
            junction_lookup.update({j: j for j in missing_junction_indices})

    net.junction.index = get_indices(net.junction.index, junction_lookup)
    if hasattr(net, "res_junction"):
        net.res_junction.index = get_indices(net.res_junction.index, junction_lookup)

    for element, value in element_junction_tuples(net=net):
        if element in net.keys():
            net[element][value] = get_indices(net[element][value], junction_lookup)
    net["junction_geodata"].set_index(get_indices(net["junction_geodata"].index, junction_lookup),
                                      inplace=True)
    return junction_lookup


def reindex_elements(net, element, new_indices, old_indices=None):
    """
    Changes the index of net[element].

    :param net: pandapipes network
    :type net: pandapipesNet
    :param element: name of the element table
    :type element: str
    :param new_indices: list of new indices
    :type new_indices: iterable
    :param old_indices: list of old/previous indices which will be replaced. If None, all indices\
            are considered.
    :type old_indices: iterable, default None
    :return: No output.
    """

    old_indices = old_indices if old_indices is not None else net[element].index
    if not len(new_indices) or not net[element].shape[0]:
        return
    if len(new_indices) != len(old_indices):
        raise UserWarning("The length of new indices to replace existing ones for %s does not "
                          "match: %d (new) vs. %d (old)."
                          % (element, len(new_indices), len(old_indices)))
    lookup = dict(zip(old_indices, new_indices))

    if element == "junction":
        reindex_junctions(net, lookup)
        return

    # --- reindex
    net[element]["index"] = net[element].index
    net[element].loc[old_indices, "index"] = get_indices(old_indices, lookup)
    net[element].set_index("index", inplace=True)

    # --- adapt geodata index
    geotable = element + "_geodata"
    if geotable in net and net[geotable].shape[0]:
        net[geotable]["index"] = net[geotable].index
        net[geotable].loc[old_indices, "index"] = get_indices(old_indices, lookup)
        net[geotable].set_index("index", inplace=True)


def create_continuous_junction_index(net, start=0, store_old_index=False):
    """
    Creates a continuous junction index starting at 'start' and replaces all
    references of old indices by the new ones.

    :param net: pandapipes network
    :type net: pandapipesNet
    :param start: index begins with "start"
    :type start: int, default 0
    :param store_old_index: if True, stores the old index in net.junction["old_index"]
    :type store_old_index: bool, default False
    :return: junction_lookup - mapping of old to new index
    :rtype: dict
    """
    net.junction.sort_index(inplace=True)
    if store_old_index:
        net.junction["old_index"] = net.junction.index.values
    new_junction_idxs = list(np.arange(start, len(net.junction) + start))
    junction_lookup = dict(zip(net["junction"].index.values, new_junction_idxs))
    reindex_junctions(net, junction_lookup)
    return junction_lookup


def create_continuous_elements_index(net, start=0, add_df_to_reindex=None):
    """
    Creating a continuous index for all the elements, starting at zero and replaces all references
    of old indices by the new ones.

    :param net: pandapipes network with unodered indices
    :type net: pandapipesNet
    :param start: new index begins with "start"
    :type start: int
    :param add_df_to_reindex: by default all useful pandapower elements for power flow will be\
          selected. Customized DataFrames can also be considered here.
    :type add_df_to_reindex: iterable, default None
    :return: net - pandapipes network with odered and continuous indices
    :rtype: pandapipesNet
    """
    add_df_to_reindex = set() if add_df_to_reindex is None else set(add_df_to_reindex)
    elements = pp_elements(include_res_elements=True, net=net)

    # create continuous junction index
    create_continuous_junction_index(net, start=start)
    elements -= {"junction", "junction_geodata", "res_junction"}

    elements |= add_df_to_reindex

    # run reindex_elements() for all elements
    for elm in list(elements):
        net[elm].sort_index(inplace=True)
        new_index = list(np.arange(start, len(net[elm]) + start))

        if elm in net and isinstance(net[elm], pd.DataFrame):
            if elm in ["junction_geodata", "pipe_geodata"]:
                logger.info(elm + " don't need to bo included to 'add_df_to_reindex'. It is " +
                            "already included by elm=='" + elm.split("_")[0] + "'.")
            else:
                reindex_elements(net, elm, new_index)
        else:
            logger.debug("No indices could be changed for element '%s'." % elm)

    return net


def fuse_junctions(net, j1, j2, drop=True):
    """
    Reroutes any connections to junctions in j2 to the given junction j1. Additionally drops the
    junctions j2, if drop=True (default).

    :param net: pandapipes network
    :type net: pandapipesNet
    :param j1: junction into which to fuse the other junction(s)
    :type j1: int
    :param j2: junction(s) that shall be fused into junction 1
    :type j2: Iterable or int
    :param drop: if True, junction(s) j2 will be dropped after fusing all elements
    :type drop: boolean, default True
    :return: net - the new pandapipes network
    :rtype: pandapipesNet
    """
    j2 = set(j2) - {j1} if isinstance(j2, Iterable) else [j2]

    for element, value in element_junction_tuples(net=net):
        i = net[element][net[element][value].isin(j2)].index
        net[element].loc[i, value] = j1

    if drop:
        # drop_elements=False because the elements must be connected to new junctions now
        drop_junctions(net, j2, drop_elements=False)
        # TODO: If junctions j1 and j2 are connected by one branch element, it should be dropped.
    return net


def select_subnet(net, junctions, include_results=False, keep_everything_else=False,
                  remove_internals=True, remove_unused_components=False):
    """
    Selects a subnet by a list of junction indices and returns a net with all components connected
    to them.
    """
    junctions = list(junctions)

    if keep_everything_else:
        p2 = copy.deepcopy(net)
        if not include_results:
            clear_result_tables(p2)
        if remove_internals:
            for inter in [k for k in p2.keys() if k.startswith("_")]:
                p2.pop(inter)
    else:
        p2 = create_empty_network(add_stdtypes=False)
        p2["std_types"] = copy.deepcopy(net["std_types"])
        net_parameters = ["name", "fluid", "user_pf_options", "component_list"]
        for net_parameter in net_parameters:
            if net_parameter in net.keys():
                p2[net_parameter] = copy.deepcopy(net[net_parameter])

    p2.junction = net.junction.loc[junctions]
    comp_tuples = element_junction_tuples(include_node_elements=True, include_branch_elements=True,
                                          net=net)
    comp_junc_rows = {tbl: [jr for el, jr in comp_tuples if el == tbl] for tbl in
                      set([v[0] for v in comp_tuples])}
    for comp_tbl, junc_rows in comp_junc_rows.items():
        isin_all = np.all([net[comp_tbl][jr].isin(junctions) for jr in junc_rows], axis=0)
        p2[comp_tbl] = net[comp_tbl][isin_all]

    if include_results:
        for table in net.keys():
            if net[table] is None or not isinstance(net[table], pd.DataFrame) or not \
               net[table].shape[0] or not table.startswith("res_") or table[4:] not in \
               net.keys() or not isinstance(net[table[4:]], pd.DataFrame) or not \
               net[table[4:]].shape[0]:
                continue
            elif table == "res_junction":
                p2[table] = net[table].loc[net[table].index.intersection(junctions)]
            else:
                p2[table] = net[table].loc[p2[table[4:]].index.intersection(net[table].index)]
    if "junction_geodata" in net:
        p2["junction_geodata"] = net.junction_geodata.loc[p2.junction.index.intersection(
            net.junction_geodata.index)]
    if "pipe_geodata" in net:
        p2["pipe_geodata"] = net.pipe_geodata.loc[p2.pipe.index.intersection(
            net.pipe_geodata.index)]

    if remove_unused_components:
        remove_empty_components(p2)

    return pandapipesNet(p2)


def remove_empty_components(net):
    removed = set()
    for comp in net.component_list:
        if net[comp.table_name()].empty:
            del net[comp.table_name()]
            removed.add(comp)
    net.component_list = [c for c in net.component_list if c not in removed]


def drop_junctions(net, junctions, drop_elements=True):
    """
    Drops specified junctions, their junction_geodata and by default drops all elements connected to
    them as well.

    :param net: pandapipes network
    :type net: pandapipesNet
    :param junctions: junctions to drop
    :type junctions: Iterable
    :param drop_elements: if True, all elements connected to the junction will be dropped as well
    :type drop_elements: bool, default True
    :return: No output.
    """
    net["junction"].drop(junctions, inplace=True)
    net["junction_geodata"].drop(set(junctions) & set(net["junction_geodata"].index), inplace=True)
    if "res_junction" in net.keys():
        res_junctions = net.res_junction.index.intersection(junctions)
        net["res_junction"].drop(res_junctions, inplace=True)
    if drop_elements:
        drop_elements_at_junctions(net, junctions)


def drop_elements_at_junctions(net, junctions, node_elements=True, branch_elements=True):
    """
    drop elements connected to given junctions

    :param net: pandapipes network
    :type net: pandapipesNet
    :param junctions: junctions from which to remove all elements
    :type junctions: Iterable
    :param node_elements: flag stating if node elements (such as sinks or sources) shall be dropped
    :type node_elements: bool, default True
    :param branch_elements: flag stating if branch elements (such as pipes or valves) shall be \
            dropped
    :type branch_elements: bool, default True
    :return: No output.
    """
    for element, column in element_junction_tuples(node_elements, branch_elements,
                                                   include_res_elements=False, net=net):
        if any(net[element][column].isin(junctions)):
            eid = net[element][net[element][column].isin(junctions)].index
            if element == 'pipe':
                drop_pipes(net, eid)
            # elif element == 'trafo' or element == 'trafo3w':
            #     drop_trafos(net, eid, table=element)
            else:
                n_el = net[element].shape[0]
                net[element].drop(eid, inplace=True)
                # res_element
                res_element = "res_" + element
                if res_element in net.keys() and isinstance(net[res_element], pd.DataFrame):
                    res_eid = net[res_element].index.intersection(eid)
                    net[res_element].drop(res_eid, inplace=True)
                if net[element].shape[0] < n_el:
                    logger.info("dropped %d %s elements" % (n_el - net[element].shape[0], element))


def drop_pipes(net, pipes):
    """
    Deletes all pipes and their geodata in the given list of indices.

    :param net: pandapipes network
    :type net: pandapipesNet
    :param pipes: pipes to be dropped from the network
    :type pipes: Iterable
    :return: No output.
    """
    # drop lines and geodata
    net["pipe"].drop(pipes, inplace=True)
    net["pipe_geodata"].drop(set(pipes) & set(net["pipe_geodata"].index), inplace=True)
    if "res_pipe" in net.keys():
        res_pipes = net.res_pipe.index.intersection(pipes)
        net["res_pipe"].drop(res_pipes, inplace=True)
    logger.info("dropped %d pipes" % len(list(pipes)))


def check_pressure_controllability(net, to_junction, controlled_junction):
    mg = create_nxgraph(net, include_pressure_circ_pumps=False, include_compressors=False,
                        include_mass_circ_pumps=False, include_press_controls=False)
    return has_path(mg, to_junction, controlled_junction)


# TODO: change to pumps??
# def drop_trafos(net, trafos, table="trafo"):
#     """
#     Deletes all trafos and in the given list of indices and removes
#     any switches connected to it.
#     """
#     if table not in ('trafo', 'trafo3w'):
#         raise UserWarning("parameter 'table' must be 'trafo' or 'trafo3w'")
#     # drop any switches
#     num_switches = 0
#     if table == 'trafo':  # remove as soon as the trafo3w switches are implemented
#         i = net["switch"].index[(net["switch"]["element"].isin(trafos)) &
#                                 (net["switch"]["et"] == "t")]
#         net["switch"].drop(i, inplace=True)
#         num_switches = len(i)
#
#     # drop the trafos
#     net[table].drop(trafos, inplace=True)
#     res_trafos = net["res_" + table].index.intersection(trafos)
#     net["res_" + table].drop(res_trafos, inplace=True)
#     logger.info("dropped %d %s elements with %d switches" % (len(trafos), table, num_switches))


node_pit_indices = {
    TABLE_IDX_NODE: "TABLE_IDX",
    ELEMENT_IDX_NODE: "ELEMENT_IDX",
    NODE_TYPE_NODE: "NODE_TYPE",
    ACTIVE_NODE: "ACTIVE",
    RHO_NODE: "RHO",
    PINIT_NODE: "PINIT",
    LOAD_NODE: "LOAD",
    HEIGHT_NODE: "HEIGHT",
    TINIT_NODE: "TINIT",
    PAMB_NODE: "PAMB",
    LOAD_T_NODE: "LOAD_T",
    NODE_TYPE_T_NODE: "NODE_TYPE_T",
    EXT_GRID_OCCURENCE_NODE: "EXT_GRID_OCCURENCE",
    EXT_GRID_OCCURENCE_T_NODE: "EXT_GRID_OCCURENCE_T",
}

branch_pit_indices = {
    TABLE_IDX_BRANCH: "TABLE_IDX",
    ELEMENT_IDX_BRANCH: "ELEMENT_IDX",
    FROM_NODE_BRANCH: "FROM_NODE",
    TO_NODE_BRANCH: "TO_NODE",
    ACTIVE_BRANCH: "ACTIVE",
    LENGTH_BRANCH: "LENGTH",
    D_BRANCH: "D",
    AREA_BRANCH: "AREA",
    RHO_BRANCH: "RHO",
    ETA_BRANCH: "ETA",
    K_BRANCH: "K",
    TINIT_BRANCH: "TINIT",
    VINIT_BRANCH: "VINIT",
    RE_BRANCH: "RE",
    LAMBDA_BRANCH: "LAMBDA",
    JAC_DERIV_DV_BRANCH: "JAC_DERIV_DV",
    JAC_DERIV_DP_BRANCH: "JAC_DERIV_DP",
    JAC_DERIV_DP1_BRANCH: "JAC_DERIV_DP1",
    LOAD_VEC_BRANCHES_BRANCH: "LOAD_VEC_BRANCHES",
    JAC_DERIV_DV_NODE_BRANCH: "JAC_DERIV_DV_NODE",
    LOAD_VEC_NODES_BRANCH: "LOAD_VEC_NODES",
    LOSS_COEFFICIENT_BRANCH: "LOSS_COEFFICIENT",
    CP_BRANCH: "CP",
    ALPHA_BRANCH: "ALPHA",
    JAC_DERIV_DT_BRANCH: "JAC_DERIV_DT",
    JAC_DERIV_DT1_BRANCH: "JAC_DERIV_DT1",
    LOAD_VEC_BRANCHES_T_BRANCH: "LOAD_VEC_BRANCHES_T",
    T_OUT_BRANCH: "T_OUT",
    JAC_DERIV_DT_NODE_BRANCH: "JAC_DERIV_DT_NODE",
    LOAD_VEC_NODES_T_BRANCH: "LOAD_VEC_NODES_T",
    VINIT_T_BRANCH: "VINIT_T",
    FROM_NODE_T_BRANCH: "FROM_NODE_T",
    TO_NODE_T_BRANCH: "TO_NODE_T",
    QEXT_BRANCH: "QEXT",
    TEXT_BRANCH: "TEXT",
    STD_TYPE_BRANCH: "STD_TYPE",
    PL_BRANCH: "PL",
    TL_BRANCH: "TL",
    BRANCH_TYPE_BRANCH: "BRANCH_TYPE",
    PRESSURE_RATIO_BRANCH: "PRESSURE_RATIO",
}


pit_types = {TYPE_P: "P", TYPE_L: "L", TYPE_NONE: "NONE", TYPE_T: "T", TYPE_PC: "PC", 0: "NONE"}


def get_internal_tables_pandas(net):
    """
    Convert the internal structure (pit) for nodes and branches into readable pandas DataFrames.

    :param net: pandapipes network
    :type net: pandapipesNet
    :return: node_table, branch_table
    :rtype: pandas.DataFrame
    """
    if "_pit" not in net:
        logger.warning("The net does not contain an internal pandapipes structure. Please try "
                       "running a pipeflow first.")
        return None, None
    branch_pit = net["_pit"]["branch"]
    node_pit = net["_pit"]["node"]

    missing_nodes = node_pit.shape[1] - node_cols
    missing_branches = branch_pit.shape[1] - branch_cols

    if missing_nodes > 0:
        logger.warning("%d node pit entries are missing. Please verify the correctness of the "
                       "table." % missing_nodes)
    if missing_branches > 0:
        logger.warning("%d branch pit entries are missing. Please verify the correctness of the "
                       "table." % missing_branches)

    node_table_lookup = pandapipes.get_lookup(net, "node", "table")
    node_table = pd.DataFrame(node_pit)
    node_table.rename(columns=node_pit_indices, inplace=True)
    node_table["NODE_TYPE"].replace(pit_types, inplace=True)
    node_table["NODE_TYPE_T"].replace(pit_types, inplace=True)
    node_table["TABLE_IDX"].replace(node_table_lookup["n2t"], inplace=True)

    branch_table_lookup = pandapipes.get_lookup(net, "branch", "table")
    branch_table = pd.DataFrame(branch_pit)
    branch_table.rename(columns=branch_pit_indices, inplace=True)
    branch_table["BRANCH_TYPE"].replace(pit_types, inplace=True)
    branch_table["TABLE_IDX"].replace(branch_table_lookup["n2t"], inplace=True)

    return node_table, branch_table
