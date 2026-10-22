# Copyright (c) 2020-2026 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
import copy
import os
from collections.abc import Iterable
import warnings

import numpy as np
import pandas as pd
from networkx import has_path
from pandapower.auxiliary import get_indices
from pandapower.toolbox import dataframes_equal
from pandapower.toolbox.result_info import clear_result_tables

import pandapipes
from pandapipes.component_models.abstract_models.branch_models import BranchComponent
from pandapipes.component_models.abstract_models.node_element_models import NodeElementComponent
from pandapipes.create import create_empty_network
from pandapipes.idx_branch import IdxBranch
from pandapipes.idx_node import IdxNode
from pandapipes.pandapipes_net import pandapipesNet
from pandapipes.topology import create_nxgraph

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def nets_equal(net1, net2, check_only_results=False, exclude_elms=None, **kwargs):
    """Compares the DataFrames of two networks.

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
    """Utility function.

    Provides the tuples of elements and corresponding columns for junctions they are connected to.

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
        Compressor, FlowControlComponent, MassStorage, HeatConsumer
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
                     FlowControlComponent, ValvePipe, HeatConsumer, MassStorage]
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
    """Provides a list of all pandapipes elements belonging to the desired element types.

    If a net is given, the elements are derived from the component list.

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
    """Changes the index of net.junction and considers the new junction indices in all other pandapipes element tables.

    :param net: pandapipes network
    :type net: pandapipesNet
    :param junction_lookup: the keys are the old junction indices, the values the new junction \
            indices
    :type junction_lookup: dict
    :return: junction_lookup - the finally reindexed junction lookup (with corrections if necessary)
    :rtype: dict
    """
    junction_lookup = reindex_elements(net, "junction", lookup=junction_lookup)
    return junction_lookup


def reindex_pipes(net, pipe_lookup):
    """Changes the index of net.pipe and considers the new pipe indices in pandapipes valve table.

    :param net: pandapipes network
    :type net: pandapipesNet
    :param pipe_lookup: the keys are the old pipe indices, the values the new pipe \
            indices
    :type pipe_lookup: dict
    :return: pipe_lookup - the finally reindexed pipe lookup (with corrections if necessary)
    :rtype: dict
    """
    pipe_lookup = reindex_elements(net, "pipe", lookup=pipe_lookup)
    return pipe_lookup


def reindex_elements(net, element, lookup):
    """Changes the index of net[element].

    :param net: pandapipes network
    :type net: pandapipesNet
    :param element: name of the element table
    :type element: str
    :param lookup: the keys are the old indices, the values the new indices
    :type lookup: dict
    :return: No output.
    """
    if element not in net:
        return lookup
    not_fitting_lookup_keys = set(lookup.keys()) - set(net[element].index)
    if len(not_fitting_lookup_keys):
        logger.error(f"These {element} indices are unknown to net. Thus, they cannot be reindexed: " +
                     str(not_fitting_lookup_keys))

    missing_indices = sorted(set(net[element].index) - set(lookup.keys()))
    if len(missing_indices):
        duplicates = set(missing_indices).intersection(set(lookup.values()))
        if len(duplicates):
            logger.error(f"The {element} %s are not listed in the lookup but their index is "
                         "used as a new index. This would result in duplicated indices."
                         % (str(duplicates)))
        else:
            lookup.update({j: j for j in missing_indices})

    # --- reindex
    net[element].index = get_indices(net[element].index, lookup)

    # --- adapt geodata index
    geo_table = element + "_geodata"
    if geo_table  in net and net[geo_table ].shape[0]:
        net[geo_table ].index = get_indices(net[geo_table ].index, lookup)

    res_table = "res_" + element
    if hasattr(net, res_table):
        net[res_table].index = get_indices(net[res_table].index, lookup)

    if element == "junction":
        for element, value in element_junction_tuples(net=net):
            if element in net.keys():
                if (element == "valve") and (value == "element"):
                    net[element].loc[net[element]["et"] == "ju", value] = (
                        get_indices(net[element].loc[net[element]["et"] == "ju", value], lookup))
                else:
                    net[element][value] = get_indices(net[element][value], lookup)
    elif element == "pipe":
        if "valve" in net:
            pipe_valves = net["valve"].loc[net["valve"]["et"] == "pi", "element"]
            net["valve"].loc[pipe_valves.index, "element"] = get_indices(pipe_valves, lookup)

    return lookup

def create_continuous_junction_index(net, start=0, store_old_index=False):
    """Creates a continuous junction index starting at 'start' and replaces all references of old indices by the new ones.

    :param net: pandapipes network
    :type net: pandapipesNet
    :param start: index begins with "start"
    :type start: int, default 0
    :param store_old_index: if True, stores the old index in net.junction["old_index"]
    :type store_old_index: bool, default False
    :return: junction_lookup - mapping of old to new index
    :rtype: dict
    """
    junction_lookup = create_continuous_element_index(net, "junction", start, store_old_index=store_old_index)
    return junction_lookup

def create_continuous_element_index(net, element, start=0, store_old_index=False):
    """Creates a continuous element index starting at 'start' and replaces all references of old indices by the new ones.

    :param net: pandapipes network
    :type net: pandapipesNet
    :param element: pandapipes element
    :type element: str
    :param start: index begins with "start"
    :type start: int, default 0
    :param store_old_index: if True, stores the old index in e.g. net.junction["old_index"]
    :type store_old_index: bool, default False
    :return: junction_lookup - mapping of old to new index
    :rtype: dict
    """
    if element not in net:
        return
    net[element].sort_index(inplace=True)
    if store_old_index:
        net[element]["old_index"] = net[element].index.values
    new_index = list(np.arange(start, len(net[element]) + start))
    lookup = dict(zip(net[element].index.values, new_index))
    if element in net and isinstance(net[element], pd.DataFrame):
        lookup = reindex_elements(net, element, lookup)
    else:
        logger.debug("No indices could be changed for element '%s'." % element)
    return lookup

def create_continuous_elements_index(net, start=0, add_df_to_reindex=None, store_old_index=False):
    """Creating a continuous index for all the elements and replaces all references of old indices by the new ones.

    :param net: pandapipes network with unodered indices
    :type net: pandapipesNet
    :param start: new index begins with "start"
    :type start: int
    :param add_df_to_reindex: by default all useful pandapower elements for power flow will be\
          selected. Customized DataFrames can also be considered here.
    :type add_df_to_reindex: iterable, default None
    :param store_old_index: if True, stores the old index in e.g. net.junction["old_index"]
    :type store_old_index: bool, default False
    :return: net - pandapipes network with odered and continuous indices
    :rtype: pandapipesNet
    """
    add_df_to_reindex = set() if add_df_to_reindex is None else set(add_df_to_reindex)
    elements = pp_elements(include_res_elements=True, net=net)
    elements |= add_df_to_reindex

    # run reindex_elements() for all elements
    lookups = dict()
    for elm in list(elements):
        if elm in ["junction_geodata", "pipe_geodata"]:
            logger.info(f"The table {elm} doesn't need to be included to 'add_df_to_reindex'. It is "
                        f"already included by element==\'{elm.split('_')[0]}\'.")
        else:
            lookups[elm] = create_continuous_element_index(net, elm, start, store_old_index=store_old_index)
    return lookups


def fuse_junctions(net, j1, j2, drop=True):
    """Reroutes any connections to junctions in j2 to the given junction j1.

    Additionally drops the junctions j2, if drop=True (default).

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
    """Selects a subnet by a list of junction indices and returns a net with all components connected to them."""
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
    """Drops specified junctions, their junction_geodata and by default drops all elements connected to them as well.

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
    """Drop elements connected to given junctions.

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
    """Deletes all pipes and their geodata in the given list of indices.

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


pit_types = {IdxNode.P: "P", IdxNode.L: "L", IdxNode.T: "T", IdxNode.PC: "PC"}
int_cols = ["FROM_NODE", "TO_NODE", "ELEMENT_IDX"]
bool_cols = ["ACTIVE"]


def get_pit_lookup(pit_type="node"):
    """Retrieve a lookup for "indices" and "types" from the idx_branch or idx_node files.

    :param pit_type: the pit for which the lookup is generated ("branch" or "node")
    :type pit_type: str, default "node"
    :return: idx_lookup: dictionary of structure {column: name} derived from idx_node.py or \
        idx_branch.py to describe the structure of the pit
    :rtype: dict
    """
    idx_lookup = {"indices": dict(), "types": dict()}
    with open(os.path.join(pandapipes.pp_dir, f"idx_{pit_type}.py")) as f:
        lookup_type = None
        for ln in f.readlines():
            if ln.strip().startswith("#"):
                if "indices" in ln:
                    lookup_type = "indices"
                elif "types" in ln:
                    lookup_type = "types"
            if lookup_type is None:
                continue
            txt = ln.split("#")[0]
            if "=" not in txt:
                continue
            part1, part2 = txt.split("=")
            idx_lookup[lookup_type][int(part2.strip())] = part1.strip()
    return idx_lookup


def get_internal_tables_pandas(net, convert_types=True):
    """Convert the internal structure (pit) for nodes and branches into readable pandas DataFrames.

    :param net: pandapipes network
    :type net: pandapipesNet
    :param convert_types: if True, convert some columns into a more readable form
    :type convert_types: bool, default True
    :return: node_table, branch_table
    :rtype: pandas.DataFrame
    """
    if "_pit" not in net:
        logger.warning("The net does not contain an internal pandapipes structure. Please try "
                       "running a pipeflow first.")
        return None, None
    branch_pit = net["_pit"]["branch"]
    node_pit = net["_pit"]["node"]

    missing_nodes = node_pit.shape[1] - IdxNode.node_cols
    missing_branches = branch_pit.shape[1] - IdxBranch.branch_cols

    if missing_nodes > 0:
        logger.warning("%d node pit entries are missing. Please verify the correctness of the "
                       "table." % missing_nodes)
    if missing_branches > 0:
        logger.warning("%d branch pit entries are missing. Please verify the correctness of the "
                       "table." % missing_branches)

    node_lookup = get_pit_lookup("node")
    branch_lookup = get_pit_lookup("branch")

    node_table_lookup = pandapipes.get_lookup(net, "node", "table")
    node_table = pd.DataFrame(node_pit)
    node_table.rename(columns=node_lookup["indices"], inplace=True)

    branch_table_lookup = pandapipes.get_lookup(net, "branch", "table")
    branch_table = pd.DataFrame(branch_pit)
    branch_table.rename(columns=branch_lookup["indices"], inplace=True)

    if convert_types:
        # TODO: replace types with a version retrieved from get_pit_lookup, but before, the types
        #       need to be defined properly inside the files!
        node_table.replace({"NODE_TYPE": pit_types}, inplace=True)
        node_table.replace({"NODE_TYPE_T": pit_types}, inplace=True)
        node_table.replace( {"TABLE_IDX": node_table_lookup["n2t"]}, inplace=True)

        branch_table.replace({"BRANCH_TYPE": pit_types}, inplace=True)
        branch_table.replace({"TABLE_IDX": branch_table_lookup["n2t"]}, inplace=True)

        for col in int_cols + bool_cols:
            for tbl in (node_table, branch_table):
                if col in tbl.columns:
                    tbl[col] = tbl[col].astype(np.int32)
                    if col in bool_cols:
                        tbl[col] = tbl[col].astype(np.bool_)

    return node_table, branch_table

def _get_fluid_heat_capacity(net, cp_w_per_kg=None):
    """
    Determines the heat capacity of the net's fluid, either from the explicitly given value or from
    the fluid properties.

    :param net: pandapipes network
    :type net: pandapipesNet
    :param cp_w_per_kg: heat capacity in [J/(kg K)] that will be used if given
    :type cp_w_per_kg: float, default None
    :return: cp_w_per_kg - heat capacity in [J/(kg K)]
    :rtype: float
    """
    if cp_w_per_kg is not None:
        return cp_w_per_kg
    try:
        from pandapipes.properties.fluids import get_fluid
        t_ref = net.junction.tfluid_k.mean() if len(net.junction) > 0 else 293.15
        if t_ref is None or np.isnan(t_ref):
            t_ref = 293.15
        return float(get_fluid(net).get_heat_capacity(t_ref))
    except Exception:
        logger.warning("The heat capacity could not be retrieved from the fluid of the net. "
                       "A value of 4182 J/(kg K) (water at ~40 deg) is assumed.")
        return 4182.


def convert_open_to_closed_loop(net, deltat_k=None, plift_bar=1.0, cp_w_per_kg=None,
                                pump_type="const_pressure", t_flow_k=None, geodata_offset=(0., -1.),
                                **kwargs):
    """
    Converts an open loop heating grid (hydraulic boundary defined by an external grid as well as
    sinks and sources) into a closed loop heating grid.

    A copy of the net is created and returned; the original net is not modified. The conversion
    performs the following steps:

    - Every junction of the open grid becomes part of the "flow" line. For each of these junctions,
      a corresponding junction of the "return" line is created (with the same parameters).
    - Every pipe is mirrored into the return line with reversed flow direction, i.e. the return pipe
      connects the return junction of the original pipe's to_junction with the return junction of
      the original pipe's from_junction. Other branch elements (valves, pumps, heat exchangers, ...)
      remain on the flow line only.
    - Every sink becomes a heat consumer element that extracts the given mass flow from the flow
      line and feeds it back into the return line, cooling it down by deltat_k.
    - Every source becomes a heat consumer element with negative heat flux (heat feed-in) that
      transports the given mass flow from the flow line into the return line.
    - The external grid is replaced by a circulation pump that transports the water from the return
      junction (corresponding to the external grid junction) into the flow junction.
    - All junctions and pipes receive a "system" column ("flow" or "return") if not already
      present.

    :param net: an open loop pandapipes network
    :type net: pandapipesNet
    :param deltat_k: temperature difference between flow and return line that is used for all heat \
        consumers replacing sinks and sources; must be given explicitly
    :type deltat_k: float
    :param plift_bar: pressure lift of the circulation pump replacing the external grid
    :type plift_bar: float, default 1.0
    :param cp_w_per_kg: heat capacity of the fluid for the calculation of the heat power of the \
        consumers replacing the sources; if None, it is derived from the fluid of the net if \
        possible, otherwise a water value is assumed (with a warning)
    :type cp_w_per_kg: float, default None
    :param pump_type: type of the circulation pump that replaces the external grid, either \
        "const_pressure" or "const_mass_flow"; for "const_mass_flow", the mass flow is inferred \
        from the heat consumers (sum of the controlled mass flows) unless \
        "mdot_flow_kg_per_s" is given via kwargs
    :type pump_type: str, default "const_pressure"
    :param t_flow_k: temperature set point of the circulation pump; if None, the temperature of \
        the external grid is used
    :type t_flow_k: float, default None
    :param geodata_offset: (x, y) offset that is added to the geodata of all junctions to place \
        the return line
    :type geodata_offset: tuple(float, float), default (0, -1)
    :param kwargs: additional keyword arguments for the circulation pump creation (p_flow_bar, \
        mdot_flow_kg_per_s, name, ...)
    :type kwargs: dict
    :return: net - a copy of the net converted to a closed loop
    :rtype: pandapipesNet
    """
    if deltat_k is None:
        raise AttributeError("The parameter 'deltat_k' must be given to convert sinks and sources "
                             "into heat consumers. It defines the temperature difference between "
                             "flow and return line.")
    if pump_type not in ["const_pressure", "const_mass_flow"]:
        raise ValueError(f"pump_type must be 'const_pressure' or 'const_mass_flow', got "
                         f"'{pump_type}'.")
    if "circ_pump_pressure" in net and len(net.circ_pump_pressure) > 0 \
            or "circ_pump_mass" in net and len(net.circ_pump_mass) > 0:
        raise UserWarning("The net already contains circulation pumps. It is probably already a "
                          "closed loop network.")
    if "heat_consumer" in net and len(net.heat_consumer) > 0:
        raise UserWarning("The net already contains heat consumers. It is probably already a "
                         "closed loop network.")
    if "ext_grid" not in net or len(net.ext_grid) == 0:
        raise UserWarning("The net contains no external grid. Cannot convert an open loop "
                          "network without a hydraulic boundary.")
    if len(net.ext_grid[net.ext_grid.in_service]) != 1:
        raise UserWarning("The conversion expects exactly one in-service external grid, but the "
                          "net contains %d." % len(net.ext_grid[net.ext_grid.in_service]))

    net = copy.deepcopy(net)
    cp = _get_fluid_heat_capacity(net, cp_w_per_kg)

    # --- remember original junction set and mark them as flow line
    junctions = list(net.junction.index)
    net.junction["system"] = "flow"
    if "pipe" in net:
        net.pipe["system"] = "flow"

    # --- create the return junctions
    flow_to_return = dict()
    for j in junctions:
        row = net.junction.loc[j]
        geo = None
        if j in net.junction_geodata.index:
            geo = (net.junction_geodata.at[j, "x"] + geodata_offset[0],
                   net.junction_geodata.at[j, "y"] + geodata_offset[1])
        flow_to_return[j] = pandapipes.create_junction(
            net, pn_bar=row.pn_bar, tfluid_k=row.tfluid_k, height_m=row.height_m,
            name=f"return of {row['name']}", in_service=row.in_service, type=row.type,
            system="return", geodata=geo)

    # --- create the return pipes (reversed direction)
    for p in list(net.pipe.index):
        pipe = net.pipe.loc[p]
        if isinstance(pipe.std_type, str) and pipe.std_type in net.std_types.get("pipe", {}):
            pandapipes.create_pipe(
                net, from_junction=flow_to_return[pipe.to_junction],
                to_junction=flow_to_return[pipe.from_junction], std_type=pipe.std_type,
                length_km=pipe.length_km, loss_coefficient=pipe.loss_coefficient,
                sections=pipe.sections, text_k=pipe.text_k, name=f"return of {pipe['name']}",
                in_service=pipe.in_service, type=pipe.type, system="return")
        else:
            pandapipes.create_pipe_from_parameters(
                net, from_junction=flow_to_return[pipe.to_junction],
                to_junction=flow_to_return[pipe.from_junction], length_km=pipe.length_km,
                inner_diameter_mm=pipe.inner_diameter_mm, outer_diameter_mm=pipe.outer_diameter_mm,
                k_mm=pipe.k_mm, loss_coefficient=pipe.loss_coefficient, sections=pipe.sections,
                u_w_per_m2k=pipe.u_w_per_m2k, text_k=pipe.text_k,
                name=f"return of {pipe['name']}", in_service=pipe.in_service, type=pipe.type,
                system="return")

    # --- sinks and sources become heat consumers between flow and return line
    eg_index = net.ext_grid.index[net.ext_grid.in_service.astype(bool)][0]
    eg_row = copy.deepcopy(net.ext_grid.loc[eg_index])
    eg_junction = int(eg_row.junction)
    eg_p_bar = copy.deepcopy(eg_row.p_bar)
    eg_t_k = copy.deepcopy(eg_row.t_k)
    if "sink" in net:
        for idx in list(net.sink.index):
            sink = net.sink.loc[idx]
            mdot = sink.mdot_kg_per_s * sink.scaling
            pandapipes.create_heat_consumer(
                net, from_junction=sink.junction, to_junction=flow_to_return[sink.junction],
                controlled_mdot_kg_per_s=mdot, deltat_k=deltat_k,
                name=f"heat consumer of {sink['name']}", in_service=sink.in_service)
    if "source" in net:
        for idx in list(net.source.index):
            source = net.source.loc[idx]
            mdot = source.mdot_kg_per_s * source.scaling
            pandapipes.create_heat_consumer(
                net, from_junction=source.junction, to_junction=flow_to_return[source.junction],
                controlled_mdot_kg_per_s=mdot, qext_w=-mdot * cp * deltat_k,
                name=f"heat producer of {source['name']}", in_service=source.in_service)

    # --- remove the open loop boundary elements (tables, result tables and components)
    _drop_component_tables(net, ["sink", "source", "ext_grid"])

    # --- drop all result tables (they refer to the original grid topology)
    for tbl in [k for k in list(net.keys()) if k.startswith("res_")]:
        del net[tbl]

    # --- replace the external grid by a circulation pump from return to flow line
    t_flow = t_flow_k if t_flow_k is not None else eg_t_k
    name = kwargs.pop("name", f"circulation pump of {eg_row['name']}")
    if pump_type == "const_mass_flow":
        mdot_flow = kwargs.pop("mdot_flow_kg_per_s", None)
        if mdot_flow is None:
            if "heat_consumer" in net:
                mdot_flow = net.heat_consumer.controlled_mdot_kg_per_s.sum()
        if mdot_flow is None or np.isnan(mdot_flow) or mdot_flow == 0:
            raise UserWarning("The mass flow of the circulation pump cannot be determined. "
                              "Please provide 'mdot_flow_kg_per_s' via kwargs.")
        pandapipes.create_circ_pump_const_mass_flow(
            net, return_junction=flow_to_return[eg_junction], flow_junction=eg_junction,
            p_flow_bar=kwargs.pop("p_flow_bar", eg_p_bar), mdot_flow_kg_per_s=mdot_flow,
            t_flow_k=t_flow, name=name, **kwargs)
    else:
        kwargs.pop("mdot_flow_kg_per_s", None)
        pandapipes.create_circ_pump_const_pressure(
            net, return_junction=flow_to_return[eg_junction], flow_junction=eg_junction,
            p_flow_bar=kwargs.pop("p_flow_bar", eg_p_bar), plift_bar=plift_bar,
            t_flow_k=t_flow, name=name, **kwargs)

    return net


def convert_closed_to_open_loop(net, p_bar=None, t_k=None, cp_w_per_kg=None, **kwargs):
    """
    Converts a closed loop heating grid (hydraulic boundary defined by a circulation pump as well
    as heat consumers and a parallel flow and return line) into an open loop grid.

    A copy of the net is created and returned; the original net is not modified. The conversion
    performs the following steps:

    - The return line of the grid is identified, either by the "system" column of the junction
      table (if it labels the junctions as "flow" and "return") or by a topology search from the
      return junction of the circulation pump and the to_junctions of the heat consumers.
    - All heat consumers are replaced by sinks (heat extraction) or sources (heat feed-in) that
      are connected to the from_junction (flow line) of the respective heat consumer.
    - The circulation pump is replaced by an external grid that is connected to the flow junction
      of the pump.
    - The return line (return junctions and return pipes) is removed from the grid.

    :param net: a closed loop pandapipes network
    :type net: pandapipesNet
    :param p_bar: pressure set point of the external grid replacing the circulation pump; if \
        None, the pressure set point of the pump is used
    :type p_bar: float, default None
    :param t_k: temperature set point of the external grid replacing the circulation pump; if \
        None, the temperature set point of the pump is used
    :type t_k: float, default None
    :param cp_w_per_kg: heat capacity of the fluid for the calculation of the mass flow of the \
        sinks and sources (if it cannot be derived from the heat consumer parameters); if None, \
        it is derived from the fluid of the net if possible, otherwise a water value is assumed \
        (with a warning)
    :type cp_w_per_kg: float, default None
    :param kwargs: additional keyword arguments for the external grid creation (name, type, ...)
    :type kwargs: dict
    :return: net - a copy of the net converted to an open loop
    :rtype: pandapipesNet
    """
    for tbl in ["ext_grid", "sink", "source"]:
        if tbl in net and len(net[tbl]) > 0:
            raise UserWarning("The net contains %s elements. It is probably already an open loop "
                              "network." % tbl)
    in_service_pumps = {tbl: net[tbl].index[net[tbl].in_service.astype(bool)]
                        for tbl in ["circ_pump_pressure", "circ_pump_mass"]
                        if tbl in net and len(net[tbl]) > 0}
    if len(in_service_pumps) > 1:
        raise UserWarning("The net contains in-service circulation pumps of more than one type "
                          "(%s). The conversion expects exactly one in-service circulation pump."
                          % ", ".join(in_service_pumps.keys()))
    if len(in_service_pumps) == 0 or len(list(in_service_pumps.values())[0]) != 1:
        raise UserWarning("The conversion expects exactly one in-service circulation pump.")

    net = copy.deepcopy(net)

    # --- drop all result tables (they refer to the original grid topology)
    for tbl in [k for k in list(net.keys()) if k.startswith("res_")]:
        del net[tbl]

    # --- extract the circulation pump
    pump_tbl = list(in_service_pumps.keys())[0]
    pump_idx = in_service_pumps[pump_tbl][0]
    pump = copy.deepcopy(net[pump_tbl].loc[pump_idx])
    pump_flow_junction = int(pump.flow_junction)
    pump_return_junction = int(pump.return_junction)
    cp = _get_fluid_heat_capacity(net, cp_w_per_kg)

    # --- identify the return line
    if "heat_consumer" in net and len(net.heat_consumer) > 0:
        consumer_rows = net.heat_consumer
    else:
        consumer_rows = pd.DataFrame(columns=["from_junction", "to_junction"])
    if "system" in net.junction.columns and (net.junction["system"] == "return").any():
        return_junctions = set(net.junction.index[net.junction["system"] == "return"])
    else:
        # topology search: breadth first search over the pipes only, starting from the return
        # junction of the pump and the to_junctions of the heat consumers
        adjacency = dict()
        if "pipe" in net:
            for pipe in net.pipe.itertuples():
                adjacency.setdefault(pipe.from_junction, set()).add(pipe.to_junction)
                adjacency.setdefault(pipe.to_junction, set()).add(pipe.from_junction)
        seeds = {pump_return_junction} | set(consumer_rows.to_junction.astype(int))
        return_junctions = set(seeds)
        frontier = list(seeds)
        while frontier:
            next_frontier = [nb for j in frontier for nb in adjacency.get(j, set())
                             if nb not in return_junctions]
            return_junctions.update(next_frontier)
            frontier = next_frontier
        if len(consumer_rows) == 0 and return_junctions == set(net.junction.index):
            raise UserWarning("The return line could not be identified by topology search. "
                              "Please label the junctions of the grid with a 'system' column "
                              "('flow' or 'return').")
    if pump_flow_junction in return_junctions or pump_return_junction not in return_junctions:
        raise UserWarning("The return line identification seems inconsistent with the "
                          "circulation pump (flow junction %d, return junction %d, return line "
                          "%s). Please check the 'system' column of the junctions."
                          % (pump_flow_junction, pump_return_junction, sorted(return_junctions)))
    if any(consumer_rows.to_junction.astype(int).apply(
            lambda j: j not in return_junctions)):
        raise UserWarning("Some heat consumers are not connected to the return line with their "
                          "to_junction. Please check the 'system' column of the junctions.")

    # --- heat consumers become sinks and sources at their from_junction (flow line)
    for hc in consumer_rows.index:
        row = consumer_rows.loc[hc]
        mdot = row.controlled_mdot_kg_per_s
        if mdot is None or pd.isna(mdot):
            qext = row.qext_w
            if qext is None or pd.isna(qext):
                raise UserWarning("The mass flow of heat consumer %s cannot be determined. "
                                  "Please provide 'controlled_mdot_kg_per_s' or 'qext_w' "
                                  "for the heat consumers." % row["name"])
            if not pd.isna(row.deltat_k) and row.deltat_k != 0:
                mdot = abs(qext) / (cp * row.deltat_k)
            elif not pd.isna(row.treturn_k) and not pd.isna(pump.t_flow_k) \
                    and (pump.t_flow_k - row.treturn_k) != 0:
                mdot = abs(qext) / (cp * (pump.t_flow_k - row.treturn_k))
            else:
                raise UserWarning("The mass flow of heat consumer %s cannot be determined. "
                                  "Please provide 'deltat_k' or 'treturn_k' for the heat "
                                  "consumers or 'cp_w_per_kg' and a pump temperature set "
                                  "point." % row["name"])
        if not pd.isna(row.qext_w) and row.qext_w < 0:
            pandapipes.create_source(
                net, junction=int(row.from_junction), mdot_kg_per_s=mdot,
                name=f"source of {row['name']}", in_service=row.in_service)
        else:
            pandapipes.create_sink(
                net, junction=int(row.from_junction), mdot_kg_per_s=mdot,
                name=f"sink of {row['name']}", in_service=row.in_service)

    # --- remove the closed loop boundary elements (tables, result tables and components)
    _drop_component_tables(net, [pump_tbl, "heat_consumer", "circ_pump_mass",
                                 "circ_pump_pressure"])

    # --- replace the circulation pump by an external grid at the flow junction
    p_bar = p_bar if p_bar is not None else pump.p_flow_bar
    t_k = t_k if t_k is not None else pump.t_flow_k
    pandapipes.create_ext_grid(net, junction=pump_flow_junction, p_bar=p_bar, t_k=t_k, **kwargs)

    # --- drop the return pipes and return junctions
    if "pipe" in net:
        mask = net.pipe.from_junction.isin(return_junctions) \
            & net.pipe.to_junction.isin(return_junctions)
        if "system" in net.pipe.columns:
            mask = mask | (net.pipe["system"] == "return")
        drop_pipes(net, net.pipe.index[mask])
    drop_junctions(net, return_junctions, drop_elements=True)

    # --- remove the "system" column if it only contains flow entries now
    for tbl in ["junction", "pipe"]:
        if tbl in net and "system" in net[tbl].columns:
            systems = net[tbl]["system"].dropna()
            if not len(systems) or (systems == "flow").all():
                net[tbl].drop(columns="system", inplace=True)

    return net


def _drop_component_tables(net, tables):
    for tbl in tables:
        if tbl in net:
            del net[tbl]
        if "res_" + tbl in net:
            del net["res_" + tbl]
    net.component_list = [c for c in net.component_list if c.table_name() not in tables]

def _deprecation_check_u(kwargs):
    if ("alpha_w_per_m2k" in kwargs) and not ("u_w_per_m2k" in kwargs):
        warnings.warn("The parameter alpha_w_per_m2k has been renamed to u_w_per_m2k "
                      "and will be directly extracted from the std_type in the future."
                      , DeprecationWarning)
        u = kwargs.pop('alpha_w_per_m2k')
    elif "u_w_per_m2k" in kwargs:
        warnings.warn("The parameter u_w_per_m2k will be directly extracted from the std_type in the future."
                      , DeprecationWarning)
        u = kwargs.pop("u_w_per_m2k")
    else:
        u = None
    return u

def _deprecation_check_k(kwargs, params):
    if 'k_mm' in kwargs:
        warnings.warn("The parameter k_mm will be directly extracted from the std_type in the future."
                      , DeprecationWarning)
        k = kwargs.pop('k_mm')
    elif "k_mm" not in params:
        warnings.warn("Please update your std_types as k_mm-values are not given. These are set equals to zero.")
        k = 0
    else:
        k = None
    return k
