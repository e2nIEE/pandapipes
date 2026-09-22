# Copyright (c) 2020-2026 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
import copy

import numpy as np
import pandas as pd

from pandapipes import get_fluid
from pandapipes.constants import NORMAL_PRESSURE, TEMP_GRADIENT_KPM, AVG_TEMPERATURE_K, \
    HEIGHT_EXPONENT
from pandapipes.idx_branch import IdxBranch
from pandapipes.idx_node import IdxNode
from pandapipes.pf.pipeflow_setup import get_lookup, get_net_option
from pandapipes.pf.system_index import ComponentEquations, EqWriteMode, HydVarEq, ThermVarEq
from pandas import Index


def get_hydraulic_options(net):
    """``options`` dict expected by calculate_derivatives_hydraulic.

    Factored out because every branch component's own register_hydraulic_equations rebuilt
    this same 2-key dict from net["_options"] independently.
    """
    return {"use_numba": get_net_option(net, "use_numba"),
            "friction_model": get_net_option(net, "friction_model")}


def get_thermal_options(net):
    """``options`` dict expected by calculate_derivatives_branch_thermal/calculate_derivatives_node_thermal.

    Same rationale as get_hydraulic_options; the thermal derivatives only ever need
    use_numba, not friction_model.
    """
    return {"use_numba": get_net_option(net, "use_numba")}


def register_branch_node_mass_balance(sys_idx, registry, fn, tn, mdot_col, df_dm_node, load_fn, load_tn):
    """Register a branch's own mass flow (MDOTINIT) into both its from- and to-node balances.

    Feeds additively into both its from-node's and its to-node's mass-balance equation,
    with opposite sign (mass leaving the from-node is mass entering the to-node) - this
    exact block used to be duplicated near-verbatim in every branch component that calls
    calculate_derivatives_hydraulic (pipe, valve, flow_control, pump, pressure_control,
    heat_exchanger), plus heat_consumer's own differently-derived equivalent.

    df_dm_node is always plain ones (see calculate_derivatives_hydraulic's own df_dm_nodes, or
    heat_consumer's np.ones_like(branch_idx)) - a unit of mdot change always changes a node's mass
    balance by exactly that same unit - so the +1/-1 split is baked in here. load_fn/load_tn are
    NOT re-signed here, unlike df_dm_node: callers must pass them already carrying whatever sign
    their own upstream computation assigns (calculate_derivatives_hydraulic's callers pass
    -load_fn/load_tn; heat_consumer, which derives load_fn = -MDOTINIT itself, passes load_fn/
    load_tn unchanged) - this function only assembles the (row, col, data) COO triples and
    registers them, it never touches the derivative math itself.
    """
    fn_eq = sys_idx.idx(HydVarEq.NODE, fn)
    tn_eq = sys_idx.idx(HydVarEq.NODE, tn)

    rows_node = np.concatenate([fn_eq, tn_eq]).astype(np.int32)
    cols_node = np.concatenate([mdot_col, mdot_col]).astype(np.int32)
    data_node = np.concatenate([-df_dm_node, df_dm_node]).astype(np.float64)
    load_rows_node = np.concatenate([fn_eq, tn_eq]).astype(np.int32)
    load_node = np.concatenate([load_fn, load_tn]).astype(np.float64)

    registry.add(ComponentEquations(
        rows=rows_node,
        cols=cols_node,
        data=data_node,
        load_rows=load_rows_node,
        load_data=load_node,
    ))


def register_branch_node_thermal_balance(sys_idx, registry, tn, t_tn_col, t_out_col, dfnt_dt, dfnt_dtout, fnt):
    """Register a branch's own thermal continuity equation at its to-node.

    Every branch component's own thermal continuity equation (mixed-temperature energy
    balance at its own to-node, from calculate_derivatives_branch_thermal's fnt/dfnt_dt/
    dfnt_dtout) was duplicated near-verbatim in every branch component that calls it (pipe,
    valve, flow_control, pump, pressure_control, heat_exchanger, heat_consumer) - this function
    only assembles the (row, col, data) COO triples and registers them, it never touches the
    derivative math itself. t_tn_col/t_out_col are the to-node's TINIT column and the branch's
    own TOUTINIT column respectively (callers derive them via sys_idx.idx(ThermVarEq.TINIT, tn)/
    sys_idx.idx(ThermVarEq.TOUTINIT, branch_idx)).
    """
    tn_eq = sys_idx.idx(ThermVarEq.NODE, tn)

    rows_node = np.concatenate([tn_eq, tn_eq]).astype(np.int32)
    cols_node = np.concatenate([t_tn_col, t_out_col]).astype(np.int32)
    data_node = np.concatenate([dfnt_dt, dfnt_dtout]).astype(np.float64)
    load_rows_node = tn_eq.astype(np.int32)
    load_node = fnt.astype(np.float64)

    registry.add(ComponentEquations(
        rows=rows_node,
        cols=cols_node,
        data=data_node,
        load_rows=load_rows_node,
        load_data=load_node,
    ))


def register_circ_pump_node_continuity(net, branch_pit, sys_idx, registry, table_name):
    """Register a circulation pump's own branch flow into both its nodes' mass balances.

    A circulation pump's own branch (return_junction -> flow_junction) has no momentum
    equation of its own - it prescribes flow rather than deriving a pressure drop from
    friction (unlike calculate_derivatives_hydraulic's branch components) - so its
    contribution to both nodes' mass balance is just its own MDOTINIT flowing straight
    through: d(mdot)/d(mdot) == 1, load = the branch's own signed mass flow. This was
    duplicated near-identically in CirculationPumpMass's and CirculationPumpPressure's own
    register_hydraulic_equations before being factored out here; the actual (row, col, data)
    assembly is register_branch_node_mass_balance's.
    """
    f, t = get_lookup(net, "branch", "from_to_active_hydraulics")[table_name]
    if f == t:
        return

    branch_idx = np.arange(f, t, dtype=np.int32)
    b_pit = branch_pit[f:t]
    fn = b_pit[:, IdxBranch.FROM_NODE].astype(np.int32)
    tn = b_pit[:, IdxBranch.TO_NODE].astype(np.int32)
    mdot_col = sys_idx.idx(HydVarEq.MDOTINIT, branch_idx)
    m = b_pit[:, IdxBranch.MDOTINIT]
    dm_node = np.ones(len(branch_idx), dtype=np.float64)

    register_branch_node_mass_balance(sys_idx, registry, fn, tn, mdot_col, dm_node, -m, m)


def register_circ_pump_slack_equations(net, node_pit, sys_idx, registry, table_name,
                                       active_identifier, to_junction_col, connected_node_table):
    """Register the pressure/slack-mass equations for a circulation pump's own flow junction.

    A circ pump's own flow junction gets ``NODE_TYPE = P`` purely to anchor an absolute
    pressure reference (pressure is only ever defined up to a constant otherwise) - it has
    no genuine external connection to freely supply/absorb mass, unlike a real ext_grid.

    So: the pressure-fix equation is always registered for its own flow junction (MEAN,
    letting it coexist with ExtGrid's own pressure-fix there if a real ext_grid happens to
    sit at the same node too - see ``ExtGrid.register_hydraulic_equations``). But
    ``MDOTSLACKINIT`` is only forced to 0 there if there's no real ext_grid also present
    (``COUNT_VAR_MASS_SLACK``, set by ``ExtGrid.register_pit_node_entries``) - where one is,
    ExtGrid's own registration already lets ``MDOTSLACKINIT`` freely absorb residual mass
    there, and this must not fight it for ownership of that row.
    """
    tbl = net[table_name]
    tbl = tbl[tbl[active_identifier].values]
    if not len(tbl):
        return

    p_pumps = tbl[np.isin(tbl.type.values, ["p", "pt"])]
    if not len(p_pumps):
        return

    # "index_active_hydraulics" (not the plain "index" lookup!) maps onto the ACTIVE/reduced
    # pit this method operates on - see ExtGrid.register_hydraulic_equations for why the
    # plain lookup is wrong here. -1 means disconnected - skip those.
    junction_lookup = get_lookup(net, "node", "index_active_hydraulics")[connected_node_table]
    # one entry per circ_pump ROW - not deduplicated, mirrors ExtGrid's own pressure-fix
    pump_nodes = junction_lookup[p_pumps[to_junction_col].values].astype(np.int32)
    pump_nodes = pump_nodes[pump_nodes != -1]
    if not len(pump_nodes):
        return

    p_col = sys_idx.idx(HydVarEq.PINIT, pump_nodes)
    slack_eq = sys_idx.idx(HydVarEq.SLACK, pump_nodes)

    registry.add(ComponentEquations(
        rows=slack_eq.astype(np.int32),
        cols=p_col.astype(np.int32),
        data=np.ones(len(slack_eq), dtype=np.float64),
        load_rows=slack_eq.astype(np.int32),
        load_data=np.zeros(len(slack_eq), dtype=np.float64),
        mode=EqWriteMode.MEAN,
    ))

    # Where a real ext_grid also sits (COUNT_VAR_MASS_SLACK != 0), ExtGrid's own registration
    # already adds MDOTSLACKINIT to this node's balance - skip those nodes entirely here,
    # or the coefficient would double. Deduplicated by node (unlike the pressure-fix above):
    # there is exactly one shared MDOTSLACKINIT unknown per node to reset/contribute to, not
    # one share per pump instance.
    force_zero = np.unique(pump_nodes[node_pit[pump_nodes, IdxNode.COUNT_VAR_MASS_SLACK] == 0])
    node_pit[force_zero, IdxNode.MDOTSLACKINIT] = 0.

    n_eq = sys_idx.idx(HydVarEq.NODE, force_zero)
    slack_col = sys_idx.idx(HydVarEq.MDOTSLACKINIT, force_zero)

    # plain add() (ADDITIVE, default) - joins the node's genuine balance (pipe/sink flows,
    # contributed by other components), does not replace or strip it
    registry.add(ComponentEquations(
        rows=n_eq.astype(np.int32),
        cols=slack_col.astype(np.int32),
        data=np.ones(len(n_eq), dtype=np.float64),
        load_rows=n_eq.astype(np.int32),
        load_data=node_pit[force_zero, IdxNode.MDOTSLACKINIT].astype(np.float64),  # == 0. now
    ))


def get_internal_lookup_structure(internals, table_name, internal_elements, start=0):
    internals[table_name] = np.empty((len(internal_elements), 2), dtype=np.int32)
    end = np.cumsum(internal_elements) - 1 + start
    diff = internal_elements - 1
    internals[table_name][:, 0] = end - diff
    internals[table_name][:, 1] = end

def p_correction_height_air(height):
    """Calculate the atmospheric pressure correction for a height using the barometric formula.

    :param height:
    :type height:
    :return:
    :rtype:
    """
    return NORMAL_PRESSURE * np.power(1 - height * TEMP_GRADIENT_KPM / AVG_TEMPERATURE_K,
                                      HEIGHT_EXPONENT)


def vinterp(min_vals, max_vals, lengths):
    """Compute linearly interpolated values between min_vals and max_vals for each range.

    :param min_vals:
    :type min_vals:
    :param max_vals:
    :type max_vals:
    :param lengths: lengths for each range (same length as starts)
    :type lengths: numpy.array
    :return:
    :rtype:
    """
    intervals = (max_vals - min_vals) / (lengths + 1)
    steps = np.repeat(intervals, lengths)
    counter = np.arange(lengths.sum()) - np.repeat(lengths.cumsum() - lengths, lengths) + 1
    return np.repeat(min_vals, lengths) + steps * counter


def vrange(starts, lengths):
    """Create concatenated ranges of integers for multiple start/length.

    :param starts: starts for each range
    :type starts: numpy.array
    :param lengths: lengths for each range (same length as starts)
    :type lengths: numpy.array
    :return: cat_range - concatenated ranges
    :rtype: numpy.array

    :Example:
    >>> starts = np.array([1, 3, 4, 6])
    >>> lengths = np.array([0, 2, 3, 0])
    >>> print vrange(starts, lengths)
    """
    # Repeat start position index length times and concatenate
    starting_array = np.repeat(starts, lengths)
    # Create group counter that resets for each start/length
    length_ranges = np.arange(lengths.sum()) - np.repeat(lengths.cumsum() - lengths, lengths)
    # Add group counter to group specific starts
    return starting_array + length_ranges


def init_results_element(net, element, output, all_float):
    """Initialize the results table for an element type.

    :param net: The pandapipes network
    :type net: pandapipesNet
    :param element:
    :type element:
    :param output:
    :type output:
    :param all_float:
    :type all_float:
    :return: No Output.
    """
    res_element = "res_" + element
    if all_float:
        net[res_element] = pd.DataFrame(np.nan, columns=output, index=net[element].index,
                                        dtype=np.float64)
    else:
        net[res_element] = pd.DataFrame(np.zeros(0, dtype=output), index=[])
        net[res_element] = pd.DataFrame(np.nan, index=net[element].index,
                                        columns=net[res_element].columns)


def add_new_component(net, component, overwrite=False):
    """Add a new component to the net, creating its table if necessary.

    :param net:
    :type net:
    :param component:
    :type component:
    :param overwrite:
    :type overwrite:
    :return:
    :rtype:
    """
    name = component.table_name()
    if not overwrite and name in net:
        # logger.info('%s is already in net. Try overwrite if you want to get a new entry' %name)
        return
    else:
        if hasattr(component, 'geodata'):
            geodata = component.geodata()
        else:
            geodata = None

        comp_input = component.get_component_input()
        if name not in net:
            net['component_list'].append(component)
        net.update({name: comp_input})
        if isinstance(net[name], list):
            net[name] = pd.DataFrame(np.zeros(0, dtype=net[name]), index=Index([], dtype=np.int64))
        # init_empty_results_table(net, name, component.get_result_table(net))

        if geodata is not None:
            net.update({name + '_geodata': geodata})
            if isinstance(net[name + '_geodata'], list):
                net[name + '_geodata'] = pd.DataFrame(np.zeros(0, dtype=net[name + '_geodata']),
                                                      index=Index([], dtype=np.int64))


def set_entry_check_repeat(pit, column, entry, repeat_number, repeated=True):
    pit[:, column] = np.repeat(entry, repeat_number) if repeated else entry


def build_pit_entries(rows: np.ndarray, cols: list, data: list) -> tuple:
    n = len(rows)
    all_rows = np.tile(rows, len(cols))
    all_cols = np.concatenate([np.full(n, c, dtype=np.int32) for c in cols])
    all_data = np.concatenate([
        np.full(n, d, dtype=np.float64) if np.isscalar(d) else np.asarray(d, dtype=np.float64)
        for d in data
    ])
    return all_rows, all_cols, all_data


def standard_branch_wo_internals_result_lookup(net):
    required_results_hyd = [
        ("p_from_bar", "p_from"), ("p_to_bar", "p_to"), ("mdot_to_kg_per_s", "mf_to"),
        ("mdot_from_kg_per_s", "mf_from")
    ]
    required_results_ht = [("t_from_k", "temp_from"), ("t_to_k", "temp_to"), ("t_outlet_k", "t_outlet")]

    if get_fluid(net).is_gas:
        required_results_hyd.extend([
            ("normfactor_from", "normfactor_from"),
            ("normfactor_to", "normfactor_to"), ("vdot_norm_m3_per_s", "vf")
        ])
    else:
        required_results_hyd.extend([("vdot_m3_per_s", "vf")])

    return required_results_hyd, required_results_ht


def get_component_array(net, component_name, component_type="branch", mode='hydraulics', only_active=True):  # pylint: disable=unused-argument
    """Returns the internal array of a component.

    :param net: The pandapipes network
    :type net: pandapipesNet
    :param component_name: Table name of the component for which to extract internal array
    :type component_name: str
    :param component_type: Kept for API compatibility; the active reduction is now performed once
        in :func:`~pandapipes.pf.pipeflow_setup.reduce_component_pits` (called from
        :func:`~pandapipes.pf.pipeflow_setup.reduce_pit`), which already resolves the component's
        type on its own.
    :type component_type: str, default "branch"
    :param mode: Kept for API compatibility; ``net["_active_pit"]`` always reflects whichever mode
        ``reduce_pit`` was last called with, and every caller only requests entries for that same
        mode, so this no longer needs to be applied here.
    :type mode: str, default "hydraulics"
    :param only_active: If True, only return entries of active elements (included in _active_pit)
    :type only_active: bool
    :return: component_array - internal array of the component, row-aligned with
        ``net["_active_pit"][component_type][f:t]`` for this component's own from/to range
    :rtype: numpy.ndarray
    """
    if not only_active:
        return net["_pit"]["components"][component_name]
    return net["_active_pit"]["components"][component_name]


def get_std_type_lookup(net, table_name):
    return np.array(list(net.std_types[table_name].keys()))


def retrieve_u(params):
    params = copy.deepcopy(params)
    if not "u_w_per_m2k" in params:
        params["u_w_per_m2k"] = np.nan
    if not "u_w_per_mk" in params:
        params["u_w_per_mk"] = np.nan
    if not np.isnan(params["u_w_per_m2k"]) and not np.isnan(params["u_w_per_mk"]):
        raise UserWarning(r'u_w_per_m2k and u_w_per_mk have been both defined. '
                          r'This might lead to problems due to ambiguity! '
                          r'Delete one value and update your standard type!')
    elif not np.isnan(params["u_w_per_mk"]):
        params["u_w_per_m2k"] = params["u_w_per_mk"] / (params["outer_diameter_mm"] * np.pi) * 1000.
    return params
