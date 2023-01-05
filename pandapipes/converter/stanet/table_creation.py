# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

from itertools import chain

import numpy as np
import pandas as pd

import pandapipes
from pandapipes.component_models.component_toolbox import vrange
from pandapipes.converter.stanet.valve_pipe_component import create_valve_pipe_from_parameters

try:
    from shapely.geometry import LineString
    from shapely.ops import substring
    SHAPELY_INSTALLED = True
except ImportError:
    SHAPELY_INSTALLED = False

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


NODE_TYPE = 1
HOUSE_TYPE = 8
HOUSE_CONNECTION_TYPE = 35
METER_TYPE = 36
HOUSE_NODE_TYPE = 37

MAIN_PIPE_TYPE = 2
HOUSE_PIPE_TYPE = 38
CLIENT_TYPES_OF_NODES = {NODE_TYPE: "nodes", HOUSE_TYPE: "houses",
                         HOUSE_CONNECTION_TYPE: "house_connections", METER_TYPE: "meters",
                         HOUSE_NODE_TYPE: "house_nodes"}
CLIENT_TYPES_OF_PIPES = {MAIN_PIPE_TYPE: "main", HOUSE_PIPE_TYPE: "house"}


def create_junctions_from_nodes(net, stored_data, net_params, index_mapping, add_layers):
    """
    Creates pandapipes junctions from given STANET nodes.
    :param net: pandapipes Net
    :type net: pandapipesNet
    :param stored_data: STANET data
    :type stored_data: dic
    :param net_params:
    :type net_params:
    :param index_mapping:
    :type index_mapping:
    :param add_layers:
    :type add_layers:
    :return:
    :rtype:
    """
    if "nodes" not in stored_data:
        return
    logger.info("Creating junctions from STANET nodes.")
    node_table = stored_data["nodes"]
    stanet_nrs = node_table.RECNO.astype(np.int32)
    knams = node_table.KNAM.astype(str).values
    add_info = {"stanet_id": node_table.STANETID.astype(str).values
                if "STANETID" in node_table.columns else knams,
                "p_stanet": node_table.PRECH.values.astype(np.float64),
                "stanet_valid": ~node_table.CALCBAD.values.astype(np.bool_)}
    if hasattr(node_table, "KFAK"):
        add_info["K_stanet"] = node_table.KFAK.values.astype(np.float64)
    if add_layers:
        add_info["stanet_layer"] = node_table.LAYER.values.astype(str)
    temperatures = pd.Series(net_params["medium_temp_K"], index=node_table.index, dtype=np.float64)
    eg_ind = ((node_table.FSTATUS == '?') & (node_table.DSTATUS == '!')).values
    eg_temps = node_table.loc[eg_ind, "TMESS"].values + 273.15
    if net_params["calculate_temp"]:
        temperatures.loc[eg_ind] = eg_temps
    else:
        eg_temps = temperatures.loc[eg_ind]
    heights = node_table.GEOH.values.astype(np.float64)
    geodata = node_table.loc[:, ["XRECHTS", "YHOCH"]].values
    eg_press = node_table.loc[eg_ind, "PMESS"].values
    pressures = pd.Series(np.nan, index=node_table.index, dtype=np.float64)
    pressures.loc[eg_ind] = eg_press

    junction_indices = pandapipes.create_junctions(
        net, len(node_table), pressures.values, temperatures.values, heights, name=knams,
        geodata=geodata, stanet_nr=stanet_nrs, type="node",
        stanet_active=node_table.ISACTIVE.values.astype(np.bool_),
        stanet_system=CLIENT_TYPES_OF_PIPES[MAIN_PIPE_TYPE], **add_info)
    for eg_junc, p_bar, t_k in zip(junction_indices[eg_ind], eg_press, eg_temps):
        pandapipes.create_ext_grid(net, eg_junc, p_bar, t_k, type="pt",
                                   stanet_system=CLIENT_TYPES_OF_PIPES[MAIN_PIPE_TYPE])
    index_mapping["nodes"] = dict(zip(stanet_nrs, junction_indices))


def create_valve_and_pipe(net, stored_data, index_mapping, net_params, stanet_like_valves, add_layers):
    """
    Creates pandapipes valves and pipes from STANET data.
    :param net:
    :type net:
    :param stored_data:
    :type stored_data:
    :param index_mapping:
    :type index_mapping:
    :param net_params:
    :type net_params:
    :param stanet_like_valves:
    :type stanet_like_valves:
    :param add_layers:
    :type add_layers:
    :return:
    :rtype:
    """
    if "valves" not in stored_data:
        return
    logger.info("Creating all vallves with their pipes.")
    node_mapping = index_mapping["nodes"]
    valves = stored_data['valves']
    for row in valves.itertuples():
        valve_name = str(row.STANETID)
        from_stanet_nr, to_stanet_nr = int(row.ANFNR), int(row.ENDNR)
        from_name, to_name = str(row.ANFNAM), str(row.ENDNAM)
        contained = [from_stanet_nr in node_mapping, to_stanet_nr in node_mapping]
        if not all(contained):
            if not contained[0]:
                logger.warning("The valve and pipe %s cannot be created, because the from junction"
                               " %s (%d) is missing in the pandapipes net."
                               % (valve_name, from_name, from_stanet_nr))
            if not contained[1]:
                logger.warning("The valve and pipe %s cannot be created, because the to junction"
                               " %s (%d) is missing in the pandapipes net."
                               % (valve_name, to_name, to_stanet_nr))
            continue
        add_info = dict()
        if add_layers:
            add_info["stanet_layer"] = str(row.LAYER)
        if stanet_like_valves:
            create_valve_pipe_from_parameters(
                net, node_mapping[from_stanet_nr], node_mapping[to_stanet_nr],
                length_km=row.RORL / 1000, diameter_m=float(row.DM / 1000), k_mm=row.RAU,
                opened=row.AUF == 'J', loss_coefficient=row.ZETA,
                name="valve_pipe_%s_%s" % (row.ANFNAM, row.ENDNAM), in_service=bool(row.ISACTIVE),
                stanet_nr=int(row.RECNO), stanet_id=str(row.STANETID),  v_stanet=row.VM, **add_info
            )
        else:
            j_ref = net.junction.loc[node_mapping[from_stanet_nr], :]
            j_ref_geodata = net.junction_geodata.loc[node_mapping[from_stanet_nr], :]
            j_aux = pandapipes.create_junction(
                net, np.NaN, tfluid_k=net_params["medium_temp_K"], height_m=j_ref['height_m'],
                name='aux_' + j_ref['stanet_id'], geodata=(j_ref_geodata.x, j_ref_geodata.y),
                stanet_nr=-999, stanet_id='aux_' + j_ref['stanet_id'], p_stanet=np.NaN,
                stanet_active=bool(row.ISACTIVE), **add_info
            )
            pandapipes.create_pipe_from_parameters(
                net, node_mapping[from_stanet_nr], j_aux, length_km=row.RORL / 1000,
                diameter_m=float(row.DM / 1000), k_mm=row.RAU, loss_coefficient=row.ZETA,
                name="pipe_%s_%s" % (str(row.ANFNAM), 'aux_' + str(row.ENDNAM)),
                in_service=bool(row.ISACTIVE), stanet_nr=-999,
                stanet_id='pipe_valve_' + str(row.STANETID), v_stanet=row.VM,
                stanet_active=bool(row.ISACTIVE), stanet_valid=False, **add_info
            )
            pandapipes.create_valve(
                net, j_aux, node_mapping[to_stanet_nr], diameter_m=float(row.DM / 1000),
                opened=row.AUF == 'J', loss_coefficient=0,
                name="valve_%s_%s" % ('aux_' + str(row.ENDNAM), str(row.ENDNAM)),
                stanet_nr=int(row.RECNO), stanet_id=str(row.STANETID), v_stanet=np.NaN,
                stanet_active=bool(row.ISACTIVE), **add_info
            )


def create_slider_valves(net, stored_data, index_mapping, add_layers):
    """
    Creates pandapipes slider valves from STANET data.
    :param net:
    :type net:
    :param stored_data:
    :type stored_data:
    :param index_mapping:
    :type index_mapping:
    :param add_layers:
    :type add_layers:
    :return:
    :rtype:
    """
    if "slider_valves" not in stored_data:
        return
    logger.info("Creating all slider valves.")
    slider_valves = stored_data["slider_valves"]

    # identify all junctions that are connected on each side of the slider valves
    svf = index_mapping["slider_valves_from"]
    svt = index_mapping["slider_valves_to"]
    from_junctions = np.array([svf[sv] for sv in slider_valves.RECNO.values])
    to_junctions = np.array([svt[sv] for sv in slider_valves.RECNO.values])

    # these types can be converted to normal valves
    # --> there are many types of slider valves in STANET, the behavior is not always clear, so
    #     if you want to convert another type, identify the correct valve behavior in pandapipes
    #     that matches this type.
    opened_sv = [2, 6, 10, 18]
    closed_sv = [3, 7, 11, 19]
    opened_types = {o: True for o in opened_sv}
    opened_types.update({c: False for c in closed_sv})
    sv_types = set(slider_valves.TYP.values.astype(np.int32))
    if len(sv_types - set(opened_types.keys())):
        raise UserWarning("The slider valve types %s cannot be converted."
                          % (sv_types - set(opened_types.keys())))

    # create all slider valves --> most important are the opened and loss_coefficient entries
    valve_system = slider_valves.CLIENTTYP.replace(CLIENT_TYPES_OF_PIPES).values
    add_info = dict()
    if add_layers:
        add_info["stanet_layer"] = slider_valves.LAYER.values.astype(str)
    # account for sliders with diameter 0 m
    if any(slider_valves.DM == 0):
        logger.warning(f"{sum(slider_valves.DM == 0)} sliders have a inner diameter of 0 m! "
                       f"The diameter will be set to 1 m.")
        slider_valves.DM[slider_valves.DM == 0] = 1e3
    pandapipes.create_valves(
        net, from_junctions, to_junctions, slider_valves.DM.values / 1000,
        opened=slider_valves.TYP.astype(np.int32).replace(opened_types).values,
        loss_coefficient=slider_valves.ZETA.values, name=slider_valves.STANETID.values,
        type="slider_valve_" + valve_system, stanet_nr=slider_valves.RECNO.values.astype(np.int32),
        stanet_id=slider_valves.STANETID.values.astype(str), stanet_system=valve_system,
        stanet_active=slider_valves.ISACTIVE.values.astype(np.bool_), **add_info
    )


# noinspection PyTypeChecker
def create_pumps(net, pump_table, index_mapping, add_layers):
    """
    Creates pandapipes pumps from STANET data.
    :param net:
    :type net:
    :param pump_table:
    :type pump_table:
    :param index_mapping:
    :type index_mapping:
    :param add_layers:
    :type add_layers:
    :return:
    :rtype:
    """
    logger.info("Creating all pumps.")
    node_mapping = index_mapping["nodes"]
    pumps = pump_table
    for row in pumps.itertuples():
        pump_name = str(row.STANETID)
        from_stanet_nr, to_stanet_nr = int(row.ANFNR), int(row.ENDNR)
        from_name, to_name = str(row.ANFNAM), str(row.ENDNAM)
        contained = [from_stanet_nr in node_mapping, to_stanet_nr in node_mapping]
        if not all(contained):
            if not contained[0]:
                logger.warning("The pump %s cannot be created, because the from junction"
                               " %s (%d) is missing in the pandapipes net."
                               % (pump_name, from_name, from_stanet_nr))
            if not contained[1]:
                logger.warning("The pump %s cannot be created, because the to junction"
                               " %s (%d) is missing in the pandapipes net."
                               % (pump_name, to_name, to_stanet_nr))
            continue
        add_info = dict()
        if add_layers:
            add_info["stanet_layer"] = str(row.LAYER)
        pandapipes.create_pump(
            net, node_mapping[from_stanet_nr], node_mapping[to_stanet_nr],
            std_type=row.PUMPENTYP, in_service=bool(row.EIN == 'J'), stanet_nr=int(row.RECNO),
            stanet_id=str(row.STANETID),  ps_stanet=-row.DP, stanet_active=bool(row.ISACTIVE),
            **add_info
        )


def create_control_components(net, stored_data, index_mapping, net_params, add_layers, **kwargs):
    """
    Creates pandapipes controller from STANET data.
    :param net:
    :type net:
    :param stored_data:
    :type stored_data:
    :param index_mapping:
    :type index_mapping:
    :param net_params:
    :type net_params:
    :param add_layers:
    :type add_layers:
    :param kwargs:
    :return:
    :rtype:
    """
    if "controllers" not in stored_data:
        return
    logger.info("Creating control components.")
    control_table = stored_data["controllers"]
    node_mapping = index_mapping["nodes"]
    from_junctions = np.array([node_mapping[fr] for fr in
                               control_table.ANFNR.values.astype(np.int32)])
    to_junctions = np.array([node_mapping[to] for to in
                             control_table.ENDNR.values.astype(np.int32)])
    stanet_id = control_table.STANETID.astype(str).values
    names = control_table.RNAM.values if "RNAM" in control_table.columns \
        else control_table.STANETID.values
    fully_open = control_table.OFFEN.values == "J"
    fully_closed = control_table.ZU.values == "J"
    flow = control_table.FLUSS.values.astype(np.float64) * net_params["rho"] / 3600

    consider_controlled = kwargs.get("consider_control_status", False)

    if not all([np.all((control_table[col] == "J") | (control_table[col] == "N"))
                for col in ["OFFEN", "ZU", "AKTIV"]]):
        logger.warning("There is an error in the control table! Please check the columns 'OFFEN',"
                       " 'ZU' and 'AKTIV', which should only contain 'J' or 'N'.")

    control_active = (control_table.AKTIV.values == "J").astype(np.bool_)
    if consider_controlled:
        control_active &= fully_open
    in_service = control_table.ISACTIVE.values.astype(np.bool_)
    if consider_controlled:
        in_service &= ~(control_table.ZU.values == "J")

    is_pc = control_table.RTYP.values == "P"
    is_fc = control_table.RTYP.values == "Q"
    if not np.all(is_pc | is_fc):
        raise UserWarning("There are controllers of types %s that cannot be converted!" \
                                  % set(control_table.RTYP.values[~is_pc & ~is_fc]))

    if np.any(is_pc):
        logger.info("Creating pressure controls.")
        add_info = dict()
        if add_layers:
            add_info["stanet_layer"] = control_table.LAYER.values[is_pc].astype(str)
        pandapipes.create_pressure_controls(
            net, from_junctions[is_pc], to_junctions[is_pc], to_junctions[is_pc],
            control_table.PMESS.astype(np.float64).values[is_pc],
            name=names[is_pc],
            in_service=in_service[is_pc],
            control_active=control_active[is_pc],
            loss_coefficient=control_table.ZETA.values.astype(np.float64)[is_pc],
            stanet_nr=control_table.RECNO.values[is_pc].astype(np.int32),
            stanet_id=stanet_id[is_pc],
            dp_stanet=control_table.DIFFP.values.astype(np.float64)[is_pc],
            ploss_stanet=control_table.PLOSS.values.astype(np.float64)[is_pc],
            stanet_status=control_table.RSTATUS.values.astype(str)[is_pc],
            stanet_system=CLIENT_TYPES_OF_PIPES[MAIN_PIPE_TYPE],
            stanet_is_open=fully_open[is_pc],
            stanet_is_closed=fully_closed[is_pc],
            stanet_flow_kgps=flow[is_pc],
            stanet_active=control_table.ISACTIVE.values[is_pc].astype(np.bool_),
            **add_info
        )

        drop_eg = net.ext_grid.loc[net.ext_grid.junction.isin(to_junctions[is_pc])].index
        net.ext_grid.drop(drop_eg, inplace=True)
        net.junction.loc[to_junctions[is_pc], "pn_bar"] = np.NaN
        pandapipes.reindex_elements(net, "ext_grid", np.arange(len(net.ext_grid)))

    if np.any(is_fc):
        logger.info("Creating flow controllers.")
        add_info = dict()
        if add_layers:
            add_info["stanet_layer"] = control_table.LAYER.values[is_fc].astype(str)
        mdot = control_table.QSOLL.values[is_fc].astype(np.float64) * net_params["rho"] / 3600
        # TODO: how to derive a meaningful diameter? Generally, it is not really of importance, even
        #       in pandapipes, it is just an information and a parameter to convert between outer
        #       and inner calculation values.
        diameter = control_table.DN.where(pd.notnull(control_table.DN), 0.5)[is_fc]
        pandapipes.create_flow_controls(
            net, from_junctions[is_fc], to_junctions[is_fc], mdot, diameter,
            name=names[is_fc],
            in_service=in_service[is_fc],
            control_active=control_active[is_fc],
            stanet_nr=control_table.RECNO.values[is_fc].astype(int),
            stanet_id=stanet_id[is_fc],
            stanet_status=control_table.RSTATUS.values.astype(str)[is_fc],
            stanet_system=CLIENT_TYPES_OF_PIPES[MAIN_PIPE_TYPE],
            stanet_is_open=fully_open[is_fc],
            stanet_is_closed=fully_closed[is_fc],
            stanet_flow_kgps=flow[is_fc],
            stanet_active=control_table.ISACTIVE.values[is_fc].astype(np.bool_),
            **add_info
        )

    df_controller_mapping = pd.Series(index=control_table.ENDNR.values.astype(np.int32),
                                      data=control_table.FLUSS.values)
    return df_controller_mapping.groupby(level=0).sum()


def get_connection_types(connection_table):
    """
    Returns the connection types contained in the STANET raw values.

    :param connection_table: table of connections on pipes
    :type connection_table: pd.DataFrame
    :return:
    :rtype:
    """
    extend_from_to = ["slider_valves"]
    connection_types = list(chain.from_iterable([
        [(ct, ct)] if ct not in extend_from_to else [(ct, ct + "_from"), (ct, ct + "_to")]
        for ct in set(connection_table.type)
    ]))
    return extend_from_to, connection_types


def create_junctions_from_connections(net, connection_table, net_params, index_mapping, add_layers):
    """
    Creates pandapipes junctions from STANET connections.
    :param net:
    :type net:
    :param connection_table:
    :type connection_table:
    :param net_params:
    :type net_params:
    :param index_mapping:
    :type index_mapping:
    :param add_layers:
    :type add_layers:
    :return:
    :rtype:
    """
    if connection_table.empty:
        return
    logger.info("Creating junctions for different connections on pipes.")
    houses_in_calculation = net_params["household_results_valid"]
    # there are two types of connections: those that require just one additional junction and those
    # that require two additional junctions (inserting a branch component there), which is why we
    # need "from" and "to" junctions for those connection types
    extend_from_to, connection_types = get_connection_types(connection_table)
    for con_type, node_type in connection_types:
        cons = connection_table.loc[connection_table.type == con_type]
        stanet_ids = cons.STANETID.astype(str).values
        stanet_nrs = cons.RECNO.astype(np.int32).values
        p_stanet = cons.PRECH.astype(np.float64).values if houses_in_calculation else np.NaN
        names = stanet_ids if con_type not in extend_from_to else \
            stanet_ids + node_type.replace(con_type, "")
        geo = np.array([cons.geo.apply(lambda g: g[0]), cons.geo.apply(lambda g: g[1])]).transpose()
        in_service = np.array([True] * len(cons))
        in_service[cons.CLIENTTYP.values == HOUSE_PIPE_TYPE] = houses_in_calculation
        # types = cons.type.where(~cons.type.str.endswith("s"), cons.type.str[:-1]
        # stanet_active always true?
        add_info = dict()
        if add_layers:
            add_info["stanet_layer"] = cons.LAYER.values.astype(str)
        pp_indices = pandapipes.create_junctions(
            net, len(cons), np.NaN, net_params["medium_temp_K"], name=names,
            height_m=cons.GEOH.astype(np.float64).values, geodata=geo, type=cons.type.values,
            in_service=in_service, stanet_nr=stanet_nrs, stanet_id=stanet_ids, p_stanet=p_stanet,
            stanet_system=cons.CLIENTTYP.replace(CLIENT_TYPES_OF_PIPES).values,
            # TODO: this should be overridden (first add this value to connection table)
            stanet_active=cons.ISACTIVE.values.astype(np.bool_),
            stanet_valid=houses_in_calculation, **add_info
        )
        index_mapping[node_type] = dict(zip(stanet_nrs, pp_indices))


def determine_junctions_from_connection_nodes(pipe_sections, index_mapping):
    """

    :param pipe_sections:
    :type pipe_sections:
    :param index_mapping:
    :type index_mapping:
    :return:
    :rtype:
    """
    pipe_sections["fj"] = -999
    pipe_sections["tj"] = -999

    # TODO: from / to might have to be switched / more precise?
    for typ in set(pipe_sections.from_type.values):
        mapping = index_mapping.get(typ, index_mapping.get(typ + "_from", None))
        if mapping is None:
            raise UserWarning("No junction mapping found for connection type %s (types in mapping:"
                              " %s)" % (typ, list(index_mapping.keys())))
        pipe_sections.loc[pipe_sections.from_type == typ, "fj"] = \
            pipe_sections.loc[pipe_sections.from_type == typ, "from_node"].map(mapping)

    for typ in set(pipe_sections.to_type.values):
        mapping = index_mapping.get(typ, index_mapping.get(typ + "_to", None))
        if mapping is None:
            raise UserWarning("No junction mapping found for connection type %s (types in mapping:"
                              " %s)" % (typ, list(index_mapping.keys())))
        pipe_sections.loc[pipe_sections.to_type == typ, "tj"] = \
            pipe_sections.loc[pipe_sections.to_type == typ, "to_node"].map(mapping)


def create_pipes_from_connections(net, stored_data, connection_table, index_mapping, pipe_geodata,
                                  add_layers):
    """
    Creates pandapipes pipes from STANET connections.
    :param net:
    :type net:
    :param stored_data:
    :type stored_data:
    :param connection_table:
    :type connection_table:
    :param index_mapping:
    :type index_mapping:
    :param pipe_geodata:
    :type pipe_geodata:
    :param add_layers:
    :type add_layers:
    :return:
    :rtype:
    """
    if connection_table.empty:
        return
    logger.info("Creating all main pipes considering splitting connections.")

    cons = connection_table.loc[connection_table.CLIENTTYP == MAIN_PIPE_TYPE].sort_values(
        ["SNUM", "pos_on_line"])
    cons.index = np.arange(len(cons))
    pipe_data = stored_data["pipes"].copy()
    pipe_data.index = pipe_data.RECNO.values
    pipe_nums = cons.SNUM.values

    previous_different = np.insert(pipe_nums[1:] != pipe_nums[:-1], 0, True)
    next_different = np.append(pipe_nums[:-1] != pipe_nums[1:], True)
    previous_different_num = np.where(previous_different)[0]
    next_different_num = np.where(next_different)[0]

    pipe_numbers = np.insert(pipe_nums, previous_different_num, pipe_nums[previous_different])
    positions = cons.pos_on_line.values
    start_pos = np.insert(positions, previous_different_num, 0)
    end_pos = np.insert(positions, next_different_num + 1, 1)
    rel_lengths = end_pos - start_pos
    type_from = np.insert(cons.type.values, previous_different_num, "nodes")
    type_to = np.insert(cons.type.values, next_different_num + 1, "nodes")
    con_from = np.insert(cons.RECNO.values, previous_different_num,
                         pipe_data.loc[pipe_nums, "ANFNR"].values[previous_different])
    con_to = np.insert(cons.RECNO.values, next_different_num + 1,
                       pipe_data.loc[pipe_nums, "ENDNR"].values[next_different])
    vm_from = np.insert(cons.VMB.values, next_different_num, cons.VMA.values[next_different_num])
    vm_to = np.insert(cons.VMA.values, next_different_num, cons.VMB.values[next_different_num])
    vm = (vm_from + vm_to) / 2

    pipe_sections = pd.DataFrame({
        "SNUM": pipe_numbers, "rel_length": rel_lengths, "start_pos": start_pos, "end_pos": end_pos,
        "from_type": type_from, "to_type": type_to, "from_node": con_from, "to_node": con_to,
        "full_geo": pipe_geodata.loc[pipe_numbers], "vm": vm,
        "length": rel_lengths * pipe_data.RORL.loc[pipe_numbers].values,
        "aux": np.ones(len(pipe_numbers), dtype=np.int32)
    })
    pipe_sections["section_no"] = pipe_sections.groupby("SNUM").aux.cumsum()

    determine_junctions_from_connection_nodes(pipe_sections, index_mapping)

    def create_geodata_sections(row):
        sub = substring(LineString(row.full_geo), start_dist=row.start_pos, end_dist=row.end_pos,
                        normalized=True)
        if len(sub.coords) == 1:
            return [tuple(sub.coords)[0], tuple(sub.coords)[0]]
        return list(sub.coords)

    pipe_sections["section_geo"] = pipe_sections.apply(create_geodata_sections, axis=1)

    pipes = pipe_data.loc[pipe_numbers, :]
    add_info = dict()
    if add_layers:
        add_info["stanet_layer"] = pipes.LAYER.values.astype(str)
    # TODO: v_stanet might have to be extended by house connections VMA and VMB
    pandapipes.create_pipes_from_parameters(
        net, pipe_sections.fj.values, pipe_sections.tj.values, pipe_sections.length.values / 1000,
        pipes.DM.values / 1000, pipes.RAU.values, pipes.ZETA.values, type="main_pipe",
        stanet_std_type=pipes.ROHRTYP.values, in_service=pipes.ISACTIVE.values,
        name=["pipe_%s_%s_%s" % (nf, nt, sec) for nf, nt, sec in zip(
            pipes.ANFNAM.values, pipes.ENDNAM.values, pipe_sections.section_no.values)],
        stanet_nr=pipes.RECNO.values, stanet_id=pipes.STANETID.values,
        geodata=pipe_sections.section_geo.values, v_stanet=pipe_sections.vm.values,
        stanet_system=CLIENT_TYPES_OF_PIPES[MAIN_PIPE_TYPE],
        stanet_active=pipes.ISACTIVE.values.astype(np.bool_),
        stanet_valid=~pipes.CALCBAD.values.astype(np.bool_),
        **add_info
    )


def create_heat_exchangers(net, stored_data, connection_table, index_mapping, add_layers):
    """
    Creates pandapipes heat exchangers from STANET connections.
    :param net:
    :type net:
    :param stored_data:
    :type stored_data:
    :param connection_table:
    :type connection_table:
    :param index_mapping:
    :type index_mapping:
    :param add_layers:
    :type add_layers:
    :return:
    :rtype:
    """
    if "heat_exchangers" not in stored_data:
        return
    heat_ex_table = stored_data["heat_exchangers"]
    logger.info("Creating all heat exchangers.")
    heat_exchanger = heat_ex_table.loc[~heat_ex_table.RECNO.isin(connection_table.SNUM.values)]
    node_mapping = index_mapping["nodes"]

    for row in heat_exchanger.itertuples():
        ex_name = str(row.STANETID)
        from_stanet_nr, to_stanet_nr = int(row.ANFNR), int(row.ENDNR)
        from_name, to_name = str(row.ANFNAM), str(row.ENDNAM)
        contained = [from_stanet_nr in node_mapping, to_stanet_nr in node_mapping]
        if not all(contained):
            if not contained[0]:
                logger.warning("The heat exchanger %s cannot be created, because the from junction"
                               " %s (%d) is missing in the pandapipes net."
                               % (ex_name, from_name, from_stanet_nr))
            if not contained[1]:
                logger.warning("The heat exchanger %s cannot be created, because the to junction"
                               " %s (%d) is missing in the pandapipes net."
                               % (ex_name, to_name, to_stanet_nr))
            continue

        add_info = dict()
        if add_layers:
            add_info["stanet_layer"] = str(row.LAYER)
        # TODO: there is no qext given!!!
        pandapipes.create_heat_exchanger(
            net, node_mapping[from_stanet_nr], node_mapping[to_stanet_nr], qext_w=0,
            diameter_m=float(row.DM / 1000), loss_coefficient=row.ZETA, std_type=row.ROHRTYP,
            in_service=bool(row.ISACTIVE), name="heat_exchanger_%s_%s" % (row.ANFNAM, row.ENDNAM),
            stanet_nr=int(row.RECNO), stanet_id=str(row.STANETID), v_stanet=row.VM,
            stanet_active=bool(row.ISACTIVE), **add_info
        )


def create_pipes_from_remaining_pipe_table(net, stored_data, connection_table, index_mapping,
                                           pipe_geodata, add_layers):
    """
    
    :param net:
    :type net:
    :param stored_data:
    :type stored_data:
    :param connection_table:
    :type connection_table:
    :param index_mapping:
    :type index_mapping:
    :param pipe_geodata:
    :type pipe_geodata:
    :param add_layers:
    :type add_layers:
    :return:
    :rtype:
    """
    if "pipes" not in stored_data:
        return
    pipe_table = stored_data["pipes"]
    remaining_pipes = pipe_table.loc[~pipe_table.RECNO.isin(
        connection_table.loc[connection_table.CLIENTTYP == MAIN_PIPE_TYPE].SNUM)]
    if len(remaining_pipes) == 0:
        return
    logger.info("Creating all main pipes without intersections from household connections.")
    node_mapping = index_mapping["nodes"]
    pipes_names = remaining_pipes.STANETID.astype(str)
    from_stanet_nr = remaining_pipes.ANFNR.values.astype(np.int32)
    to_stanet_nr = remaining_pipes.ENDNR.values.astype(np.int32)
    from_names, to_names = remaining_pipes.ANFNAM.astype(str), remaining_pipes.ENDNAM.astype(str)
    contained_from = np.isin(from_stanet_nr, list(node_mapping))
    contained_to = np.isin(to_stanet_nr, list(node_mapping))
    if np.any(~contained_from):
        logger.warning("The pipes %s cannot be created, because the from junctions %s (%s) are"
                       " missing in the pandapipes net."
                       % (pipes_names[~contained_from], from_names[~contained_from],
                          from_stanet_nr[~contained_from]))
    if np.any(~contained_to):
        logger.warning("The pipes %s cannot be created, because the to junctions %s (%s) are"
                       " missing in the pandapipes net."
                       % (pipes_names[~contained_to], to_names[~contained_to],
                          to_stanet_nr[~contained_to]))

    valid = contained_from & contained_to
    from_junctions = [node_mapping[jn] for jn in from_stanet_nr[valid]]
    to_junctions = [node_mapping[jn] for jn in to_stanet_nr[valid]]
    p_tbl = remaining_pipes.loc[valid, :]
    geodata = pipe_geodata.loc[p_tbl.RECNO.values].values
    add_info = dict()
    if add_layers:
        add_info["stanet_layer"] = p_tbl.LAYER.values.astype(str)
    pandapipes.create_pipes_from_parameters(
        net, from_junctions, to_junctions, length_km=p_tbl.RORL.values.astype(np.float64) / 1000,
        type="main_pipe", diameter_m=p_tbl.DM.values.astype(np.float64) / 1000,
        loss_coefficient=p_tbl.ZETA.values, stanet_std_type=p_tbl.ROHRTYP.values,
        k_mm=p_tbl.RAU.values, in_service=p_tbl.ISACTIVE.values.astype(np.bool_),
        name=["pipe_%s_%s" % (anf, end) for anf, end in zip(from_names[valid], to_names[valid])],
        stanet_nr=p_tbl.RECNO.values.astype(np.int32),
        stanet_id=p_tbl.STANETID.values.astype(str), v_stanet=p_tbl.VM.values, geodata=geodata,
        stanet_system=CLIENT_TYPES_OF_PIPES[MAIN_PIPE_TYPE],
        stanet_active=p_tbl.ISACTIVE.values.astype(np.bool_),
        stanet_valid=~p_tbl.CALCBAD.values.astype(np.bool_),
        **add_info
    )


def check_connection_client_types(hh_pipes, all_client_types, node_client_types):
    # create pipes for household connections (from house to supply pipe), which is a separate table
    # in the STANET CSV file
    # --> there are many ways how household connections can be created in STANET,
    # therefore this approach might not be sufficient for new data sets, please verify your
    # output network
    if not np.all(hh_pipes.CLIENTTYP.isin(all_client_types)):
        raise UserWarning("The connection types %s can not all be converted for house connections "
                          "pipes." % set(hh_pipes.CLIENTTYP))
    if not np.all(hh_pipes.CLIENT2TYP.isin(all_client_types)):
        raise UserWarning("The connection types %s can not all be converted for house connections "
                          "pipes." % set(hh_pipes.CLIENT2TYP))
    clientnodetype = hh_pipes.CLIENTTYP.isin(node_client_types)
    client2nodetype = hh_pipes.CLIENT2TYP.isin(node_client_types)
    if not np.all(clientnodetype | client2nodetype):
        raise UserWarning(
            f"One of the household connection sides must be connected to a node (type {NODE_TYPE} element)\n"
            f"or a connection (type {HOUSE_CONNECTION_TYPE} element with ID CON...) "
            f"or a house node (type {HOUSE_CONNECTION_TYPE} element). \n"
            f"Please check that the input data is correct. \n"
            f"Check these CLIENTTYP / CLIENT2TYP: "
            f"{set(hh_pipes.loc[~clientnodetype, 'CLIENTTYP'].values) | set(hh_pipes.loc[~client2nodetype, 'CLIENT2TYP'].values)} "
            f"in the HA LEI table (max. 10 entries shown): \n "
            f"{hh_pipes.loc[~clientnodetype & ~client2nodetype].head(10)}"
            )
    return clientnodetype, client2nodetype


def build_house_junctions(net, index_mapping, hh_types, hh_recno, house_table, meter_table,
                          node_table, net_params, houses_in_calculation, add_layers):
    client_is_house = hh_types == HOUSE_TYPE
    if np.any(client_is_house):
        house_nrs = hh_recno[client_is_house]
        # hint: index and RECNO are identical!
        connected_houses = house_table.loc[house_table.index.isin(house_nrs)]
        all_house_meters = meter_table.loc[meter_table.HAUSINDEX.isin(house_nrs)]
        heights_meters = all_house_meters.groupby("HAUSINDEX").GEOH.mean()
        heights_houses = np.array([
            heights_meters[hn] if hn in heights_meters.index else node_table.GEOH.at[con_no]
            for hn, con_no in zip(connected_houses.RECNO.values, connected_houses.KNONUM.values)
        ])
        geodata = np.array([connected_houses.XRECHTS.values,
                            connected_houses.YHOCH.values]).transpose()

        add_info = dict()
        if add_layers:
            add_info["stanet_layer"] = connected_houses.LAYER.values.astype(str)
        # create junctions for houses that are directly connected via house pipes
        pp_ind = pandapipes.create_junctions(
            net, len(connected_houses), np.NaN, tfluid_k=net_params["medium_temp_K"],
            height_m=heights_houses.astype(np.float64), geodata=geodata,
            in_service=houses_in_calculation,
            name=["house_%s" % hn for hn in connected_houses.RECNO.values],
            type="house", stanet_nr=connected_houses.RECNO.values,
            stanet_id=connected_houses.STANETID.values.astype(str),
            p_stanet=connected_houses.PRECH.values if houses_in_calculation else np.NaN,
            stanet_system=CLIENT_TYPES_OF_PIPES[HOUSE_PIPE_TYPE],
            stanet_active=connected_houses.ISACTIVE.values.astype(np.bool_),
            stanet_valid=houses_in_calculation, **add_info
        )
        index_mapping["houses"] = dict(zip(connected_houses.index.values, pp_ind))


def build_meter_junctions(net, index_mapping, hh_types, hh_recno, meter_table, net_params,
                          houses_in_calculation, add_layers):
    client_is_meter = hh_types == METER_TYPE
    if np.any(client_is_meter):
        meter_nrs = hh_recno[client_is_meter]
        # hint: index and RECNO are identical!
        connected_meters = meter_table.loc[meter_table.index.isin(meter_nrs)]
        geodata = np.array([connected_meters.XRECHTS.values,
                            connected_meters.YHOCH.values]).transpose()

        add_info = dict()
        if add_layers:
            add_info["stanet_layer"] = connected_meters.LAYER.values.astype(str)
        # create junctions for meters that are directly connected via house pipes
        pp_ind = pandapipes.create_junctions(
            net, len(connected_meters), np.NaN, tfluid_k=net_params["medium_temp_K"],
            height_m=connected_meters.GEOH.values.astype(np.float64),
            name=connected_meters.STANETID.values.astype(str), geodata=geodata, type="meter",
            stanet_nr=connected_meters.RECNO.values,
            stanet_id=connected_meters.STANETID.astype(str),
            in_service=houses_in_calculation,
            p_stanet=connected_meters.PRECH.values if houses_in_calculation else np.NaN,
            stanet_system=CLIENT_TYPES_OF_PIPES[HOUSE_PIPE_TYPE],
            stanet_active=connected_meters.ISACTIVE.values.astype(np.bool_),
            stanet_valid=houses_in_calculation, **add_info
        )
        index_mapping["meters"] = dict(zip(connected_meters.index.values, pp_ind))


def build_house_node_junctions(net, index_mapping, stored_data, net_params, houses_in_calculation,
                               add_layers):
    if "house_nodes" not in stored_data:
        return
    house_nodes = stored_data["house_nodes"].copy()
    house_nodes.index = house_nodes.RECNO.values
    house_node_nrs = house_nodes.RECNO.values
    geodata = np.array([house_nodes.XRECHTS.values, house_nodes.YHOCH.values]).transpose()
    add_info = dict()
    if add_layers:
        add_info["stanet_layer"] = house_nodes.LAYER.values.astype(str)
    pp_ind = pandapipes.create_junctions(
        net, len(house_node_nrs), np.NaN, tfluid_k=net_params["medium_temp_K"],
        height_m=house_nodes.GEOH.values.astype(np.float64),
        name=house_nodes.STANETID.values.astype(str), geodata=geodata,
        type="house_node", stanet_nr=house_node_nrs,
        stanet_id=house_nodes.STANETID.astype(str), in_service=houses_in_calculation,
        p_stanet=house_nodes.PRECH.values if houses_in_calculation else np.NaN,
        stanet_system=CLIENT_TYPES_OF_PIPES[HOUSE_PIPE_TYPE],
        stanet_active=house_nodes.ISACTIVE.values.astype(np.bool_),
        stanet_valid=houses_in_calculation, **add_info
    )
    index_mapping["house_nodes"] = dict(zip(house_node_nrs, pp_ind))


def get_house_pipe_entries(hh_pipes, index_mapping):
    # client types: see PIPE_CLIENT_TYPES
    all_client_types = [NODE_TYPE, HOUSE_TYPE, HOUSE_CONNECTION_TYPE, METER_TYPE, HOUSE_NODE_TYPE]
    node_client_types = [NODE_TYPE, HOUSE_CONNECTION_TYPE, HOUSE_NODE_TYPE]
    clientnodetype, client2nodetype = check_connection_client_types(hh_pipes, all_client_types,
                                                                    node_client_types)

    # only use pipes that connect to a house or a meter
    pipes_between_nodes = clientnodetype & client2nodetype
    hh_pipes_to_houses = hh_pipes.loc[~pipes_between_nodes]

    # identify client types for hh and node
    client_types = hh_pipes_to_houses.CLIENTTYP.values
    client_recno = hh_pipes_to_houses.CLIENTNO.values
    client_is_node = clientnodetype[~pipes_between_nodes].values

    hh_types, con_types = client_types.copy(), client_types.copy()
    hh_recno, con_recno = client_recno.copy(), client_recno.copy()

    # identify hh types -> if client 1 is the node, then use client 2 type as hh type
    hh_types[client_is_node] = hh_pipes_to_houses.CLIENT2TYP.values[client_is_node]
    hh_recno[client_is_node] = hh_pipes_to_houses.CLIENT2NO.values[client_is_node]

    # identify connection types -> choose the respective other side than before
    con_types[~client_is_node] = hh_pipes_to_houses.CLIENT2TYP.values[~client_is_node]
    con_recno[~client_is_node] = hh_pipes_to_houses.CLIENT2NO.values[~client_is_node]

    # get indices of pandapipes junctions that the pipe is connected to on the other side
    con_pp_no = np.ones(len(con_types), dtype=np.int32) * (-1)
    connection_types = {num: ct for num, ct in CLIENT_TYPES_OF_NODES.items()
                        if num in node_client_types}
    for con_num, con_name in connection_types.items():
        if con_num not in con_types:
            continue
        mapping = index_mapping[con_name]
        con_pp_no[con_types == con_num] = [mapping[cn] for cn in con_recno[con_types == con_num]]

    if len(set(con_types) - set(connection_types.keys())) > 0:
        raise UserWarning("The connection types %s have not been converted!"
                          % (set(con_types) - set(connection_types.keys())))

    # house connection types that don't appear in the client type list will raise a warning
    invalid_hh_types = ~np.isin(hh_types, list(set(all_client_types) - set(node_client_types)))
    if np.any(invalid_hh_types):
        raise UserWarning(
            "It is not possible to create node from client 2 with types %s (IDs: %s) for house "
            "connections." % (set(hh_types[invalid_hh_types]),
                              hh_pipes_to_houses.CLIENT2ID.values[invalid_hh_types])
        )

    return client_types, hh_types, hh_recno, con_recno, con_pp_no


def get_tables_in_stanet_indices(stored_data, connection_table, house_table, meter_table):
    hh_pipes = stored_data["house_pipes"]
    node_table = stored_data["nodes"].copy()
    node_table.index = node_table.RECNO.values
    hh_connections = connection_table.loc[connection_table.type == "house_connections"].copy()
    hh_connections.index = hh_connections.RECNO.values
    houses = house_table.copy()
    houses.index = houses.RECNO.values
    meters = meter_table.copy()
    meters.index = meters.RECNO.values
    return hh_pipes, node_table, hh_connections, houses, meters


def create_nodes_house_connections(net, stored_data, connection_table, meter_table, house_table,
                                   index_mapping, net_params, add_layers):
    """

    :param net:
    :type net:
    :param stored_data:
    :type stored_data:
    :param connection_table:
    :type connection_table:
    :param meter_table:
    :type meter_table:
    :param house_table:
    :type house_table:
    :param index_mapping:
    :type index_mapping:
    :param net_params:
    :type net_params:
    :param add_layers:
    :type add_layers:
    :return:
    :rtype:
    """
    if "house_pipes" not in stored_data:
        return

    logger.info("Creating junctions for houses and other house connection infrastructure.")

    # we need the house connections, houses, meters and nodes to create all house pipes and their
    # junctions on the way from the main pipes
    hh_pipes, node_table, hh_connections, houses, meters = \
        get_tables_in_stanet_indices(stored_data, connection_table, house_table, meter_table)
    houses_in_calculation = net_params["household_results_valid"]

    # create junctions for house_nodes
    build_house_node_junctions(net, index_mapping, stored_data, net_params, houses_in_calculation,
                               add_layers)

    # important information for junction creation
    client_types, hh_types, hh_recno, con_recno, con_pp_no = \
        get_house_pipe_entries(hh_pipes, index_mapping)

    # create junctions for houses
    build_house_junctions(net, index_mapping, hh_types, hh_recno, houses,
                          meters, node_table, net_params, houses_in_calculation, add_layers)

    # create junctions for meters (without houses)
    build_meter_junctions(net, index_mapping, hh_types, hh_recno, meters, net_params,
                          houses_in_calculation, add_layers)


def create_pipes_house_connections(net, stored_data, connection_table, index_mapping,
                                   house_pipe_geo, houses_in_calculation, add_layers):
    if "house_pipes" not in stored_data:
        return

    logger.info("Creating pipes for house connection infrastructure.")
    cons = connection_table.loc[connection_table.CLIENTTYP == HOUSE_PIPE_TYPE].sort_values(
        ["SNUM", "pos_on_line"])
    cons.index = np.arange(len(cons))

    pipe_data = stored_data["house_pipes"].copy()
    pipe_data.index = pipe_data.RECNO.values
    cons_on_line = cons.SNUM.value_counts()

    pipe_data["no_cons"] = 0
    pipe_data.loc[cons_on_line.index.values, "no_cons"] = cons_on_line.values
    pipe_data["from_type"] = pipe_data.CLIENTTYP.replace(CLIENT_TYPES_OF_NODES)
    pipe_data["to_type"] = pipe_data.CLIENT2TYP.replace(CLIENT_TYPES_OF_NODES)
    pipe_data["start_pos"] = 0.
    pipe_data["end_pos"] = 1.
    pipe_data["from_node"] = pipe_data.CLIENTNO.copy()
    pipe_data["to_node"] = pipe_data.CLIENT2NO.copy()

    hp_data = pd.DataFrame({col: np.repeat(pipe_data[col].values, pipe_data.no_cons.values + 1)
                            for col in pipe_data.columns})
    hp_data["section_no"] = vrange(np.ones(len(pipe_data), dtype=np.int32),
                                   pipe_data.no_cons.values + 1)
    hp_data.sort_index(inplace=True)

    has_con = hp_data.no_cons > 0
    first_section = hp_data.section_no == 1
    last_section = hp_data.section_no == hp_data.no_cons + 1
    not_first_con = has_con & ~first_section
    not_last_con = has_con & ~last_section

    hp_data.loc[not_first_con, "from_type"] = cons.type.values
    hp_data.loc[not_last_con, "to_type"] = cons.type.values
    hp_data.loc[not_first_con, "start_pos"] = cons.pos_on_line.values
    hp_data.loc[not_last_con, "end_pos"] = cons.pos_on_line.values
    hp_data.loc[not_first_con, "from_node"] = cons.RECNO.values
    hp_data.loc[not_last_con, "to_node"] = cons.RECNO.values
    hp_data["length_rel"] = hp_data.end_pos - hp_data.start_pos
    hp_data["length"] = hp_data.length_rel * hp_data.RORL

    determine_junctions_from_connection_nodes(hp_data, index_mapping)
    hp_data["full_geo"] = house_pipe_geo.loc[hp_data.RECNO.values].values
    hp_data["section_geo"] = hp_data["full_geo"].values

    def create_geodata_sections(row):
        sub = substring(LineString(row.full_geo), start_dist=row.start_pos, end_dist=row.end_pos,
                        normalized=True)
        if len(sub.coords) == 1:
            return [tuple(sub.coords)[0], tuple(sub.coords)[0]]
        return list(sub.coords)

    hp_data.loc[has_con, "section_geo"] = hp_data.loc[has_con].apply(create_geodata_sections,
                                                                     axis=1)

    add_info = dict()
    if add_layers:
        add_info["stanet_layer"] = hp_data.LAYER.values.astype(str)
    # TODO: v_stanet might have to be extended by house connections VMA and VMB
    pandapipes.create_pipes_from_parameters(
        net, hp_data.fj.values, hp_data.tj.values, hp_data.length.values / 1000,
        hp_data.DM.values / 1000, hp_data.RAU.values, hp_data.ZETA.values, type="house_pipe",
        stanet_std_type=hp_data.ROHRTYP.values,
        in_service=hp_data.ISACTIVE.values if houses_in_calculation else False,
        name=["pipe_%s_%s_%s" % (nf, nt, sec) for nf, nt, sec in zip(
            hp_data.CLIENTID.values, hp_data.CLIENT2ID.values, hp_data.section_no.values)],
        stanet_nr=hp_data.RECNO.values, stanet_id=hp_data.STANETID.values,
        geodata=hp_data.section_geo.values, v_stanet=hp_data.VM.values,
        stanet_active=hp_data.ISACTIVE.values.astype(np.bool_),
        stanet_valid=houses_in_calculation, **add_info
    )


def create_sinks_meters(net, meter_table, index_mapping, net_params, add_layers):
    """

    :param net:
    :type net:
    :param meter_table:
    :type meter_table:
    :param index_mapping:
    :type index_mapping:
    :param net_params:
    :type net_params:
    :param add_layers:
    :type add_layers:
    :return:
    :rtype:
    """
    if meter_table.empty:
        return
    logger.info("Creating sinks/sources for meters.")
    houses_in_calculation = net_params["household_results_valid"]
    meter_mapping = index_mapping.get("meters", None)
    house_mapping = index_mapping.get("houses", None)
    node_mapping = index_mapping["nodes"]

    assigned_node_nums = meter_table.KNONUM.astype(np.int32)
    meter_nrs = meter_table.RECNO.astype(np.int32)
    junctions_assigned = pd.Series([node_mapping.get(nn, np.NaN) for nn in assigned_node_nums],
                                   index=meter_table.index, dtype=float)

    junctions_connected = pd.Series(index=meter_table.index, dtype=float)
    if meter_mapping is not None:
        meters_with_nodes = meter_nrs.isin(meter_mapping.keys())
        junctions_connected.loc[meters_with_nodes] = [meter_mapping[mn] for mn in
                                                      meter_nrs.loc[meters_with_nodes].values]
    else:
        meters_with_nodes = pd.Series(False, index=meter_table.index)

    if house_mapping is not None:
        meters_with_house_nodes = meter_table.HAUSINDEX.isin(house_mapping.keys())\
                                  & ~meters_with_nodes
        junctions_connected[meters_with_house_nodes] = \
            [house_mapping[hn] for hn in meter_table.HAUSINDEX.loc[meters_with_house_nodes].values]

    if np.any(junctions_connected.isnull()):
        junctions_connected.loc[junctions_connected.isnull()] = \
            [node_mapping.get(n, np.NaN) if not np.isnan(n) else np.NaN for n in
             meter_table.KNONUM.loc[junctions_connected.isnull()].values]
        logger.warning("The meters %s cannot be mapped to a house connection node and will be "
                       "assigned to nodes %s"
                       % (junctions_connected[junctions_connected.isnull()].index,
                          junctions_connected[junctions_connected.isnull()].values))

    if np.any(junctions_connected.isnull()):
        logger.warning("The meters %s are not connected to any node and will be skipped."
                       % junctions_connected.index[junctions_connected.isnull()])

    junctions_to_use = junctions_connected if houses_in_calculation else junctions_assigned
    try:
        junctions = junctions_to_use.astype(np.int32)
    except pd.errors.IntCastingNaNError:
        logger.warning("The connected junctions of the meter sinks will be given as floats, "
                       "as there are NaN values.")
        junctions = junctions_to_use.where(pd.notnull(junctions_to_use), -1).astype(np.int32)

    sources = (meter_table.ZUFLUSS.values > 0) & junctions_connected.notnull().values
    sinks = (meter_table.ZUFLUSS <= 0) & junctions_connected.notnull().values
    if np.any(sources):
        logger.info("Creating all sources.")
        source_tbl = meter_table.loc[sources, :]
        stanet_id = source_tbl.STANETID.astype(str).values
        add_info = dict()
        if add_layers:
            add_info["stanet_layer"] = source_tbl.LAYER.values.astype(str)
        pandapipes.create_sources(
            net, junctions.loc[sources], source_tbl.ZUFLUSS.values / 3600 * net_params['rho'],
            name=stanet_id, stanet_id=stanet_id, stanet_nr=source_tbl.RECNO.astype(np.int32).values,
            stanet_assigned_node=source_tbl.KNONUM.values.astype(np.int32),
            stanet_junction_connected=junctions_connected.loc[sources].values.astype(np.int32),
            stanet_junction_alternative=junctions_assigned.loc[sources].values.astype(np.int32),
            stanet_active=source_tbl.ISACTIVE.values.astype(np.bool_),
            profile_id=source_tbl.PROFIL.astype(str).values,
            demand_m3_per_a=source_tbl.VERBRAUCH.values.astype(np.float64), **add_info
        )
    if np.any(sinks):
        logger.info("Creating all sinks.")
        sink_tbl = meter_table.loc[sinks, :]
        stanet_id = sink_tbl.STANETID.astype(str).values
        add_info = dict()
        if add_layers:
            add_info["stanet_layer"] = sink_tbl.LAYER.values.astype(str)
        pandapipes.create_sinks(
            net, junctions.loc[sinks].values,
            sink_tbl.ZUFLUSS.values / 3600 * net_params['rho'] * (-1), type="meter",
            name=stanet_id, stanet_id=stanet_id, stanet_nr=sink_tbl.RECNO.astype(str).values,
            stanet_assigned_node=sink_tbl.KNONUM.values.astype(np.int32),
            stanet_junction_connected=junctions_connected.loc[sinks].values.astype(np.int32),
            stanet_junction_alternative=junctions_assigned.loc[sinks].values.astype(np.int32),
            stanet_active=sink_tbl.ISACTIVE.values.astype(np.bool_),
            profile_id=sink_tbl.PROFIL.astype(str).values,
            demand_m3_per_a=sink_tbl.VERBRAUCH.values.astype(np.float64), **add_info)


def create_sinks_from_nodes(net, node_table, index_mapping, net_params, sinks_defined,
                            control_flows, add_layers):
    """

    :param net:
    :type net:
    :param node_table:
    :type node_table:
    :param index_mapping:
    :type index_mapping:
    :param net_params:
    :type net_params:
    :param sinks_defined:
    :type sinks_defined:
    :param control_flows:
    :type control_flows:
    :param add_layers:
    :type add_layers:
    :return:
    :rtype:
    """
    if sinks_defined:
        return
    logger.info("Creating sinks for fixed feed-in or consumption nodes.")

    node_table = node_table.copy()
    node_table.index = node_table.RECNO.values
    fixed_p_nodes = node_table.DSTATUS.values == "!"
    unknown_flow_nodes = node_table.FSTATUS.values == "?"
    if np.any(fixed_p_nodes != unknown_flow_nodes):
        raise UserWarning("The following nodes cannot be interpreted correctly, as they have a "
                          "fixed pressure, but not unknown flow: %s"
                          % node_table.RECNO.values[fixed_p_nodes != unknown_flow_nodes])


    node_table.loc[control_flows.index, "ZUFLUSS"] -= control_flows.values

    nodes_with_source = (node_table.ZUFLUSS.values > 0) & ~fixed_p_nodes
    nodes_with_sink = (node_table.ZUFLUSS.values < 0) & ~fixed_p_nodes

    junctions = np.array([index_mapping["nodes"][nn] for nn in node_table.RECNO.values])
    flow = node_table.ZUFLUSS.values / 3600 * net_params["rho"]

    try:
        add_info = {"stanet_id": node_table.STANETID.values.astype(str)}
    except AttributeError:
        add_info = dict()
    if add_layers:
        add_info["stanet_layer"] = node_table.LAYER.values.astype(str)

    if np.any(nodes_with_source):
        names_source = np.array(["source_node_%s" % knam for knam in
                                 node_table.KNAM.values[nodes_with_source]])
        pandapipes.create_sources(
            net, junctions[nodes_with_source], mdot_kg_per_s=flow[nodes_with_source],
            name=names_source, stanet_nr=node_table.RECNO.values[nodes_with_source],
            stanet_active=node_table.ISACTIVE.values[nodes_with_source].astype(np.bool_),
            **{k: v[nodes_with_source] for k, v in add_info.items()})

    if np.any(nodes_with_sink):
        names_sink = np.array(["sink_node_%s" % knam for knam in
                               node_table.KNAM.values[nodes_with_sink]])
        pandapipes.create_sinks(
            net, junctions[nodes_with_sink], mdot_kg_per_s=flow[nodes_with_sink] * (-1),
            name=names_sink, stanet_nr=node_table.RECNO.values[nodes_with_sink],
            stanet_active=node_table.ISACTIVE.values[nodes_with_sink].astype(np.bool_),
            **{k: v[nodes_with_sink] for k, v in add_info.items()})


def create_sinks_from_customers(net, stored_data, index_mapping, net_params, sinks_defined,
                                add_layers):
    if "customers" not in stored_data:
        return sinks_defined
    logger.info("Creating sinks for customers.")
    customer_table = stored_data["customers"]
    node_mapping = index_mapping["nodes"]
    assigned_node_nums = customer_table.KNONUM.astype(np.int32)

    junctions = assigned_node_nums.apply(lambda nn: node_mapping[nn]).values.astype(np.int32)
    customer_nrs = customer_table.RECNO.values.astype(np.int32)
    customer_ids = customer_table.STANETID.astype(str).values

    add_info = dict()
    if add_layers:
        add_info["stanet_layer"] = customer_table.LAYER.values.astype(str)
    pandapipes.create_sinks(
        net, junctions,
        customer_table.ABFLUSS.values.astype(np.float64) / 3600 * net_params['rho'] * (-1),
        name=customer_ids, in_service=customer_table.ISACTIVE.values.astype(np.bool_),
        type="customer", stanet_id=customer_ids, stanet_nr=customer_nrs,
        stanet_active=customer_table.ISACTIVE.values.astype(np.bool_),
        profile_id=customer_table.PROFIL.astype(str).values,
        demand_m3_per_a=customer_table.VERBRAUCH.values.astype(np.float64), **add_info)
    return True
