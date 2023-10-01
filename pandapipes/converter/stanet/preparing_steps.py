# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

from itertools import chain, product

import numpy as np
import pandas as pd

import pandapipes
from pandapipes.converter.stanet.table_creation import CLIENT_TYPES_OF_NODES
from pandapipes.properties.fluids import FluidPropertySutherland, _add_fluid_to_net

try:
    from shapely.geometry import Point, LineString
    SHAPELY_INSTALLED = True
except ImportError:
    SHAPELY_INSTALLED = False

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def get_stanet_raw_data(stanet_path, read_options=None, add_layers=True, return_line_info=False,
                        keywords=None):
    """
    Extract raw data from STANET file.
    :param stanet_path:  Path to STANET .csv file
    :type stanet_path: string
    :param read_options:
    :type read_options:
    :param add_layers:
    :type add_layers: bool, default:True
    :param return_line_info:
    :type return_line_info: bool, default:False
    :param keywords:
    :type keywords:
    :return: stored data
    :rtype: dic
    """
    read_options = read_options if read_options is not None else dict()
    if keywords is None:
        # the given keywords will be sought in the STANET CSV file to identify tables according to
        # the following rule:
        # 1st line: keyword (e.g. "REM Leitungsdaten")
        # 2nd line: human-readable header of table
        # 3rd line: units
        # 4th line: STANET internal header of table (this is the first line to convert to pandas
        #           dataframe and return to the converter)
        # everything until the next empty line will be added to the dataframe
        keywords = {"pipes": ['REM Leitungsdaten'],
                    "house_pipes": ['REM HA Leitungsdaten'],
                    "nodes": ['REM Knotendaten'],
                    "house_nodes": ["REM HA Knotendaten"],
                    "valves": ['REM Ventiledaten'],
                    "pumps_gas": ['REM Kompressorendaten'],
                    "pumps_water": ['REM Pumpendaten'],
                    "net_parameters": ['REM Netzparameterdaten'],
                    "houses": ["REM Hausdaten"],
                    "house_connections": ["REM HA Verbindungsdaten"],
                    "meters": ["REM HA Zählerdaten"],
                    "controllers": ["REM Reglerdaten"],
                    "slider_valves": ["REM Schieberdaten"],
                    "inflexion_points": ["REM Knickpunktdaten"],
                    "heat_exchangers": ["REM Wärmetauscherdaten"],
                    "customers": ["REM Abnehmerdaten"],
                    "house_inflexion_points": ["REM HA Knickpunktdaten"],
                    "layers": ["REM Layerdaten"]}
    stored_data = dict()

    logger.info("Reading STANET csv-file.")
    # read full csv file
    read_line_info = dict()
    encoding = read_options.get("encoding", "cp1252")
    with open(stanet_path, 'rt', encoding=encoding) as f:
        all_lines = pd.Series(f.readlines())

    last_count = len(all_lines)
    # list of all empty lines in file (important to identify table ends)
    newlines = all_lines[all_lines == "\n"].index
    # insert an empty line at the very end, as the file does not always contain an empty line at
    # the end
    newlines = newlines.insert(len(newlines), last_count)

    # search for keywords to identify all lines belonging to respective tables
    for key, keyword_list in keywords.items():
        for keyword in keyword_list:
            has_key = all_lines.str.startswith(keyword)
            no_key = np.sum(has_key)
            if no_key == 0:
                logger.debug("No table %s in data." % keyword)
                continue
            if no_key > 1:
                logger.warning("%s occurs several times in data. Only using one occurence."
                               % keyword)
            # the table starts 3 lines after the keyword and ends with the next empty line
            start = has_key[has_key].index[0] + 3
            end = newlines[newlines > start][0]
            if not end > start:
                raise UserWarning("Something went wrong in reading the data. For key %s, the first"
                                  " found line is %s and the last one %s." % (key, start, end))
            read_line_info[key] = [start, end]

    # read tables separately by calling pd.read_csv(path, skiprows=[all rows except those between
    #                                                               start and end of respective
    #                                                               table])
    rows = list(range(last_count + 1))
    for key, (from_line, to_line) in read_line_info.items():
        read_args = read_options.get("global", dict())
        read_args.update(read_options.get(key, dict()))
        logger.debug("Reading CSV table %s into pandas." % key)
        data = pd.read_csv(stanet_path, encoding=encoding, sep=';', index_col=False,
                           skiprows=rows[:from_line] + rows[to_line:], **read_args)
        data.columns = [col[1:] if isinstance(col, str) and col.startswith("!")
                        else col for col in data.columns]
        stored_data[key] = data
        if add_layers:
            if "LAYER" not in stored_data[key].columns \
                    and key not in ["layers", "inflexion_points", "house_inflexion_points"]:
                stored_data[key]["LAYER"] = [None] * len(stored_data[key])

    # check which data could not be exported (the irrelevant data is expected not to be exported
    # anyway)
    irrelevant_data = {"REM Meldungendaten"}
    if not add_layers:
        irrelevant_data |= {"REM Layerdaten"}
    table_lines = all_lines.str.startswith("REM ") & all_lines.str.endswith("daten\n")
    remaining_lines = set(all_lines[table_lines].str.replace("\n", ""))\
        - set(chain.from_iterable(keywords.values())) - irrelevant_data
    if len(remaining_lines):
        logger.warning("The following table data cannot be converted and will be neglected: \n"
                       + "\n".join([li.replace("REM ", "") for li in remaining_lines]))

    if return_line_info:
        return stored_data, read_line_info
    return stored_data


def get_key_from_value(val, used_dict):
    """
    Reversed mapping operation.

    :param val:
    :type val:
    :param used_dict:
    :type used_dict:
    :return:
    :rtype:
    """
    for key, value in used_dict.items():
        if val == value:
            return key


def get_net_params(net, stored_data):
    """
    Returns pandapipesNet Parameters from STANET data.
    :param net: Empty pandapipesNet
    :type net: pandapipesNet
    :param stored_data: dict of STANET tables
    :type stored_data: dict
    :return: net parameters
    :rtype:
    """
    known_friction_models = {1: "swamee-jain", 3: "nikuradse", 5: "colebrook"}
    compressibility_models = {0: "linear", 1: "AGA", 2: "GERG-88"}
    net_params = dict()
    net_data = stored_data["net_parameters"]
    net_params['medium'] = net_data.at[0, "MEDI"]
    net_params["rho"] = net_data.at[0, "RHON"]
    net_params["cp"] = net_data.at[0, "CG"]
    net_params["eta"] = net_data.at[0, "ETA"] * 1e-6
    net_params["t_sutherland"] = net_data.at[0, "TS"]
    net_params["t0_sutherland"] = net_data.at[0, "T0"]
    net_params["calculate_temp"] = str(net_data.at[0, "TEMPCALC"]) == "J"
    pp_calc_mode = "all" if net_params["calculate_temp"] else "hydraulics"
    pandapipes.set_user_pf_options(net, mode=pp_calc_mode)
    net_params["medium_temp_C"] = net_data.at[0, "TEMP"]
    net_params["medium_temp_K"] = net_data.at[0, "TEMP"] + 273.15
    net_params["calculation_results_valid"] = not bool(net_data.at[0, "CALCDIRTY"])
    net_params["household_results_valid"] = not bool(net_data.at[0, "HCALCDIRTY"])
    net_params["comp_factor"] = net_data.at[0, "KPAR"]
    net_params["friction_model"] = int(net_data.at[0, "LAM"])
    net_params["max_iterations"] = int(net_data.at[0, "IMAX"])
    net_params["compress_model"] = compressibility_models[int(net_data.at[0, "KFAKT"])]
    if net_params["friction_model"] not in known_friction_models.keys():
        known_str = " or ".join("%s (%d)" % (m_name.capitalize(), m_nr)
                                for m_nr, m_name in known_friction_models.items())
        logger.warning("The friction model %d of STANET cannot be modeled in pandapipes. This might"
                       " lead to incorrect results. The possible friction models are %s."
                       % (net_params["friction_model"], known_str))
    else:
        pandapipes.set_user_pf_options(
            net, friction_model=known_friction_models[net_params["friction_model"]],
            iter=net_params["max_iterations"]
        )
    if net_params["compress_model"] != "linear":
        logger.warning("The compressibility model %s is not implemented in pandapipes, which might "
                       "lead to wrong results." % net_params["compress_model"])

    pandapipes.set_user_pf_options(net, ambient_temperature=net_data.at[0, "AIRTEMP"] + 273.15)
    state = 'liquid' if net_params['medium'] == 'W' else 'gas'
    fluid = pandapipes.create_constant_fluid(
        'STANET_fluid', state, density=net_params["rho"], viscosity=net_params["eta"],
        heat_capacity=net_params["cp"], der_compressibility=net_params["comp_factor"])
    if state == "gas" and net_params["t0_sutherland"] != 0:
        viscosity = FluidPropertySutherland(net_params["eta"], net_params["t0_sutherland"],
                                            net_params["t_sutherland"])
        fluid.add_property("viscosity", viscosity, overwrite=True, warn_on_duplicates=False)

    if net_params["comp_factor"] == 0:
        fluid.add_property('compressibility', pandapipes.FluidPropertyConstant(1.),
                           overwrite=True, warn_on_duplicates=False)
    else:
        fluid.add_property(
            'compressibility', pandapipes.FluidPropertyLinear(net_params["comp_factor"], 1),
            overwrite=True, warn_on_duplicates=False)

    _add_fluid_to_net(net, fluid, overwrite=True)
    return net_params


def adapt_pipe_data_according_to_nodes(pipe_data, pipes_to_check, node_geo, pipe_rec, is_x,
                                       is_start, node_type, node_cols, coord_names):
    node_name = "XRECHTS" if is_x else "YHOCH"
    coord = "x" if is_x else "y"
    locat = "from" if is_start else "to"
    run = 0 if is_x else 2
    run += 0 if is_start else 1
    pipe_name = coord_names[run]
    node_nr = node_cols[0] if is_start else node_cols[1]
    node_val = node_geo.loc[pipe_data.loc[pipes_to_check, node_nr].values, node_name].values

    if pipe_name not in pipe_data.columns:
        pipe_data[pipe_name] = np.NaN
        pipe_data.loc[pipes_to_check, pipe_name] = node_val
    current_pipe_data = pipe_data.loc[pipes_to_check]
    if not np.allclose(node_val, current_pipe_data[pipe_name].values):
        wrong_data = ~np.isclose(node_val, current_pipe_data[pipe_name].values)
        df = pd.DataFrame(
            {"pipe_%s_%s" % (coord, locat): current_pipe_data[pipe_name].values[wrong_data],
             "node_%s_%s" % (coord, locat): node_val[wrong_data]}, index=pipe_rec[wrong_data]
        )
        logger.warning("Found diverging geodata between nodes (type %s) and pipe %s points:\n%s"
                       % (node_type, "start" if is_start else "end", df))
        logger.warning("Adapting the pipe geodata to match nodes.")
        adapt_indices = pipe_data.index[pipes_to_check]
        adapt_indices = adapt_indices[wrong_data]
        pipe_data.loc[adapt_indices, pipe_name] = node_val[wrong_data]


def adapt_pipe_data(stored_data, pipe_data, coord_names, use_clients):
    if "CLIENTTYP" not in pipe_data.columns:
        client_types_ind = {1: [pd.Series(True, index=pipe_data.index)] * 2}
    else:
        all_client_types = set(pipe_data.CLIENTTYP.values) | set(pipe_data.CLIENT2TYP.values)
        if len(all_client_types - set(CLIENT_TYPES_OF_NODES.keys())) > 0:
            raise UserWarning("The following node client types cannot be identified: %s"
                              % (all_client_types - set(CLIENT_TYPES_OF_NODES.keys())))
        client_types_ind = {c: [pipe_data.CLIENTTYP == c, pipe_data.CLIENT2TYP == c]
                            for c in all_client_types}

    for client_num, indices in client_types_ind.items():
        node_name = CLIENT_TYPES_OF_NODES[client_num]
        node_data = stored_data[node_name]
        node_geo = node_data.loc[:, ["XRECHTS", "YHOCH"]]
        node_geo.index = node_data.RECNO.values
        node_cols = ["CLIENTNO", "CLIENT2NO"] if use_clients else ["ANFNR", "ENDNR"]

        # the following code is just a check whether pipe and node geodata fit together
        # in case of deviations, the pipe geodata is adapted on the basis of the node geodata
        pipe_rec = pipe_data.RECNO.values
        for is_x, is_start in product([True, False], [True, False]):
            current_index_range = indices[0] if is_start else indices[1]
            current_pipe_nums = pipe_rec[current_index_range.values]
            adapt_pipe_data_according_to_nodes(
                pipe_data, current_index_range, node_geo, current_pipe_nums, is_x, is_start,
                node_name, node_cols, coord_names
            )


def get_pipe_geo(stored_data, modus):
    """
    Identify the geodata of all pipes. If inflexion points are given, they must be inserted for the
    correct pipes.
    STANET-like:
    pipe -> [(x_start, y_start), (x_end, y_end)]; inflexion_points -> [(x_infl, y_infl), pipe]
    pandapipes-like:
    pipe_geo ->  [(x_start, y_start), (x_infl1, y_infl1), (x_infl2, y_infl2), ... , (x_end, y_end)]

    :param stored_data: dict of STANET tables
    :type stored_data: dict
    :param modus: either "main" or "houses"
    :type modus: str
    :return: pipe_geo
    :rtype: pd.Series
    """
    # set varibles depent on kind of pipe
    pipe_table_name = "pipes" if modus == "main" else "house_pipes"
    if pipe_table_name not in stored_data:
        return

    coord_names = ["XRA", "XRB", "YHA", "YHB"] if modus == "main" \
        else ["XRECHTS", "XRECHTS2", "YHOCH", "YHOCH2"]
    inflexion_points = "inflexion_points" if modus == "main" else "house_inflexion_points"

    pipe_data = stored_data[pipe_table_name]

    adapt_pipe_data(stored_data, pipe_data, coord_names, modus != "main")

    pipe_geo_data = pipe_data.loc[:, coord_names + ["RECNO"]]
    if inflexion_points in stored_data:
        used_cols, sort_cols = ["XRECHTS", "YHOCH", "SNUM", "KNICKNO"], ["SNUM", "KNICKNO"]
        ipt = stored_data["{}".format(inflexion_points)].loc[:, used_cols].sort_values(sort_cols)
        # make lists of all x- and y-values of the inflexion points grouped by the pipe index
        xr = ipt.groupby("SNUM").XRECHTS.apply(list)
        xr.name = "XRECHTS"
        yh = ipt.groupby("SNUM").YHOCH.apply(list)
        yh.name = "YHOCH"
        # merge the x- and y-value lists - result looks like:
        # SNUM (pipe) ; XRECHTS      ; YHOCH
        # 1           ; (12, 23, 18) ; (1, 28, 49)
        xy = pd.merge(xr, yh, left_index=True, right_index=True)
        # make list of x-y-tuples from the merged df
        ipt_geo = xy.apply(lambda ip: [(x, y) for x, y in zip(ip.XRECHTS, ip.YHOCH)], axis=1)
        ipt_geo.name = "IPT"
        # merge the resulting series with the pipe-dataframe and create the final form of geodata
        # tuples as required by pandapipes
        all_geo = pd.merge(pipe_geo_data, ipt_geo, left_on="RECNO", right_index=True, how="left")

        def sum_coords(geo):
            from_coords = (geo[coord_names[0]], geo[coord_names[2]])
            to_coords = (geo[coord_names[1]], geo[coord_names[3]])
            inflex = geo.IPT if not np.any(np.isnan(geo.IPT)) else []
            return [from_coords] + inflex + [to_coords]

        pipe_geo = all_geo.apply(sum_coords, axis=1)
    else:
        pipe_geo = pipe_geo_data.apply(lambda geo: [(geo[coord_names[0]], geo[coord_names[2]]),
                                                    (geo[coord_names[1]], geo[coord_names[3]])],
                                       axis=1)
    pipe_geo.index = pipe_geo_data.RECNO.values
    return pipe_geo


def connection_pipe_section_table(stored_data, pipe_geodata, house_pipe_geodata,
                                  remove_unused_household_connections):
    """
    Returns pipe and house connection lines
    :param stored_data: dict of STANET tables
    :type stored_data: dict
    :param pipe_geodata: geodata of all pipes
    :type pipe_geodata: panda series
    :param house_pipe_geodata: geodata of all house connections
    :type house_pipe_geodata: panda series
    :param remove_unused_household_connections:
    :type remove_unused_household_connections:
    :return: connections
    :rtype:
    """
    logger.info("Calculation of pipe sections due to internal connections.")
    if "pipes" not in stored_data:
        raise UserWarning("There are no pipes in the stored data from STANET. It will not be "
                          "possible to generate pipe sections with connections to households etc.")

    # These are the required entries to create nodes and pipe sections from the connections on
    # STANET pipes. N.B.: A connection is a node on a STANET pipe that does not explicitly split
    # this pipe within STANET, but must be considered in pandapipes, as such components do not
    # exist there. I.e. in pandapipes, the pipes that contain connections will be split at the
    # respective position.
    required_columns = ["XRECHTS", "YHOCH", "GEOH", "SNUM", "RECNO", "STANETID", "PRECH",
                        "CLIENTTYP", "ISACTIVE", "LAYER", "VMA", "VMB"]
    connections = pd.DataFrame(columns=required_columns)
    if "house_connections" in stored_data:
        # first connection type: house connection lines (in STANET their individual calculation can
        # be switched on and off)
        if remove_unused_household_connections:
            hc = stored_data["house_connections"]
            used_connections = np.isin(hc.STANETID, stored_data["house_pipes"].CLIENTID) \
                | np.isin(hc.STANETID, stored_data["house_pipes"].CLIENT2ID)
            logger.info("Removing household connections %s" % hc.index[~used_connections])
        else:
            used_connections = stored_data["house_connections"].index.values
        c1 = stored_data["house_connections"].loc[used_connections, required_columns]
        c1["type"] = "house_connections"
        connections = pd.concat([connections, c1], ignore_index=True)

    if "slider_valves" in stored_data:
        # second connection type: slider valves (will be represented as valves in pandapipes, which
        # means that they need two junctions inserted)
        # --> In case pipe valves will be introduced in pandapipes, this behavior could be changed,
        #     but requires checks (e.g. positioning on pipe, max. 2 valves per pipe)
        c2 = stored_data["slider_valves"].loc[:, [c for c in required_columns if c not in
                                                  ["PRECH", "VMA", "VMB"]]]
        c2["PRECH"] = np.NaN
        c2["VMA"] = np.NaN
        c2["VMB"] = np.NaN
        c2["type"] = "slider_valves"
        connections = pd.concat([connections, c2], ignore_index=True)

    if connections.empty:
        return connections

    def dist_on_line(row):
        return LineString(row.line_geo).project(Point(row.geo), normalized=True)

    if not np.all(np.isin(connections.CLIENTTYP.values, [2, 38])):
        raise UserWarning("Connections cannot be created for clients (pipes) of type %s"
                          % (set(connections.CLIENTTYP.values) - {2, 38}))

    # in the end, add geo information (pipe splitting will be performed based on the geodata)
    connections["geo"] = connections.apply(lambda c: (c.XRECHTS, c.YHOCH), axis=1)
    connections["line_geo"] = pd.Series("", index=connections.index, dtype=object)
    main_connection = connections.CLIENTTYP == 2
    if np.any(main_connection):
        connections.line_geo.loc[main_connection] = \
            pipe_geodata.loc[connections.SNUM.loc[main_connection].values].values
    house_connection = connections.CLIENTTYP == 38
    if np.any(house_connection):
        connections.line_geo.loc[house_connection] = \
            house_pipe_geodata.loc[connections.SNUM.loc[house_connection].values].values
    connections["pos_on_line"] = connections.apply(dist_on_line, axis=1)
    # connections = connections.sort_values(["type", "SNUM"])
    # connections.index = np.arange(len(connections))

    return connections


def create_meter_table(stored_data, connections, index_mapping, **kwargs):
    sinks_defined = False
    if "meters" in stored_data:
        sinks_defined = True
        meter_table = stored_data["meters"]
        if kwargs.get("remove_disconnected_meters", False):
            meter_table = meter_table.loc[meter_table.KNONUM != 0]
        if connections.empty:
            # this might be wrong, as meter nodes could be created later again...
            index_mapping["meters"] = dict(zip(
                meter_table.RECNO.astype(np.int32),
                map(lambda nn: index_mapping["nodes"][nn], meter_table.KNONUM.astype(np.int32))
            ))
    else:
        meter_table = pd.DataFrame(columns=["HAUSINDEX", "RECNO"])
    return meter_table, sinks_defined


def create_house_table(stored_data, sinks_defined):
    if "houses" in stored_data:
        sinks_defined = True
        house_table = stored_data["houses"]
    else:
        house_table = pd.DataFrame(columns=["RECNO"])
    return house_table, sinks_defined
