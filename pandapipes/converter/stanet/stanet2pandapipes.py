# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
import pandas as pd

from pandapipes.converter.stanet.preparing_steps import get_net_params, get_pipe_geo, \
    connection_pipe_section_table, get_stanet_raw_data, create_meter_table, create_house_table
from pandapipes.converter.stanet.table_creation import create_junctions_from_nodes, \
    create_valve_and_pipe, create_pumps, create_junctions_from_connections, \
    create_pipes_from_connections, create_heat_exchangers, create_slider_valves, \
    create_pipes_from_remaining_pipe_table, create_nodes_house_connections, \
    create_sinks_meters, create_sinks_from_nodes, create_control_components, \
    create_sinks_from_customers, create_pipes_house_connections
from pandapipes.create import create_empty_network

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


# TODO: Some things are left to do for this converter:
#         - in case of house connection calculation in STANET, the connections have results VMA and
#           VMB. These results can be used for comparison, but also for sorting of connections on a
#           main pipe --> especially if connection nodes on the main pipe have the same coordinates.
#         - some TODOs in table_creation
#         - maybe it will be necessary to remove deleted data from the STANET tables, otherwise they
#           might be inserted into the pandapipes net erroneously
def stanet_to_pandapipes(stanet_path, name="net", remove_unused_household_connections=True,
                         stanet_like_valves=False, read_options=None, add_layers=True, **kwargs):
    """Converts STANET csv-file to pandapipesNet.

    :param stanet_path: path to csv-file exported from STANET
    :type stanet_path: str
    :param name: name for the created network
    :type name: str, default "net"
    :param remove_unused_household_connections: if True, the intermediate nodes on pipes that are \
            not connected to any households will be skipped.
    :type remove_unused_household_connections: bool, default True
    :param stanet_like_valves: whether pipes with valves should be treated as one component
            (valve_pipe) or split into separate pipes and valves (common pandapipes practice)
    :type stanet_like_valves: bool, default False
    :param read_options: Additional kwargs for the tables to be read with pd.read_csv. If None, no\
            kwargs will be handed over.
    :type read_options: dict, default None
    :param add_layers: If True, adds information on layers of different components if provided by \
            STANET
    :type add_layers: bool, default True
    :return: net
    :rtype: pandapipesNet
    """
    net = create_empty_network(name=name)

    # stored_data contains different dataframes read from the STANET CSV file for different
    # components, such as junctions, pipes etc., but in the raw STANET form
    stored_data = get_stanet_raw_data(stanet_path, read_options, add_layers)

    logger.info("Getting global calculation parameters.")

    # get general parameters of the network (contains e.g. fluid and simulation parameters)
    if stored_data["net_parameters"] is not None:
        net_params = get_net_params(net, stored_data)
    else:
        raise UserWarning("No net parameters identified from STANET. Please check if the table "
                          "'Netzparameterdaten' is contained in the .csv file.")

    # derive the pipe geodata from the two connected nodes and additional inflexion points.
    # add modua to select main pipes mit modus = "main" or house pipes with modus = "house_pipes"
    pipe_geodata = get_pipe_geo(stored_data, modus="main")
    house_pipe_geodata = get_pipe_geo(stored_data, modus="houses")

    # Connections split pipes into sections, thus they need to be considered before pipe creation.
    # The most common type of pipe sectioning is from house connections.
    connections = connection_pipe_section_table(stored_data, pipe_geodata, house_pipe_geodata,
                                                remove_unused_household_connections)

    # create pandapipes tables from the STANET tables
    index_mapping = dict()

    # The junctions are the first component to be created and can be independent of all others.
    # We consider the STANET nodes and additional nodes (connections) on pipes that create sections.
    create_junctions_from_nodes(net, stored_data, net_params, index_mapping, add_layers)
    create_junctions_from_connections(net, connections, net_params, index_mapping, add_layers)
    create_pipes_from_connections(net, stored_data, connections, index_mapping, pipe_geodata,
                                  add_layers)

    # create pipes, while considering already created pipe sections
    create_pipes_from_remaining_pipe_table(net, stored_data, connections, index_mapping,
                                           pipe_geodata, add_layers)

    create_heat_exchangers(net, stored_data, connections, index_mapping, add_layers)

    # valves always have a length in STANET, therefore, they are created as valve with pipe in
    # pandapipes
    create_valve_and_pipe(net, stored_data, index_mapping, net_params, stanet_like_valves, add_layers)

    create_slider_valves(net, stored_data, index_mapping, add_layers)

    if "pumps_water" in stored_data:
        create_pumps(net, stored_data['pumps_water'], index_mapping, add_layers)

    if "pumps_gas" in stored_data:
        create_pumps(net, stored_data['pumps_gas'], index_mapping, add_layers)

    control_flows = create_control_components(net, stored_data, index_mapping, net_params,
                                              add_layers, **kwargs)

    # create the sinks, which in STANET can either be created as meters or customers
    meter_table, sinks_defined = create_meter_table(stored_data, connections, index_mapping,
                                                    **kwargs)

    house_table, sinks_defined = create_house_table(stored_data, sinks_defined)

    create_nodes_house_connections(net, stored_data, connections, meter_table, house_table,
                                   index_mapping, net_params, add_layers)

    create_pipes_house_connections(net, stored_data, connections, index_mapping, house_pipe_geodata,
                                   net_params["household_results_valid"], add_layers)

    create_sinks_meters(net, meter_table, index_mapping, net_params, add_layers)

    sinks_defined = create_sinks_from_customers(net, stored_data, index_mapping, net_params,
                                                sinks_defined, add_layers)

    create_sinks_from_nodes(net, stored_data["nodes"], index_mapping, net_params, sinks_defined,
                            control_flows, add_layers)

    if add_layers:
        if hasattr(stored_data, "layers"):
            net["stanet_layers"] = stored_data["layers"]
        else:
            logger.warning("Layer data could not be found in the CSV file.")

    # add a p_n value to the nodes, which does not exist in STANET
    add_rated_p_values(net, **kwargs)

    # without temperature calculation: set temperature at all nodes to a global network parameter
    if not net_params["calculate_temp"]:
        net.junction.t_fluid_k = net_params["medium_temp_K"]

    if not net_params["calculation_results_valid"]:
        logger.warning("The STANET results written to net are currently not valid!")

    change_dtypes(net)

    logger.info("Conversion finished. Please check the user_pf_options in your net, as they contain"
                " important STANET calculation parameters.")

    return net


def add_rated_p_values(net, **kwargs):
    """

    :param net:
    :type net:
    :return:
    :rtype:
    """
    junctions_with_p_rated = net.junction.loc[pd.notnull(net.junction.pn_bar)]
    if len(junctions_with_p_rated) == 1:
        net.junction.pn_bar = junctions_with_p_rated.pn_bar.iloc[0]
    elif len(set(junctions_with_p_rated.pn_bar.values)) == 1:
        net.junction.pn_bar = junctions_with_p_rated.pn_bar.iloc[0]
    elif "rated_p_values" in kwargs:
        if "overwrite_rated_p" in kwargs and kwargs["overwrite_rated_p"]:
            overwrite_junctions = pd.Series(True, index=net.junction.index)
        else:
            overwrite_junctions = net.junction.pn_bar.isnull()
        for rated_p, p_range in kwargs["rated_p_values"].items():
            net.junction.loc[(net.junction.p_stanet >= p_range[0])
                             & (net.junction.p_stanet < p_range[1])
                             & overwrite_junctions, "pn_bar"] = rated_p
        if np.any(net.junction.pn_bar.isnull()):
            logger.warning("The given rated pressure ranges could not be used to define rated "
                           "pressure for all junctions! The following junctions are not defined: %s"
                           % net.junction.index[net.junction.pn_bar.isnull()])
    else:
        net.junction.loc[pd.isnull(net.junction.pn_bar), 'pn_bar'] = \
            np.mean(junctions_with_p_rated.pn_bar.values)
        raise UserWarning("Adding the rated pressure to the grid nodes with several feed-ins or "
                          "pressure levels is critical and should be re-considered in the future.")


def change_dtypes(net):

    dtypes = {"stanet_nr": np.int32,
              "stanet_id": str,
              "stanet_status": str,
              "v_stanet": np.float64}

    for comp in net.component_list:
        table_name = comp.table_name()
        if table_name not in net:
            continue
        for col, dt in dtypes.items():
            if col in net[table_name]:
                net[table_name][col] = net[table_name][col].astype(dt)
