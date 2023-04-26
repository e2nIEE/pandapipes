# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np

from pandapipes.component_models.abstract_models.branch_wzerolength_models import \
    BranchWZeroLengthComponent
from pandapipes.component_models.component_toolbox import set_fixed_node_entries, \
    get_mass_flow_at_nodes
from pandapipes.idx_branch import D, AREA, ACTIVE
from pandapipes.idx_node import PINIT
from pandapipes.pf.pipeflow_setup import get_lookup

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


class CirculationPump(BranchWZeroLengthComponent):

    @classmethod
    def table_name(cls):
        raise NotImplementedError

    @classmethod
    def get_component_input(cls):
        raise NotImplementedError

    @classmethod
    def get_result_table(cls, net):
        """

        :param net: The pandapipes network
        :type net: pandapipesNet
        :return: (columns, all_float) - the column names and whether they are all float type. Only
                if False, returns columns as tuples also specifying the dtypes
        :rtype: (list, bool)
        """
        return ["mdot_flow_kg_per_s", "deltap_bar"], True

    @classmethod
    def active_identifier(cls):
        raise NotImplementedError

    @classmethod
    def from_to_node_cols(cls):
        return "return_junction", "flow_junction"

    @classmethod
    def get_connected_node_type(cls):
        from pandapipes.component_models.junction_component import Junction
        return Junction

    @classmethod
    def create_pit_node_entries(cls, net, node_pit):
        """
        Function which creates pit node entries.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param node_pit:
        :type node_pit:
        :return: No Output.
        """
        circ_pump_tbl = net[cls.table_name()][net[cls.table_name()][cls.active_identifier()].values]

        junction = net[cls.table_name()][cls.from_to_node_cols()[1]].values

        # TODO: there should be a warning, if any p_bar value is not given or any of the types does
        #       not contain "p", as this should not be allowed for this component
        press = circ_pump_tbl.p_flow_bar.values
        set_fixed_node_entries(net, node_pit, junction, circ_pump_tbl.type.values, press,
                               circ_pump_tbl.t_flow_k.values, cls.get_connected_node_type())
        return circ_pump_tbl, press

    @classmethod
    def create_pit_branch_entries(cls, net, branch_pit):
        """
        Function which creates pit branch entries with a specific table.
        :param net: The pandapipes network
        :type net: pandapipesNet
        :param branch_pit:
        :type branch_pit:
        :return: No Output.
        """
        circ_pump_pit = super().create_pit_branch_entries(net, branch_pit)
        circ_pump_pit[:, D] = 0.1
        circ_pump_pit[:, AREA] = circ_pump_pit[:, D] ** 2 * np.pi / 4
        circ_pump_pit[:, ACTIVE] = False

    @classmethod
    def calculate_temperature_lift(cls, net, pipe_pit, node_pit):
        raise NotImplementedError

    @classmethod
    def extract_results(cls, net, options, branch_results, nodes_connected, branches_connected):
        """
        Function that extracts certain results.

        :param nodes_connected:
        :type nodes_connected:
        :param branches_connected:
        :type branches_connected:
        :param branch_results:
        :type branch_results:
        :param net: The pandapipes network
        :type net: pandapipesNet
        :param options:
        :type options:
        :return: No Output.
        """
        circ_pump_tbl = net[cls.table_name()]

        if len(circ_pump_tbl) == 0:
            return

        res_table = net["res_" + cls.table_name()]

        branch_pit = net['_pit']['branch']
        node_pit = net["_pit"]["node"]

        junction_lookup = get_lookup(net, "node", "index")[
            cls.get_connected_node_type().table_name()]
        fn_col, tn_col = cls.from_to_node_cols()
        # get indices in internal structure for flow_junctions in circ_pump tables which are
        # "active"
        flow_junctions = circ_pump_tbl[tn_col].values
        flow_nodes = junction_lookup[flow_junctions]
        in_service = circ_pump_tbl.in_service.values
        p_grids = np.isin(circ_pump_tbl.type.values, ["p", "pt"]) & in_service
        sum_mass_flows, inverse_nodes, counts = get_mass_flow_at_nodes(net, node_pit, branch_pit,
                                                                       flow_nodes[p_grids], cls)

        # positive results mean that the circ_pump feeds in, negative means that the ext grid
        # extracts (like a load)
        res_table["mdot_flow_kg_per_s"].values[p_grids] = - (sum_mass_flows / counts)[inverse_nodes]

        return_junctions = circ_pump_tbl[fn_col].values
        return_nodes = junction_lookup[return_junctions]

        deltap_bar = node_pit[flow_nodes, PINIT] - node_pit[return_nodes, PINIT]
        res_table["deltap_bar"].values[in_service] = deltap_bar[in_service]
