# Copyright (c) 2020-2024 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

from numpy import dtype, isin, unique

from pandapipes.component_models._node_element_models import NodeElementComponent
from pandapipes.component_models.component_toolbox import set_fixed_node_entries
from pandapipes.idx_node import MDOTSLACKINIT, VAR_MASS_SLACK, JAC_DERIV_MSL
from pandapipes.utils.internals import get_lookup


class ExtGrid(NodeElementComponent):
    """

    """
    @property
    def table_name(self):
        return "ext_grid"

    @property
    def sign(self):
        return 1.

    @property
    def node_col(self):
        return "junction"

    def create_pit_node_entries(self, net, node_pit):
        """
        Function which creates pit node entries.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param node_pit:
        :type node_pit:
        :return: No Output.
        """
        ext_grids = net[self.table_name]
        ext_grids = ext_grids[ext_grids[self.active_identifier].values]

        junction = ext_grids[self.node_col].values
        types = ext_grids.type.values
        p_values = ext_grids.p_bar.values
        t_values = ext_grids.t_k.values
        index_p = set_fixed_node_entries(
            net, node_pit, junction, types, p_values, self.connected_node_type(), 'p')
        set_fixed_node_entries(net, node_pit, junction, types, t_values, self.connected_node_type(), 't')
        node_pit[index_p, JAC_DERIV_MSL] = -1.
        node_pit[index_p, VAR_MASS_SLACK] = True
        return ext_grids, p_values

    def extract_results(self, net, options, branch_results, mode):
        """
        Function that extracts certain results.

        :param branch_results:
        :type branch_results:
        :param net: The pandapipes network
        :type net: pandapipesNet
        :param options:
        :type options:
        :param mode:
        :type mode:
        :return: No Output.
        """
        ext_grids = net[self.table_name]

        if len(ext_grids) == 0:
            return

        res_table = net["res_" + self.table_name]

        branch_pit = net['_pit']['branch']
        node_pit = net["_pit"]["node"]

        p_grids = isin(ext_grids.type.values, ["p", "pt"]) & ext_grids.in_service.values
        junction = self._get_connected_junction(net).values
        # get indices in internal structure for junctions in ext_grid tables which are "active"
        eg_nodes = get_lookup(net, "node", "index")[self.connected_node_type().table_name][
            junction[p_grids]]
        node_uni, inverse_nodes, counts = unique(eg_nodes, return_counts=True, return_inverse=True)
        sum_mass_flows = node_pit[node_uni, MDOTSLACKINIT]

        # positive results mean that the ext_grid feeds in, negative means that the ext grid
        # extracts (like a load)
        res_table["mdot_kg_per_s"].values[p_grids] = \
            self.sign * (sum_mass_flows / counts)[inverse_nodes]
        return res_table, ext_grids, node_pit, branch_pit

    def _get_connected_junction(self, net):
        junction = net[self.table_name].junction
        return junction

    def get_component_input(self):
        """

        :return:
        :rtype:
        """
        return [("name", dtype(object)),
                ("junction", "u4"),
                ("p_bar", "f8"),
                ("t_k", "f8"),
                ("in_service", "bool"),
                ('type', dtype(object))]
