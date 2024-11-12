# Copyright (c) 2020-2024 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

from numpy import dtype, nan_to_num, isin

from pandapipes.component_models._node_element_models import NodeElementComponent
from pandapipes.component_models.component_registry import ComponentRegistry
from pandapipes.idx_node import LOAD, ELEMENT_IDX
from pandapipes.utils.internals import get_net_option, get_lookup, _sum_by_group


class ConstFlow(NodeElementComponent):
    def create_pit_node_entries(self, net, node_pit):
        """
        Function which creates pit node entries.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param node_pit:
        :type node_pit:
        :return: No Output.
        """
        loads = net[self.table_name]
        helper = loads.in_service.values * loads.scaling.values * self.sign
        mf = nan_to_num(loads.mdot_kg_per_s.values)
        mass_flow_loads = mf * helper
        juncts, loads_sum = _sum_by_group(get_net_option(net, "use_numba"), loads.junction.values,
                                          mass_flow_loads)
        junction_idx_lookups = get_lookup(net, "node", "index")[
            ComponentRegistry.get(self.connected_node_type).table_name]
        index = junction_idx_lookups[juncts]
        node_pit[index, LOAD] += loads_sum

    def extract_results(self, net, options, branch_results, mode):
        """
        Function that extracts certain results.

        :param mode:
        :type mode:
        :param branch_results:
        :type branch_results:
        :param net: The pandapipes network
        :type net: pandapipesNet
        :param options:
        :type options:
        :return: No Output.
        """
        res_table = net["res_" + self.table_name]

        loads = net[self.table_name]

        is_loads = loads.in_service.values
        fj, tj = get_lookup(net, "node", "from_to")[ComponentRegistry.get(self.connected_node_type).table_name]
        junct_pit = net["_pit"]["node"][fj:tj, :]
        nodes_connected_hyd = get_lookup(net, "node", "active_hydraulics")[fj:tj]
        is_juncts = isin(loads.junction.values, junct_pit[nodes_connected_hyd, ELEMENT_IDX])

        is_calc = is_loads & is_juncts
        res_table["mdot_kg_per_s"].values[is_calc] = loads.mdot_kg_per_s.values[is_calc] \
            * loads.scaling.values[is_calc]

    def get_component_input(self):
        """

        :return:
        :rtype:
        """
        return [("name", dtype(object)),
                ("junction", "u4"),
                ("mdot_kg_per_s", "f8"),
                ("scaling", "f8"),
                ("in_service", "bool"),
                ("type", dtype(object))]

