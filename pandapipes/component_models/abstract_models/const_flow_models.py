# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
from pandapipes.component_models.abstract_models.node_element_models import NodeElementComponent
from pandapipes.idx_node import LOAD, ELEMENT_IDX
from pandapipes.pipeflow_setup import get_lookup
from pandapipes.toolbox import _sum_by_group
from numpy import dtype


class ConstFlow(NodeElementComponent):
    """

    """

    @classmethod
    def sign(cls):
        raise NotImplementedError()

    @classmethod
    def create_pit_node_entries(cls, net, node_pit, node_name):
        """
        Function which creates pit node entries.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param node_pit:
        :type node_pit:
        :return: No Output.
        """
        loads = net[cls.table_name()]
        helper = loads.in_service.values * loads.scaling.values * cls.sign()
        mf = np.nan_to_num(loads.mdot_kg_per_s.values)
        mass_flow_loads = mf * helper
        juncts, loads_sum = _sum_by_group(loads.junction.values, mass_flow_loads)
        junction_idx_lookups = get_lookup(net, "node", "index")[node_name]
        index = junction_idx_lookups[juncts]
        node_pit[index, LOAD] += loads_sum

    @classmethod
    def extract_results(cls, net, options, node_name):
        """
        Function that extracts certain results.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param options:
        :type options:
        :return: No Output.
        """
        res_table = super().extract_results(net, options, node_name)

        loads = net[cls.table_name()]

        is_loads = loads.in_service.values
        fj, tj = get_lookup(net, "node", "from_to")[node_name]
        junct_pit = net["_pit"]["node"][fj:tj, :]
        nodes_connected = get_lookup(net, "node", "active")[fj:tj]
        is_juncts = np.isin(loads.junction.values, junct_pit[nodes_connected, ELEMENT_IDX])

        is_calc = is_loads & is_juncts
        res_table["mdot_kg_per_s"].values[is_calc] = loads.mdot_kg_per_s.values[is_calc] \
                                                     * loads.scaling.values[is_calc]

    @classmethod
    def get_component_input(cls):
        """

        :return:
        :rtype:
        """
        input = [("name", dtype(object)),
                 ("junction", "u4"),
                 ("mdot_kg_per_s", "f8"),
                 ("scaling", "f8"),
                 ("in_service", "bool"),
                 ("type", dtype(object))]
        return input

    @classmethod
    def get_result_table(cls, net):
        """

        :param net: The pandapipes network
        :type net: pandapipesNet
        :return: (columns, all_float) - the column names and whether they are all float type. Only
                if False, returns columns as tuples also specifying the dtypes
        :rtype: (list, bool)
        """
        return ["mdot_kg_per_s"], True
