# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
from numpy import dtype

from pandapipes.component_models.abstract_models.node_element_models import NodeElementComponent
from pandapipes.pf.pipeflow_setup import get_lookup
from pandapipes.component_models.component_toolbox import set_fixed_node_entries

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


class ExtGrid(NodeElementComponent):
    """

    """

    @classmethod
    def table_name(cls):
        return "ext_grid"

    @classmethod
    def sign(cls):
        return 1.

    @classmethod
    def get_connected_node_type(cls):
        from pandapipes.component_models.junction_component import Junction
        return Junction

    @classmethod
    def active_identifier(cls):
        return "in_service"

    @classmethod
    def node_element_relevant(cls, net):
        return True

    @classmethod
    def create_pit_node_element_entries(cls, net, node_element_pit):
        ext_grids = net[cls.table_name()]
        ext_grids = ext_grids[ext_grids.in_service.values]
        ext_grid_pit = super().create_pit_node_element_entries(net, node_element_pit)
        p_mask = np.where(np.isin(ext_grids.type.values, ["p", "pt"]))
        t_mask = np.where(np.isin(ext_grids.type.values, ["t"]))
        ext_grid_pit[p_mask, net._idx_node_element['NODE_ELEMENT_TYPE']] = net._idx_node_element['P']
        ext_grid_pit[t_mask, net._idx_node_element['NODE_ELEMENT_TYPE']] = net._idx_node_element['T']
        ext_grid_pit[:, net._idx_node_element['MINIT']] = 0.005
        return ext_grid_pit

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
        ext_grids = net[cls.table_name()]
        ext_grids = ext_grids[ext_grids[cls.active_identifier()].values]

        junction = ext_grids[cls.get_node_col()].values
        press = ext_grids.p_bar.values
        set_fixed_node_entries(net, node_pit, junction, ext_grids.type.values, press,
                               ext_grids.t_k.values, cls.get_connected_node_type())

        return ext_grids, press

    @classmethod
    def extract_results(cls, net, options, branch_results, mode):
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

        ext_grids = net[cls.table_name()]

        if len(ext_grids) == 0:
            return

        res_table = net["res_" + cls.table_name()]
        #TODO: Check why pit and active pit lead to different results
        f, t = get_lookup(net, "node_element", "from_to")[cls.table_name()]

        node_element_pit = net["_pit"]["node_element"][f:t]
        # positive results mean that the ext_grid feeds in, negative means that the ext grid
        # extracts (like a load)
        res_table["mdot_kg_per_s"].values[:] = \
            cls.sign() * node_element_pit[:, net['_idx_node_element']['MINIT']]
        return res_table, ext_grids

    @classmethod
    def get_connected_junction(cls, net):
        junction = net[cls.table_name()].junction
        return junction

    @classmethod
    def get_node_col(cls):
        return "junction"

    @classmethod
    def get_component_input(cls):
        """

        :return:
        :rtype:
        """
        return [("name", dtype(object)),
                ("junction", "u4"),
                ("p_bar", "f8"),
                ("t_k", "f8"),
                ("fluid", dtype(object)),
                ("in_service", "bool"),
                ('type', dtype(object))]

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
