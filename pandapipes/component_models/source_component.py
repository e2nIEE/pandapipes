# Copyright (c) 2020-2022 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
from numpy import dtype

from pandapipes.component_models.abstract_models import ConstFlow
from pandapipes.internals_toolbox import _sum_by_group
from pandapipes.pipeflow_setup import get_lookup


class Source(ConstFlow):
    """

    """

    @classmethod
    def table_name(cls):
        return "source"

    @classmethod
    def sign(cls):
        return -1

    @classmethod
    def node_element_relevant(cls, net):
        return len(net._fluid) != 1

    @classmethod
    def create_pit_node_element_entries(cls, net, node_element_pit, node_name):
        fluids = net._fluid
        if len(fluids) != 1:
            source_pit = super().create_pit_node_element_entries(net, node_element_pit, node_name)
            sources = net[cls.table_name()]
            helper = sources.in_service.values * sources.scaling.values
            mf = np.nan_to_num(sources.mdot_kg_per_s.values)
            mass_flow_loads = mf * helper
            source_pit[:, net._idx_node_element['MINIT']] = mass_flow_loads

    @classmethod
    def create_pit_node_entries(cls, net, node_pit, node_name):
        super().create_pit_node_entries(net, node_pit, node_name)
        sources = net[cls.table_name()]
        fluids = net._fluid
        if len(fluids) != 1:
            helper = sources.in_service.values * sources.scaling.values * cls.sign()
            mf = np.nan_to_num(sources.mdot_kg_per_s.values)
            mass_flow_loads = mf * helper
            for fluid in fluids:
                juncts, sources_sum = _sum_by_group(sources.junction.values[sources.fluid == fluid],
                                                    mass_flow_loads[sources.fluid == fluid])
                junction_idx_lookups = get_lookup(net, "node", "index")[node_name]
                index = junction_idx_lookups[juncts]
                node_pit[index, net['_idx_node'][fluid + '_LOAD']] -= sources_sum

    @classmethod
    def get_component_input(cls):
        """
        Column names and types of the corresponding DataFrame

        :return:
        :rtype:
        """
        return [("name", dtype(object)),
                ("junction", "u4"),
                ("mdot_kg_per_s", "f8"),
                ("fluid", dtype(object)),
                ("scaling", "f8"),
                ("in_service", "bool"),
                ("type", dtype(object))]
