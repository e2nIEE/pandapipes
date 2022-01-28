# Copyright (c) 2020-2021 by Fraunhofer Institute for Energy Economics
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
                node_pit[index, net['_idx_node']['LOAD__' + fluid]] -= sources_sum

    @classmethod
    def get_component_input(cls):
        """

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
