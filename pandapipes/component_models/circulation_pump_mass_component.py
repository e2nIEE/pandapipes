# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
from pandapipes.component_models.abstract_models import CirculationPump
from pandapipes.idx_node import PINIT, NODE_TYPE, P, LOAD, EXT_GRID_OCCURENCE, TINIT, NODE_TYPE_T, \
    EXT_GRID_OCCURENCE_T, T
from pandapipes.idx_branch import FROM_NODE, TO_NODE, LOAD_VEC_NODES
from pandapipes.pipeflow_setup import get_lookup, get_fluid
from numpy import dtype

from pandapipes.toolbox import _sum_by_group

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


class CirculationPumpMass(CirculationPump):

    @classmethod
    def table_name(cls):
        return "circ_pump_mass"

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
        circ_pump, press = super().create_pit_node_entries(net, node_pit, node_name)

        mf = np.nan_to_num(circ_pump.mdot_kg_per_s.values)
        mass_flow_loads = mf * circ_pump.in_service.values
        juncts, loads_sum = _sum_by_group(circ_pump.to_junction.values, mass_flow_loads)
        junction_idx_lookups = get_lookup(net, "node", "index")[node_name]
        index = junction_idx_lookups[juncts]
        node_pit[index, LOAD] += loads_sum

    @classmethod
    def get_component_input(cls):
        """

        :return:
        :rtype:
        """
        return [("name", dtype(object)),
                ("from_junction", "u4"),
                ("to_junction", "u4"),
                ("p_bar", "f8"),
                ("t_k", "f8"),
                ("mdot_kg_per_s", "f8"),
                ("in_service", 'bool'),
                ("type", dtype(object))]
