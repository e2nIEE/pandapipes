# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
from numpy import dtype

from pandapipes.component_models.junction_component import Junction
from pandapipes.component_models.abstract_models.circulation_pump import CirculationPump
from pandapipes.idx_node import LOAD
from pandapipes.pf.internals_toolbox import _sum_by_group
from pandapipes.pf.pipeflow_setup import get_lookup
from pandapipes.pf.pipeflow_setup import get_net_option

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


class CirculationPumpMass(CirculationPump):

    @classmethod
    def table_name(cls):
        return "circ_pump_mass"

    @classmethod
    def get_component_input(cls):
        """

        :return:
        :rtype:
        """
        return [("name", dtype(object)),
                ("return_junction", "u4"),
                ("flow_junction", "u4"),
                ("p_flow_bar", "f8"),
                ("t_flow_k", "f8"),
                ("mdot_flow_kg_per_s", "f8"),
                ("in_service", 'bool'),
                ("type", dtype(object))]

    @classmethod
    def get_connected_node_type(cls):
        return Junction

    @classmethod
    def active_identifier(cls):
        return "in_service"

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
        circ_pump, _ = super().create_pit_node_entries(net, node_pit)

        mf = np.nan_to_num(circ_pump.mdot_flow_kg_per_s.values)
        mass_flow_loads = mf * circ_pump.in_service.values
        juncts, loads_sum = _sum_by_group(get_net_option(net, "use_numba"),
                                          circ_pump.return_junction.values, mass_flow_loads)
        junction_idx_lookups = get_lookup(net, "node", "index")[
            cls.get_connected_node_type().table_name()]
        index = junction_idx_lookups[juncts]
        node_pit[index, LOAD] += loads_sum

    @classmethod
    def calculate_temperature_lift(cls, net, pipe_pit, node_pit):
        pass
