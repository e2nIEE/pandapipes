# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

from numpy import dtype

from pandapipes.component_models.abstract_models.circulation_pump import CirculationPump
from pandapipes.component_models.component_toolbox import set_fixed_node_entries
from pandapipes.component_models.junction_component import Junction

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


class CirculationPumpPressure(CirculationPump):

    @classmethod
    def table_name(cls):
        return "circ_pump_pressure"

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
                ("plift_bar", "f8"),
                ("in_service", 'bool'),
                ("type", dtype(object))]

    @classmethod
    def active_identifier(cls):
        return "in_service"

    @classmethod
    def get_connected_node_type(cls):
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
        circ_pump, press = super().create_pit_node_entries(net, node_pit)

        junction = circ_pump[cls.from_to_node_cols()[0]].values
        p_in = press - circ_pump.plift_bar.values
        set_fixed_node_entries(net, node_pit, junction, circ_pump.type.values, p_in, None,
                               cls.get_connected_node_type(), "p")

    @classmethod
    def calculate_temperature_lift(cls, net, pipe_pit, node_pit):
        pass
