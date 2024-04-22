# Copyright (c) 2020-2024 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

from numpy import dtype

from pandapipes.component_models.abstract_models.circulation_pump import CirculationPump
from pandapipes.component_models.junction_component import Junction
from pandapipes.idx_branch import JAC_DERIV_DP, JAC_DERIV_DP1, PL, BRANCH_TYPE
from pandapipes.idx_node import PC, PINIT
from pandapipes.pf.pipeflow_setup import get_lookup

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
        super().create_pit_node_entries(net, node_pit)
        circ_pumps = net[cls.table_name()][net[cls.table_name()][cls.active_identifier()].values]

        junction = circ_pumps[cls.from_to_node_cols()[0]].values

        p_in = circ_pumps.p_flow_bar.values - circ_pumps.plift_bar.values
        junction_idx_lookups = get_lookup(net, "node", "index")[
            cls.get_connected_node_type().table_name()]
        index_pc = junction_idx_lookups[junction]
        node_pit[index_pc, PINIT] = p_in

    @classmethod
    def adaption_after_derivatives_hydraulic(cls, net, branch_pit, node_pit, idx_lookups, options):
        # set all pressure derivatives to 0 and velocity to 1; load vector must be 0, as no change
        # of velocity is allowed during the pipeflow iteration
        f, t = idx_lookups[cls.table_name()]
        circ_pump_pit = branch_pit[f:t, :]
        circ_pump_pit[:, JAC_DERIV_DP] = 1
        circ_pump_pit[:, JAC_DERIV_DP1] = -1
        circ_pump_pit[:, PL] = net[cls.table_name()]['plift_bar'].values
        circ_pump_pit[:, BRANCH_TYPE] = PC
