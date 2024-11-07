# Copyright (c) 2020-2024 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

from numpy import dtype

from pandapipes.component_models.abstract_models.circulation_pump import CirculationPump
from pandapipes.idx_branch import JAC_DERIV_DP, JAC_DERIV_DP1, PL


class CirculationPumpPressure(CirculationPump):
    @property
    def table_name(self):
        return "circ_pump_pressure"

    def get_component_input(self):
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

    def create_pit_branch_entries(self, net, branch_pit):
        """
        Function which creates pit branch entries with a specific table.
        :param net: The pandapipes network
        :type net: pandapipesNet
        :param branch_pit:
        :type branch_pit:
        :return: No Output.
        """
        circ_pump_pit = super().create_pit_branch_entries(net, branch_pit)
        circ_pump_pit[:, PL] = net[self.table_name]['plift_bar'].values
        return circ_pump_pit

    def adaption_after_derivatives_hydraulic(self, net, branch_pit, node_pit, idx_lookups, options):
        # set all pressure derivatives to 0 and velocity to 1; load vector must be 0, as no change
        # of velocity is allowed during the pipeflow iteration
        circ_pump_pit = super().adaption_after_derivatives_hydraulic(net, branch_pit, node_pit, idx_lookups, options)
        circ_pump_pit[:, JAC_DERIV_DP] = 1
        circ_pump_pit[:, JAC_DERIV_DP1] = -1
