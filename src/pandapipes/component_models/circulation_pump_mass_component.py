# Copyright (c) 2020-2025 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

from numpy import dtype

from pandapipes.component_models.abstract_models.circulation_pump import CirculationPump
from pandapipes.component_models.junction_component import Junction
from pandapipes.idx_branch import JAC_DERIV_DP, JAC_DERIV_DP1, JAC_DERIV_DM, MDOTINIT, \
    LOAD_VEC_BRANCHES

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
    def create_pit_branch_entries(cls, net, branch_pit):
        circ_pump_pit = super().create_pit_branch_entries(net, branch_pit)
        circ_pump_pit[:, MDOTINIT] = net[cls.table_name()].mdot_flow_kg_per_s.values

    @classmethod
    def adaption_after_derivatives_hydraulic(cls, net, branch_pit, node_pit, idx_lookups, options):
        # set all pressure derivatives to 0 and velocity to 1; load vector must be 0, as no change
        # of velocity is allowed during the pipeflow iteration
        circ_pump_pit = super().adaption_after_derivatives_hydraulic(net, branch_pit, node_pit, idx_lookups, options)
        circ_pump_pit[:, JAC_DERIV_DP] = 0
        circ_pump_pit[:, JAC_DERIV_DP1] = 0
        circ_pump_pit[:, JAC_DERIV_DM] = 1
        circ_pump_pit[:, LOAD_VEC_BRANCHES] = 0

