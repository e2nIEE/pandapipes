# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

from pandapipes.component_models.abstract_models.branch_wo_internals_models import \
    BranchWOInternalsComponent
from pandapipes.idx_branch import LENGTH, K

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


class BranchWZeroLengthComponent(BranchWOInternalsComponent):
    """

    """

    @classmethod
    def from_to_node_cols(cls):
        return NotImplementedError

    @classmethod
    def active_identifier(cls):
        return NotImplementedError

    @classmethod
    def calculate_pressure_lift(cls, net, pipe_pit, node_pit):
        return NotImplementedError

    @classmethod
    def calculate_temperature_lift(cls, net, pipe_pit, node_pit):
        return NotImplementedError

    @classmethod
    def create_pit_branch_entries(cls, net, branch_wzerolength_pit, node_name):
        """
        Function which creates pit branch entries with a specific table.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param branch_wzerolength_pit:
        :type branch_wzerolength_pit:
        :param node_name:
        :type node_name:
        :return: No Output.
        """
        branch_wizerolength_pit = \
            super().create_pit_branch_entries(net, branch_wzerolength_pit, node_name)
        branch_wizerolength_pit[:, LENGTH] = 0
        branch_wizerolength_pit[:, K] = 1000
        return branch_wizerolength_pit
