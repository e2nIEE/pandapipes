# Copyright (c) 2020-2024 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

from numpy import dtype, bool_, float64, zeros

from pandapipes.component_models.abstract_models import BranchElementComponent
from pandapipes.properties import get_fluid
from pandapipes.component_models.component_toolbox import \
    standard_branch_wo_internals_result_lookup, get_component_array
from pandapipes.idx_branch import JAC_DERIV_DP, JAC_DERIV_DP1, JAC_DERIV_DM, MDOTINIT, LOAD_VEC_BRANCHES
from pandapipes.pf.result_extraction import extract_branch_results_without_internals


class FlowControlComponent(BranchElementComponent):
    """

    """
    CONTROL_ACTIVE = 0

    internal_cols = 1

    @property
    def table_name(self):
        return "flow_control"

    def create_pit_branch_entries(self, net, branch_pit):
        """
        Function which creates pit branch entries with a specific table.
        :param net: The pandapipes network
        :type net: pandapipesNet
        :param branch_pit:
        :type branch_pit:
        :return: No Output.
        """
        fc_branch_pit = super().create_pit_branch_entries(net, branch_pit)
        fc_branch_pit[:, MDOTINIT] = net[self.table_name].controlled_mdot_kg_per_s.values

    def create_component_array(self, net, component_pits):
        """
        Function which creates an internal array of the component in analogy to the pit, but with
        component specific entries, that are not needed in the pit.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param component_pits: dictionary of component specific arrays
        :type component_pits: dict
        :return:
        :rtype:
        """
        tbl = net[self.table_name]
        fc_pit = zeros(shape=(len(tbl), self.internal_cols), dtype=float64)
        fc_pit[:, self.CONTROL_ACTIVE] = tbl.control_active.values
        component_pits[self.table_name] = fc_pit

    def adaption_after_derivatives_hydraulic(self, net, branch_pit, node_pit, idx_lookups, options):
        # set all pressure derivatives to 0 and velocity to 1; load vector must be 0, as no change
        # of velocity is allowed during the pipeflow iteration
        f, t = idx_lookups[self.table_name]
        fc_branch_pit = branch_pit[f:t, :]
        fc_array = get_component_array(net, self.table_name)
        active = fc_array[:, self.CONTROL_ACTIVE].astype(bool_)
        fc_branch_pit[active, JAC_DERIV_DP] = 0
        fc_branch_pit[active, JAC_DERIV_DP1] = 0
        fc_branch_pit[active, JAC_DERIV_DM] = 1
        fc_branch_pit[active, LOAD_VEC_BRANCHES] = 0

    def extract_results(self, net, options, branch_results, mode):
        required_results_hyd, required_results_ht = standard_branch_wo_internals_result_lookup(net)

        extract_branch_results_without_internals(net, branch_results, required_results_hyd,
                                                 required_results_ht, self.table_name, mode)

    def get_component_input(self):
        """

        Get component input.

        :return:
        :rtype:
        """
        return [("name", dtype(object)),
                ("from_junction", "u4"),
                ("to_junction", "u4"),
                ("controlled_mdot_kg_per_s", "f8"),
                ("control_active", "bool"),
                ("in_service", 'bool'),
                ("type", dtype(object))]

    def get_result_table(self, net):
        """

        Gets the result table.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :return: (columns, all_float) - the column names and whether they are all float type. Only
                if False, returns columns as tuples also specifying the dtypes
        :rtype: (list, bool)
        """
        if get_fluid(net).is_gas:
            output = ["p_from_bar", "p_to_bar",
                      "t_from_k", "t_to_k", "t_outlet_k", "mdot_from_kg_per_s", "mdot_to_kg_per_s",
                      "vdot_norm_m3_per_s", "reynolds", "lambda", "normfactor_from",
                      "normfactor_to"]
        else:
            output = ["p_from_bar", "p_to_bar", "t_from_k", "t_to_k", "t_outlet_k",
                      "mdot_from_kg_per_s", "mdot_to_kg_per_s", "vdot_m3_per_s", "reynolds",
                      "lambda"]
        return output, True
