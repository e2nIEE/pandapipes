# Copyright (c) 2020-2024 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

from numpy import dtype

from pandapipes.component_models.abstract_models.branch_element_models import BranchElementComponent
from pandapipes.component_models import standard_branch_wo_internals_result_lookup
from pandapipes.idx_branch import JAC_DERIV_DP, JAC_DERIV_DP1, JAC_DERIV_DM, \
    BRANCH_TYPE, LOSS_COEFFICIENT as LC, PC as PC_BRANCH
from pandapipes.idx_node import PINIT, NODE_TYPE, PC as PC_NODE
from pandapipes.pf.pipeflow_setup import get_lookup
from pandapipes.pf.result_extraction import extract_branch_results_without_internals
from pandapipes.properties.fluids import get_fluid


class PressureControlComponent(BranchElementComponent):
    """

    """
    @property
    def table_name(self):
        return "press_control"

    def create_pit_node_entries(self, net, node_pit):
        pcs = net[self.table_name]
        controlled = pcs.in_service & pcs.control_active
        juncts = pcs['controlled_junction'].values[controlled]
        press = pcs['controlled_p_bar'].values[controlled]
        junction_idx_lookups = get_lookup(net, "node", "index")[
            self.connected_node_type.table_name]
        index_pc = junction_idx_lookups[juncts]
        node_pit[index_pc, NODE_TYPE] = PC_NODE
        node_pit[index_pc, PINIT] = press

    def create_pit_branch_entries(self, net, branch_pit):
        """
        Function which creates pit branch entries with a specific table.
        :param net: The pandapipes network
        :type net: pandapipesNet
        :param branch_pit:
        :type branch_pit:
        :return: No Output.
        """
        pc_pit = super().create_pit_branch_entries(net, branch_pit)
        pc_pit[net[self.table_name].control_active.values, BRANCH_TYPE] = PC_BRANCH
        pc_pit[:, LC] = net[self.table_name].loss_coefficient.values

    def adaption_before_derivatives_hydraulic(self, net, branch_pit, node_pit, idx_lookups, options):
        pass

    def adaption_after_derivatives_hydraulic(self, net, branch_pit, node_pit, idx_lookups, options):
        # set all PC branches to derivatives to 0
        f, t = idx_lookups[self.table_name]
        press_pit = branch_pit[f:t, :]
        pc_branch = press_pit[:, BRANCH_TYPE] == PC_BRANCH
        press_pit[pc_branch, JAC_DERIV_DP] = 0
        press_pit[pc_branch, JAC_DERIV_DP1] = 0
        press_pit[pc_branch, JAC_DERIV_DM] = 0

    def extract_results(self, net, options, branch_results, mode):
        """
        Function that extracts certain results.

        :param mode:
        :type mode:
        :param branch_results:
        :type branch_results:
        :param net: The pandapipes network
        :type net: pandapipesNet
        :param options:
        :type options:
        :return: No Output.
        """

        required_results_hyd, required_results_ht = standard_branch_wo_internals_result_lookup(net)

        extract_branch_results_without_internals(net, branch_results, required_results_hyd,
                                                 required_results_ht, self.table_name, mode)

        res_table = net["res_" + self.table_name]
        f, t = get_lookup(net, "branch", "from_to")[self.table_name]
        p_to = branch_results["p_to"][f:t]
        p_from = branch_results["p_from"][f:t]
        res_table["deltap_bar"].values[:] = p_to - p_from

    def get_component_input(self):
        """

        Get component input.

        :return:
        :rtype:
        """
        return [("name", dtype(object)),
                ("from_junction", "u4"),
                ("to_junction", "u4"),
                ("controlled_junction", "u4"),
                ("controlled_p_bar", "f8"),
                ("control_active", "bool"),
                ("loss_coefficient", "f8"),
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
                      "vdot_norm_m3_per_s", "normfactor_from", "normfactor_to"]
        else:
            output = ["p_from_bar", "p_to_bar", "t_from_k", "t_to_k", "t_outlet_k",
                      "mdot_from_kg_per_s", "mdot_to_kg_per_s", "vdot_m3_per_s"]
        output += ["deltap_bar"]
        return output, True
