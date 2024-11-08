# Copyright (c) 2020-2024 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

from numpy import pi, int32

from pandapipes.component_models._branch_element_models import BranchElementComponent
from pandapipes.component_models.component_toolbox import set_fixed_node_entries, standard_branch_wo_internals_result_lookup
from pandapipes.idx_branch import D, AREA, BRANCH_TYPE, CIRC, LOAD_VEC_BRANCHES_T, TO_NODE
from pandapipes.idx_node import MDOTSLACKINIT, VAR_MASS_SLACK, JAC_DERIV_MSL
from pandapipes.pf.pipeflow_setup import get_fluid
from pandapipes.pf.result_extraction import extract_branch_results_without_internals


class CirculationPump(BranchElementComponent):
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
            output = ["v_from_m_per_s", "v_to_m_per_s", "v_mean_m_per_s", "p_from_bar", "p_to_bar", "t_from_k",
                      "t_to_k", "t_outlet_k", "mdot_from_kg_per_s", "mdot_to_kg_per_s", "vdot_norm_m3_per_s",
                      "normfactor_from", "normfactor_to"]
        else:
            output = ["v_mean_m_per_s", "p_from_bar", "p_to_bar", "t_from_k", "t_to_k", "t_outlet_k", "mdot_from_kg_per_s",
                      "mdot_to_kg_per_s", "vdot_m3_per_s"]
        output += ['deltat_k', 'qext_w']
        return output, True

    def create_pit_node_entries(self, net, node_pit):
        """
        Function which creates pit node entries.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param node_pit:
        :type node_pit:
        :return: No Output.
        """
        circ_pump_tbl = net[self.table_name][net[self.table_name][self.active_identifier].values]

        junction = circ_pump_tbl[self.from_to_node_cols[1]].values

        # TODO: there should be a warning, if any p_bar value is not given or any of the types does
        #       not contain "p", as this should not be allowed for this component
        types = circ_pump_tbl.type.values
        p_values = circ_pump_tbl.p_flow_bar.values
        t_values = circ_pump_tbl.t_flow_k.values
        index_p = set_fixed_node_entries(
            net, node_pit, junction, types, p_values, self.connected_node_type, 'p')
        set_fixed_node_entries(net, node_pit, junction, types, t_values, self.connected_node_type, 't')
        node_pit[index_p, JAC_DERIV_MSL] = -1.
        return circ_pump_tbl, p_values

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
        circ_pump_pit[:, D] = 0.1
        circ_pump_pit[:, AREA] = circ_pump_pit[:, D] ** 2 * pi / 4
        circ_pump_pit[:, BRANCH_TYPE] = CIRC
        return circ_pump_pit

    def adaption_after_derivatives_hydraulic(self, net, branch_pit, node_pit, idx_lookups, options):
        """
        Function which creates pit branch entries with a specific table.
        :param net: The pandapipes network
        :type net: pandapipesNet
        :param branch_pit:
        :type branch_pit:
        :return: No Output.
        """
        f, t = idx_lookups[self.table_name]
        circ_pump_pit = branch_pit[f:t, :]
        tn = circ_pump_pit[:, TO_NODE].astype(int32)
        mask = node_pit[tn, VAR_MASS_SLACK].astype(bool)
        node_pit[tn[~mask], MDOTSLACKINIT] = 0
        return circ_pump_pit

    def adaption_after_derivatives_thermal(self, net, branch_pit, node_pit, idx_lookups, options):
        """
        Function which creates pit branch entries with a specific table.
        :param net: The pandapipes network
        :type net: pandapipesNet
        :param branch_pit:
        :type branch_pit:
        :return: No Output.
        """
        f, t = idx_lookups[self.table_name]
        circ_pump_pit = branch_pit[f:t, :]
        circ_pump_pit[:, LOAD_VEC_BRANCHES_T] = 0

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

        extract_branch_results_without_internals(net, branch_results, required_results_hyd, required_results_ht,
                                                 self.table_name, mode)
