# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
from numpy import dtype

from pandapipes.component_models.junction_component import Junction
from pandapipes.component_models.abstract_models.branch_wzerolength_models import BranchWZeroLengthComponent
from pandapipes.pf.pipeflow_setup import get_lookup
from pandapipes.properties.fluids import is_fluid_gas, get_fluid
from pandapipes.pf.result_extraction import extract_branch_results_without_internals


class PressureControlComponent(BranchWZeroLengthComponent):
    """

    """

    @classmethod
    def table_name(cls):
        return "press_control"

    @classmethod
    def active_identifier(cls):
        return "in_service"

    @classmethod
    def from_to_node_cols(cls):
        return "from_junction", "to_junction"

    @classmethod
    def get_connected_node_type(cls):
        return Junction

    @classmethod
    def create_pit_node_entries(cls, net, node_pit):
        pcs = net[cls.table_name()]
        controlled = pcs.in_service & pcs.control_active
        juncts = pcs['controlled_junction'].values[controlled]
        press = pcs['controlled_p_bar'].values[controlled]
        junction_idx_lookups = get_lookup(net, "node", "index")[
            cls.get_connected_node_type().table_name()]
        index_pc = junction_idx_lookups[juncts]
        node_pit[index_pc, net['_idx_node']['NODE_TYPE']] = net['_idx_node']['PC']
        node_pit[index_pc, net['_idx_node']['PINIT']] = press

    @classmethod
    def create_pit_branch_entries(cls, net, branch_pit):
        """
        Function which creates pit branch entries with a specific table.
        :param net: The pandapipes network
        :type net: pandapipesNet
        :param branch_pit:
        :type branch_pit:
        :return: No Output.
        """
        pc_pit = super().create_pit_branch_entries(net, branch_pit)
        pc_pit[:, net['_idx_branch']['D']] = 0.1
        pc_pit[:, net['_idx_branch']['AREA']] = pc_pit[:, net['_idx_branch']['D']] ** 2 * np.pi / 4
        pc_pit[net[cls.table_name()].control_active.values, net['_idx_branch']['BRANCH_TYPE']] = net['_idx_node']['PC']
        pc_pit[:, net['_idx_branch']['LOSS_COEFFICIENT']] = net[cls.table_name()].loss_coefficient.values


    @classmethod
    def adaption_after_derivatives_hydraulic(cls, net, branch_pit, node_pit, branch_lookups, node_lookups, options):
        # set all PC branches to derivatives to 0
        f, t = branch_lookups[cls.table_name()]
        press_pit = branch_pit[f:t, :]
        pc_branch = press_pit[:, net['_idx_branch']['BRANCH_TYPE']] == net['_idx_node']['PC']
        press_pit[pc_branch, net['_idx_branch']['JAC_DERIV_DP']] = 0
        press_pit[pc_branch, net['_idx_branch']['JAC_DERIV_DP1']] = 0
        press_pit[pc_branch, net['_idx_branch']['JAC_DERIV_DV']] = 0

    @classmethod
    def calculate_temperature_lift(cls, net, branch_component_pit, node_pit):
        """

        :param net:
        :type net:
        :param branch_component_pit:
        :type branch_component_pit:
        :param node_pit:
        :type node_pit:
        :return:
        :rtype:
        """
        branch_component_pit[:, net['_idx_branch']['TL']] = 0

    @classmethod
    def extract_results(cls, net, options, branch_results, mode):
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
        required_results_hyd = [
            ("p_from_bar", "p_from"), ("p_to_bar", "p_to"), ("mdot_from_kg_per_s", "mf_from"),
            ("mdot_to_kg_per_s", "mf_to"), ("vdot_norm_m3_per_s", "vf")
        ]
        required_results_ht = [("t_from_k", "temp_from"), ("t_to_k", "temp_to")]

        if is_fluid_gas(net):
            required_results_hyd.extend([
                ("v_from_m_per_s", "v_gas_from"), ("v_to_m_per_s", "v_gas_to"),
                ("normfactor_from", "normfactor_from"), ("normfactor_to", "normfactor_to")
            ])
        else:
            required_results_hyd.extend([("v_mean_m_per_s", "v_mps")])

        extract_branch_results_without_internals(net, branch_results, required_results_hyd,
                                                 required_results_ht, cls.table_name(), mode)

        res_table = net["res_" + cls.table_name()]
        f, t = get_lookup(net, "branch", "from_to")[cls.table_name()]
        p_to = branch_results["p_to"][f:t]
        p_from = branch_results["p_from"][f:t]
        res_table["deltap_bar"].values[:] = p_to - p_from

    @classmethod
    def get_component_input(cls):
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

    @classmethod
    def get_result_table(cls, net):
        """

        Gets the result table.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :return: (columns, all_float) - the column names and whether they are all float type. Only
                if False, returns columns as tuples also specifying the dtypes
        :rtype: (list, bool)
        """
        if is_fluid_gas(net):
            output = ["v_from_m_per_s", "v_to_m_per_s", "p_from_bar", "p_to_bar",
                      "t_from_k", "t_to_k", "mdot_from_kg_per_s", "mdot_to_kg_per_s",
                      "vdot_norm_m3_per_s", "normfactor_from", "normfactor_to"]
        else:
            output = ["v_mean_m_per_s", "p_from_bar", "p_to_bar", "t_from_k", "t_to_k",
                      "mdot_from_kg_per_s", "mdot_to_kg_per_s", "vdot_norm_m3_per_s"]
        output += ["deltap_bar"]
        return output, True
