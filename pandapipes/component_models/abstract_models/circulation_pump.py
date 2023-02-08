# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np

from pandapipes.component_models.abstract_models.branch_wzerolength_models import \
    BranchWZeroLengthComponent
from pandapipes.component_models.component_toolbox import set_fixed_node_entries
from pandapipes.idx_branch import D, AREA, VINIT, CP, FROM_NODE_T, TO_NODE_T, \
    LOAD_VEC_BRANCHES_T, RHO
from pandapipes.idx_node import TINIT
from pandapipes.pf.pipeflow_setup import get_fluid
from pandapipes.pf.pipeflow_setup import get_lookup
from pandapipes.pf.result_extraction import extract_branch_results_without_internals

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


class CirculationPump(BranchWZeroLengthComponent):

    @classmethod
    def table_name(cls):
        raise NotImplementedError

    @classmethod
    def get_component_input(cls):
        raise NotImplementedError

    @classmethod
    # def get_result_table(cls, net):
    #     """
    #
    #     :param net: The pandapipes network
    #     :type net: pandapipesNet
    #     :return: (columns, all_float) - the column names and whether they are all float type. Only
    #             if False, returns columns as tuples also specifying the dtypes
    #     :rtype: (list, bool)
    #     """
    #     return ["mdot_flow_kg_per_s", "deltap_bar"], True

    def get_result_table(cls, net):
        """

        :param net: The pandapipes network
        :type net: pandapipesNet
        :return: (columns, all_float) - the column names and whether they are all float type. Only
                if False, returns columns as tuples also specifying the dtypes
        :rtype: (list, bool)
        """
        if get_fluid(net).is_gas:
            output = ["v_from_m_per_s", "v_to_m_per_s", "v_mean_m_per_s", "p_from_bar", "p_to_bar",
                      "t_from_k", "t_to_k", "mdot_from_kg_per_s", "mdot_to_kg_per_s",
                      "vdot_norm_m3_per_s", "reynolds", "lambda", "normfactor_from",
                      "normfactor_to", "qext_w"]
        else:
            output = ["v_mean_m_per_s", "p_from_bar", "p_to_bar", "t_from_k", "t_to_k",
                      "mdot_from_kg_per_s", "mdot_to_kg_per_s", "vdot_norm_m3_per_s", "reynolds",
                      "lambda", "qext_w"]
        return output, True

    @classmethod
    def active_identifier(cls):
        raise NotImplementedError

    @classmethod
    def from_to_node_cols(cls):
        return "return_junction", "flow_junction"

    @classmethod
    def get_connected_node_type(cls):
        from pandapipes.component_models.junction_component import Junction
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
        circ_pumps = net[cls.table_name()][net[cls.table_name()][cls.active_identifier()].values]
        junction = net[cls.table_name()][cls.from_to_node_cols()[1]].values

        # TODO: there should be a warning, if any p_bar value is not given or any of the types does
        #       not contain "p", as this should not be allowed for this component
        press = circ_pumps.p_flow_bar.values
        set_fixed_node_entries(net, node_pit, junction, circ_pumps.type.values, press,
                               circ_pumps.t_flow_k.values, cls.get_connected_node_type())

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
        circ_pump_pit = super().create_pit_branch_entries(net, branch_pit)
        circ_pump_pit[:, D] = 0.1
        circ_pump_pit[:, AREA] = circ_pump_pit[:, D] ** 2 * np.pi / 4
        return circ_pump_pit

    @classmethod
    def calculate_derivatives_thermal(cls, net, branch_pit, node_pit, idx_lookups, options):
        super().calculate_derivatives_thermal(net, branch_pit, node_pit, idx_lookups, options)
        f, t = idx_lookups[cls.table_name()]
        circ_pump_pit = branch_pit[f:t, :]
        circ_pump_pit[:, LOAD_VEC_BRANCHES_T] = 0

    @classmethod
    def calculate_temperature_lift(cls, net, pipe_pit, node_pit):
        raise NotImplementedError

    @classmethod
    def extract_results(cls, net, options, branch_results, nodes_connected, branches_connected):
        required_results = [
            ("p_from_bar", "p_from"), ("p_to_bar", "p_to"), ("t_from_k", "temp_from"),
            ("t_to_k", "temp_to"), ("mdot_to_kg_per_s", "mf_to"), ("mdot_from_kg_per_s", "mf_from"),
            ("vdot_norm_m3_per_s", "vf"), ("lambda", "lambda"), ("reynolds", "reynolds")
        ]

        if get_fluid(net).is_gas:
            required_results.extend([
                ("v_from_m_per_s", "v_gas_from"), ("v_to_m_per_s", "v_gas_to"),
                ("v_mean_m_per_s", "v_gas_mean"), ("normfactor_from", "normfactor_from"),
                ("normfactor_to", "normfactor_to")
            ])
        else:
            required_results.extend([("v_mean_m_per_s", "v_mps")])

        extract_branch_results_without_internals(net, branch_results, required_results,
                                                 cls.table_name(), branches_connected)
        res_table = net["res_" + cls.table_name()]
        branch_pit = net['_pit']['branch']
        node_pit = net['_pit']['node']
        f, t = get_lookup(net, "branch", "from_to_active")[cls.table_name()]
        circ_pump_pit = branch_pit[f:t]
        from_nodes = circ_pump_pit[:, FROM_NODE_T].astype(np.int32)
        to_nodes = circ_pump_pit[:, TO_NODE_T].astype(np.int32)
        t_from = node_pit[from_nodes, TINIT]
        t_to = node_pit[to_nodes, TINIT]
        res_table['qext_w'].values[:] = circ_pump_pit[:, CP] * circ_pump_pit[:, VINIT] * circ_pump_pit[:, AREA] * \
                                        circ_pump_pit[:, RHO] * (t_to - t_from) * 1000
