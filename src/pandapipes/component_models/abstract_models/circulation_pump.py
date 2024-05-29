# Copyright (c) 2020-2024 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np

from pandapipes.component_models.abstract_models.branch_wzerolength_models import BranchWZeroLengthComponent
from pandapipes.component_models.component_toolbox import set_fixed_node_entries, standard_branch_wo_internals_result_lookup
from pandapipes.idx_branch import D, AREA, PUMP_TYPE, CIRC, LOAD_VEC_BRANCHES_T, TO_NODE
from pandapipes.idx_node import MDOTSLACKINIT, CIRC_PUMP_OCCURENCE, EXT_GRID_OCCURENCE
from pandapipes.pf.pipeflow_setup import get_fluid
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
    def get_result_table(cls, net):
        """

        :param net: The pandapipes network
        :type net: pandapipesNet
        :return: (columns, all_float) - the column names and whether they are all float type. Only
                if False, returns columns as tuples also specifying the dtypes
        :rtype: (list, bool)
        """
        if get_fluid(net).is_gas:
            output = ["v_from_m_per_s", "v_to_m_per_s", "v_mean_m_per_s", "p_from_bar", "p_to_bar", "t_from_k",
                      "t_to_k", "mdot_from_kg_per_s", "mdot_to_kg_per_s", "vdot_norm_m3_per_s", "reynolds", "lambda",
                      "normfactor_from", "normfactor_to", "qext_w"]
        else:
            output = ["v_mean_m_per_s", "p_from_bar", "p_to_bar", "t_from_k", "t_to_k", "mdot_from_kg_per_s",
                      "mdot_to_kg_per_s", "vdot_norm_m3_per_s", "reynolds", "lambda", "qext_w"]
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
        circ_pump_tbl = net[cls.table_name()][net[cls.table_name()][cls.active_identifier()].values]

        junction = circ_pump_tbl[cls.from_to_node_cols()[1]].values

        # TODO: there should be a warning, if any p_bar value is not given or any of the types does
        #       not contain "p", as this should not be allowed for this component
        press = circ_pump_tbl.p_flow_bar.values
        set_fixed_node_entries(net, node_pit, junction, circ_pump_tbl.type.values, press, circ_pump_tbl.t_flow_k.values,
                               cls.get_connected_node_type(), circ_pump=True)
        return circ_pump_tbl, press

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
        circ_pump_pit[:, PUMP_TYPE] = CIRC
        return circ_pump_pit

    @classmethod
    def adaption_after_derivatives_hydraulic(cls, net, branch_pit, node_pit, idx_lookups, options):
        """
        Function which creates pit branch entries with a specific table.
        :param net: The pandapipes network
        :type net: pandapipesNet
        :param branch_pit:
        :type branch_pit:
        :return: No Output.
        """
        f, t = idx_lookups[cls.table_name()]
        circ_pump_pit = branch_pit[f:t, :]
        tn = circ_pump_pit[:, TO_NODE].astype(np.int32)
        mask = node_pit[tn, CIRC_PUMP_OCCURENCE] == node_pit[tn, EXT_GRID_OCCURENCE]
        node_pit[tn[mask], MDOTSLACKINIT] = 0
        return circ_pump_pit

    @classmethod
    def adaption_after_derivatives_thermal(cls, net, branch_pit, node_pit, idx_lookups, options):
        """
        Function which creates pit branch entries with a specific table.
        :param net: The pandapipes network
        :type net: pandapipesNet
        :param branch_pit:
        :type branch_pit:
        :return: No Output.
        """
        f, t = idx_lookups[cls.table_name()]
        circ_pump_pit = branch_pit[f:t, :]
        circ_pump_pit[:, LOAD_VEC_BRANCHES_T] = 0

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

        required_results_hyd, required_results_ht = standard_branch_wo_internals_result_lookup(net)

        extract_branch_results_without_internals(net, branch_results, required_results_hyd, required_results_ht,
                                                 cls.table_name(), mode)
