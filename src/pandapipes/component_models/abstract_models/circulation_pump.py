# Copyright (c) 2020-2025 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np

from pandapipes.component_models.abstract_models.branch_wzerolength_models import BranchWZeroLengthComponent
from pandapipes.component_models.component_toolbox import set_fixed_node_entries, standard_branch_wo_internals_result_lookup
from pandapipes.idx_branch import D, AREA, LOAD_VEC_BRANCHES_T, TO_NODE, TOUTINIT, JAC_DERIV_DT, JAC_DERIV_DTOUT, MDOTINIT
from pandapipes.idx_node import MDOTSLACKINIT, VAR_MASS_SLACK, JAC_DERIV_MSL, NODE_TYPE_T, GE, TINIT
from pandapipes.pf.pipeflow_setup import get_fluid, get_lookup
from pandapipes.pf.internals_toolbox import get_from_nodes_corrected
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

        Gets the result table.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :return: (columns, all_float) - the column names and whether they are all float type. Only
                if False, returns columns as tuples also specifying the dtypes
        :rtype: (list, bool)
        """
        if get_fluid(net).is_gas:
            output = ["p_from_bar", "p_to_bar", "t_from_k",
                      "t_to_k", "t_outlet_k", "mdot_from_kg_per_s", "mdot_to_kg_per_s", "vdot_norm_m3_per_s",
                      "normfactor_from", "normfactor_to"]
        else:
            output = ["p_from_bar", "p_to_bar", "t_from_k", "t_to_k", "t_outlet_k", "mdot_from_kg_per_s",
                      "mdot_to_kg_per_s", "vdot_m3_per_s"]
        output += ['deltat_k', 'qext_w']
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
        types = circ_pump_tbl.type.values
        p_values = circ_pump_tbl.p_flow_bar.values
        index_p = set_fixed_node_entries(
            net, node_pit, junction, types, p_values, cls.get_connected_node_type(), 'p')
        node_pit[index_p, JAC_DERIV_MSL] = -1.
        node_pit[index_p, NODE_TYPE_T] = GE
        return circ_pump_tbl, p_values

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
        circ_pump_tbl = net[cls.table_name()][net[cls.table_name()][cls.active_identifier()].values]
        circ_pump_pit = super().create_pit_branch_entries(net, branch_pit)
        circ_pump_pit[:, D] = 0.1
        circ_pump_pit[:, AREA] = circ_pump_pit[:, D] ** 2 * np.pi / 4
        circ_pump_pit[:, TOUTINIT] = circ_pump_tbl.t_flow_k.values
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
        mask = node_pit[tn, VAR_MASS_SLACK].astype(bool)
        node_pit[tn[~mask], MDOTSLACKINIT] = 0
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
        circ_pump_pit[:, JAC_DERIV_DTOUT] = 1
        circ_pump_pit[:, JAC_DERIV_DT] = 0


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
        node_pit = net['_pit']['node']
        branch_pit = net['_pit']['branch']
        branch_lookups = get_lookup(net, "branch", "from_to")
        f, t = branch_lookups[cls.table_name()]

        mask = (branch_pit[f:t, MDOTINIT] < 0) & ~np.isclose(branch_pit[f:t, MDOTINIT], 0)
        if np.any(mask):
            raise UserWarning(r'Your grid is badly modelled and would lead to a direction change in circulation pump %s'
                              % str(net[cls.table_name()].index[mask].tolist()))

        required_results_hyd, required_results_ht = standard_branch_wo_internals_result_lookup(net)

        extract_branch_results_without_internals(net, branch_results, required_results_hyd, required_results_ht,
                                                 cls.table_name(), mode)

        res_table = net["res_" + cls.table_name()]

        from_nodes = get_from_nodes_corrected(branch_pit[f:t])
        t_from = node_pit[from_nodes, TINIT]
        tout = branch_pit[f:t, TOUTINIT]
        res_table['deltat_k'].values[:] = t_from - tout

        fluid = get_fluid(net)

        cp_i = fluid.get_heat_capacity(t_from)
        cp_i1 = fluid.get_heat_capacity(tout)

        mass = branch_pit[f:t, MDOTINIT]
        res_table['qext_w'].values[:] = mass * (cp_i1 * tout - cp_i * t_from)