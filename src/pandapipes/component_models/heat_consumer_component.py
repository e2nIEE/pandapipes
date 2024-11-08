# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

from numpy import dtype, isnan, zeros, float64, any as any_

from pandapipes.component_models import (get_fluid, BranchElementComponent, get_component_array,
                                         standard_branch_wo_internals_result_lookup)
from pandapipes.idx_branch import (MDOTINIT, QEXT, JAC_DERIV_DP1, JAC_DERIV_DM,
                                   JAC_DERIV_DP, LOAD_VEC_BRANCHES, TOUTINIT, JAC_DERIV_DT,
                                   JAC_DERIV_DTOUT, LOAD_VEC_BRANCHES_T, ACTIVE)
from pandapipes.idx_node import TINIT
from pandapipes.pf.internals_toolbox import get_from_nodes_corrected
from pandapipes.pf.pipeflow_setup import get_lookup
from pandapipes.pf.result_extraction import extract_branch_results_without_internals
from pandapipes.properties.properties_toolbox import get_branch_cp

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)

class HeatConsumer(BranchElementComponent):
    """

    """
    # columns for internal array
    MASS = 0
    QEXT = 1
    DELTAT = 2
    TRETURN = 3
    MODE = 4

    internal_cols = 5

    # heat consumer modes (sum of combinations of given parameters)
    MF_DT = 1
    MF_TR = 2
    QE_MF = 3
    QE_DT = 4
    QE_TR = 5

    @property
    def table_name(self):
        return "heat_consumer"

    def create_pit_branch_entries(self, net, branch_pit):
        """
        Function which creates pit branch entries with a specific table.
        :param net: The pandapipes network
        :type net: pandapipesNet
        :param branch_pit:
        :type branch_pit:
        :return: No Output.
        """
        hc_pit = super().create_pit_branch_entries(net, branch_pit)
        qext = net[self.table_name].qext_w.values
        hc_pit[~isnan(qext), QEXT] = qext[~isnan(qext)]
        mdot = net[self.table_name].controlled_mdot_kg_per_s.values
        hc_pit[~isnan(mdot), MDOTINIT] = mdot[~isnan(mdot)]
        treturn = net[self.table_name].treturn_k.values
        hc_pit[~isnan(treturn), TOUTINIT] = treturn[~isnan(treturn)]
        mask_q0 = qext == 0 & isnan(mdot)
        if any_(mask_q0):
            hc_pit[mask_q0, ACTIVE] = False
            logger.warning(r'qext_w is equals to zero for heat consumers with index %s. '
                           r'Therefore, the defined temperature control cannot be maintained.' \
                    %net[self.table_name].index[mask_q0])
        return hc_pit

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
        consumer_array = zeros(shape=(len(tbl), self.internal_cols), dtype=float64)
        consumer_array[:, self.DELTAT] = tbl.deltat_k.values
        consumer_array[:, self.TRETURN] = tbl.treturn_k.values
        consumer_array[:, self.QEXT] = tbl.qext_w.values
        consumer_array[:, self.MASS] = tbl.controlled_mdot_kg_per_s.values
        mf = tbl.controlled_mdot_kg_per_s.values
        tr = tbl.treturn_k.values
        dt = tbl.deltat_k.values
        qe = tbl.qext_w.values
        mf = ~isnan(mf)
        tr = ~isnan(tr)
        dt = ~isnan(dt)
        qe = ~isnan(qe)
        consumer_array[mf & dt, self.MODE] = self.MF_DT
        consumer_array[mf & tr, self.MODE] = self.MF_TR
        consumer_array[qe & mf, self.MODE] = self.QE_MF
        consumer_array[qe & dt, self.MODE] = self.QE_DT
        consumer_array[qe & tr, self.MODE] = self.QE_TR
        component_pits[self.table_name] = consumer_array

    def adaption_before_derivatives_hydraulic(self, net, branch_pit, node_pit, idx_lookups, options):
        f, t = idx_lookups[self.table_name]
        hc_pit = branch_pit[f:t, :]
        consumer_array = get_component_array(net, self.table_name)

        mask = consumer_array[:, self.MODE] == self.QE_DT
        if any_(mask):
            cp = get_branch_cp(get_fluid(net), node_pit, hc_pit[mask])
            deltat = consumer_array[mask, self.DELTAT]
            mass = hc_pit[mask, QEXT] / (cp * deltat)
            hc_pit[mask, MDOTINIT] = mass

    def adaption_after_derivatives_hydraulic(self, net, branch_pit, node_pit, idx_lookups, options):
        """
        Perform adaptions to the branch pit after the derivatives have been calculated globally.

        :param net: The pandapipes network containing all relevant info
        :type net: pandapipesNet
        :param branch_pit: The branch internal array
        :type branch_pit: np.ndarray
        :param node_pit: The node internal array
        :type node_pit: np.ndarray
        :param idx_lookups: Lookup for the relevant indices in the pit
        :type idx_lookups: dict
        :param options: Options for the pipeflow
        :type options: dict
        :return: No Output.
        :rtype: None
        """
        # set all pressure derivatives to 0 and velocity to 1; load vector must be 0, as no change
        # of velocity is allowed during the pipeflow iteration
        f, t = idx_lookups[self.table_name]
        consumer_array = get_component_array(net, self.table_name)

        hc_pit = branch_pit[f:t, :]
        hc_pit[:, JAC_DERIV_DP] = 0
        hc_pit[:, JAC_DERIV_DP1] = 0
        hc_pit[:, JAC_DERIV_DM] = 1
        hc_pit[:, LOAD_VEC_BRANCHES] = 0

        mask = consumer_array[:, self.MODE] == self.QE_TR
        if any_(mask):
            cp = get_branch_cp(get_fluid(net), node_pit, hc_pit)
            from_nodes = get_from_nodes_corrected(hc_pit)
            t_in = node_pit[from_nodes, TINIT]
            t_out = hc_pit[:, TOUTINIT]

            df_dm = - cp * (t_out - t_in)
            hc_pit[mask, LOAD_VEC_BRANCHES] = - hc_pit[mask, QEXT] + df_dm[mask] * hc_pit[mask, MDOTINIT]
            mask_equal = t_out == t_in
            hc_pit[mask & mask_equal, MDOTINIT] = 0
            hc_pit[mask & ~mask_equal, JAC_DERIV_DM] = df_dm[mask & ~mask_equal]

    def adaption_before_derivatives_thermal(self, net, branch_pit, node_pit, idx_lookups, options):
        f, t = idx_lookups[self.table_name]
        hc_pit = branch_pit[f:t, :]
        consumer_array = get_component_array(net, self.table_name, mode='heat_transfer')
        mask = consumer_array[:, self.MODE] == self.MF_DT
        if any_(mask):
            cp = get_branch_cp(get_fluid(net), node_pit, hc_pit)
            q_ext = cp[mask] * hc_pit[mask, MDOTINIT] * consumer_array[mask, self.DELTAT]
            hc_pit[mask, QEXT] = q_ext

        mask = consumer_array[:, self.MODE] == self.MF_TR
        if any_(mask):
            cp = get_branch_cp(get_fluid(net), node_pit, hc_pit)
            from_nodes = get_from_nodes_corrected(hc_pit[mask])
            t_in = node_pit[from_nodes, TINIT]
            t_out = consumer_array[mask, self.TRETURN]
            q_ext = cp[mask] * hc_pit[mask, MDOTINIT] * (t_in - t_out)
            hc_pit[mask, QEXT] = q_ext

    def adaption_after_derivatives_thermal(self, net, branch_pit, node_pit, idx_lookups, options):
        f, t = idx_lookups[self.table_name]
        hc_pit = branch_pit[f:t, :]
        consumer_array = get_component_array(net, self.table_name, mode='heat_transfer')

        # Any MODE where TRETURN is given
        mask = consumer_array[:, self.MODE] == self.QE_TR
        if any_(mask):
            hc_pit[mask, LOAD_VEC_BRANCHES_T] = 0
            hc_pit[mask, JAC_DERIV_DTOUT] = 1
            hc_pit[mask, JAC_DERIV_DT] = 0

    def get_component_input(self):
        """

        Get component input.

        :return:
        :rtype:
        """
        return [("name", dtype(object)), ("from_junction", "u4"), ("to_junction", "u4"), ("qext_w", "f8"),
                ("controlled_mdot_kg_per_s", "f8"), ("deltat_k", "f8"), ("treturn_k", "f8"),
                ("in_service", "bool"), ("type", dtype(object))]

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
            output = ["p_from_bar", "p_to_bar", "t_from_k",
                      "t_to_k", "t_outlet_k", "mdot_from_kg_per_s", "mdot_to_kg_per_s", "vdot_norm_m3_per_s", "reynolds", "lambda",
                      "normfactor_from", "normfactor_to"]
        else:
            output = ["p_from_bar", "p_to_bar", "t_from_k", "t_to_k", "t_outlet_k", "mdot_from_kg_per_s",
                      "mdot_to_kg_per_s", "vdot_m3_per_s", "reynolds", "lambda"]
        output += ['deltat_k', 'qext_w']
        return output, True

    def extract_results(self, net, options, branch_results, mode):
        """

        :param net:
        :type net:
        :param options:
        :type options:
        :param branch_results:
        :type branch_results:
        :param mode:
        :type mode:
        :return:
        :rtype:
        """
        required_results_hyd, required_results_ht = standard_branch_wo_internals_result_lookup(net)

        extract_branch_results_without_internals(net, branch_results, required_results_hyd, required_results_ht,
                                                 self.table_name, mode)

        node_pit = net['_pit']['node']
        branch_pit = net['_pit']['branch']
        branch_lookups = get_lookup(net, "branch", "from_to")
        f, t = branch_lookups[self.table_name]

        res_table = net["res_" + self.table_name]

        res_table['qext_w'].values[:] = branch_pit[f:t, QEXT]
        from_nodes = get_from_nodes_corrected(branch_pit[f:t])
        t_from = node_pit[from_nodes, TINIT]
        tout = branch_pit[f:t, TOUTINIT]
        res_table['deltat_k'].values[:] = t_from - tout
