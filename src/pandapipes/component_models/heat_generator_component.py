# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
from numpy import dtype

from pandapipes.component_models import (get_fluid, BranchWOInternalsComponent, get_component_array,
                                         standard_branch_wo_internals_result_lookup, set_fixed_node_entries)
from pandapipes.component_models.junction_component import Junction
from pandapipes.idx_branch import (D, AREA, MDOTINIT, QEXT, JAC_DERIV_DP1, JAC_DERIV_DM,
                                   JAC_DERIV_DP, LOAD_VEC_BRANCHES, TO_NODE, TOUTINIT, JAC_DERIV_DT,
                                   JAC_DERIV_DTOUT, LOAD_VEC_BRANCHES_T, FLOW_RETURN_CONNECT, PL)
from pandapipes.idx_node import MDOTSLACKINIT, VAR_MASS_SLACK, JAC_DERIV_MSL, NODE_TYPE_T, GE, TINIT
from pandapipes.pf.internals_toolbox import get_from_nodes_corrected
from pandapipes.pf.pipeflow_setup import get_lookup
from pandapipes.pf.result_extraction import extract_branch_results_without_internals
from pandapipes.properties.properties_toolbox import get_branch_cp

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)

class HeatGenerator(BranchWOInternalsComponent):
    """

    """
    # columns for internal array
    MASS = 0
    QEXT = 1
    DELTAT = 2
    TFLOW = 3
    MODE = 4

    internal_cols = 5

    # heat generator modes (sum of combinations of given parameters)
    MF_DT = 1
    MF_TR = 2
    QE_MF = 3
    QE_DT = 4
    QE_TR = 5
    PR_PL = 6

    @classmethod
    def table_name(cls):
        return "heat_generator"

    @classmethod
    def get_connected_node_type(cls):
        return Junction

    @classmethod
    def from_to_node_cols(cls):
        return "return_junction", "flow_junction"

    @classmethod
    def active_identifier(cls):
        return "in_service"
    
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

        hg_tbl_all = net[cls.table_name()][net[cls.table_name()][cls.active_identifier()].values]
        mask_pr_pl = (~hg_tbl_all.preturn_bar.isna()) & (~hg_tbl_all.plift_bar.isna())
        hg_tbl = hg_tbl_all[mask_pr_pl]

        if len(hg_tbl) == 0:
            return hg_tbl, np.array([])

        junction = hg_tbl[cls.from_to_node_cols()[1]].values

        types = np.full(len(hg_tbl), "pt", dtype=object)
        p_values = hg_tbl.preturn_bar.values + hg_tbl.plift_bar.values
        index_p = set_fixed_node_entries(
            net, node_pit, junction, types, p_values, cls.get_connected_node_type(), 'p')
        node_pit[index_p, JAC_DERIV_MSL] = -1.
        node_pit[index_p, NODE_TYPE_T] = GE

        return hg_tbl, p_values
    
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

        hg_tbl = net[cls.table_name()][net[cls.table_name()][cls.active_identifier()].values]
        hg_pit = super().create_pit_branch_entries(net, branch_pit)

        qext = hg_tbl.qext_w.values
        hg_pit[~np.isnan(qext), QEXT] = -qext[~np.isnan(qext)]

        mdot = hg_tbl.controlled_mdot_kg_per_s.values
        hg_pit[~np.isnan(mdot), MDOTINIT] = mdot[~np.isnan(mdot)]

        tflow = hg_tbl.tflow_k.values
        hg_pit[~np.isnan(tflow), TOUTINIT] = tflow[~np.isnan(tflow)]
        
        pl = hg_tbl.plift_bar.values
        hg_pit[~np.isnan(pl), PL] = pl[~np.isnan(pl)]
        hg_pit[~np.isnan(pl), D] = 0.1
        hg_pit[~np.isnan(pl), AREA] = hg_pit[~np.isnan(pl), D] ** 2 * np.pi / 4 

        pr = hg_tbl.preturn_bar.values
        mask_pr_pl = (~np.isnan(pr)) & (~np.isnan(pl))
        hg_pit[~mask_pr_pl, FLOW_RETURN_CONNECT] = True
     
        return hg_pit

    @classmethod
    def create_component_array(cls, net, component_pits):
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
        tbl = net[cls.table_name()]
        hg_array = np.zeros(shape=(len(tbl), cls.internal_cols), dtype=np.float64)
        hg_array[:, cls.DELTAT] = tbl.deltat_k.values
        hg_array[:, cls.TFLOW] = tbl.tflow_k.values #TRETURN?
        hg_array[:, cls.QEXT] = tbl.qext_w.values
        hg_array[:, cls.MASS] = tbl.controlled_mdot_kg_per_s.values
        mf = tbl.controlled_mdot_kg_per_s.values
        tf = tbl.tflow_k.values
        dt = tbl.deltat_k.values
        qe = tbl.qext_w.values
        pr = tbl.preturn_bar.values
        pl = tbl.plift_bar.values
        mf = ~np.isnan(mf)
        tf = ~np.isnan(tf)
        dt = ~np.isnan(dt)
        qe = ~np.isnan(qe)
        pr = ~np.isnan(pr)
        pl = ~np.isnan(pl)
        hg_array[pr & pl & tf, cls.MODE] = cls.PR_PL
        hg_array[mf & dt, cls.MODE] = cls.MF_DT
        hg_array[mf & tf, cls.MODE] = cls.MF_TR
        hg_array[qe & mf, cls.MODE] = cls.QE_MF
        hg_array[qe & dt, cls.MODE] = cls.QE_DT
        hg_array[qe & tf, cls.MODE] = cls.QE_TR
        component_pits[cls.table_name()] = hg_array

    @classmethod
    def adaption_before_derivatives_hydraulic(cls, net, branch_pit, node_pit, branch_pit_old, node_pit_old,idx_lookups, options):
        """
        Perform adaptions to the branch pit before the derivatives have been calculated globally.

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

        f, t = idx_lookups[cls.table_name()]
        hc_pit = branch_pit[f:t, :]
        hg_array = get_component_array(net, cls.table_name())

        mask = hg_array[:, cls.MODE] == cls.QE_DT
        if np.any(mask):
            cp = get_branch_cp(get_fluid(net), node_pit, hc_pit[mask])
            deltat = hg_array[mask, cls.DELTAT]
            mass = hc_pit[mask, QEXT] / (cp * deltat)
            hc_pit[mask, MDOTINIT] = mass

    @classmethod
    def adaption_after_derivatives_hydraulic(cls, net, branch_pit, node_pit, branch_pit_old, node_pit_old, idx_lookups, options):
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

        f, t = idx_lookups[cls.table_name()]
        hg_array = get_component_array(net, cls.table_name())

        hg_pit = branch_pit[f:t, :]

        mask = hg_array[:, cls.MODE] == cls.QE_TR
        if np.any(mask):
            cp = get_branch_cp(get_fluid(net), node_pit, hg_pit)
            cp_masked = cp[mask]
            from_nodes = get_from_nodes_corrected(hg_pit)
            from_nodes_masked = from_nodes[mask]
            hg_pit[mask, JAC_DERIV_DP] = 0
            hg_pit[mask, JAC_DERIV_DP1] = 0
            t_in = node_pit[from_nodes_masked, TINIT]
            t_out = hg_pit[mask, TOUTINIT]
            qext = hg_pit[mask, QEXT]
            df_dm = - cp_masked * (t_out - t_in)

            mask_equal = np.where(qext < 0, t_in >= t_out, t_out >= t_in)  
            mask_zero_masked = qext == 0                                   
            mask_ign_masked = mask_equal | mask_zero_masked                

            mask_ign = np.zeros_like(mask, dtype=bool)
            mask_ign[mask] = mask_ign_masked

            hg_pit[mask & mask_ign, MDOTINIT] = 0

            mask_valid_masked = ~mask_ign_masked
            hg_pit[mask & ~mask_ign, JAC_DERIV_DM] = df_dm[mask_valid_masked]

            mdot_masked = hg_pit[mask, MDOTINIT]
            hg_pit[mask, LOAD_VEC_BRANCHES] = -qext + df_dm * mdot_masked

        mask = hg_array[:, cls.MODE] == cls.PR_PL
        if np.any(mask):
            tn = hg_pit[mask, TO_NODE].astype(np.int32)
            slack_mask = node_pit[tn, VAR_MASS_SLACK].astype(bool)
            node_pit[tn[~slack_mask], MDOTSLACKINIT] = 0
            hg_pit[mask, JAC_DERIV_DP] = 1
            hg_pit[mask, JAC_DERIV_DP1] = -1

    @classmethod
    def adaption_before_derivatives_thermal(cls, net, branch_pit, node_pit, branch_pit_old, node_pit_old, idx_lookups, options):
        """
        Perform adaptions to the branch pit before the derivatives have been calculated globally.

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
        f, t = idx_lookups[cls.table_name()]
        hg_pit = branch_pit[f:t, :]
        hg_array = get_component_array(net, cls.table_name(), mode='heat_transfer')
        mask = hg_array[:, cls.MODE] == cls.MF_DT
        if np.any(mask):
            cp = get_branch_cp(get_fluid(net), node_pit, hg_pit)
            q_ext = cp[mask] * hg_pit[mask, MDOTINIT] * hg_array[mask, cls.DELTAT]
            hg_pit[mask, QEXT] = q_ext

        mask = hg_array[:, cls.MODE] == cls.MF_TR
        if np.any(mask):
            cp = get_branch_cp(get_fluid(net), node_pit, hg_pit)
            from_nodes = get_from_nodes_corrected(hg_pit[mask])
            t_in = node_pit[from_nodes, TINIT]
            t_out = hg_array[mask, cls.TFLOW]
            q_ext = cp[mask] * hg_pit[mask, MDOTINIT] * (t_in - t_out)
            hg_pit[mask, QEXT] = q_ext

    @classmethod
    def adaption_after_derivatives_thermal(cls, net, branch_pit, node_pit, branch_pit_old, node_pit_old, idx_lookups, options):
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
        f, t = idx_lookups[cls.table_name()]
        hg_pit = branch_pit[f:t, :]
        hg_array = get_component_array(net, cls.table_name(), mode='heat_transfer')

        mask= hg_array[:, cls.MODE] == cls.QE_TR
        if np.any(mask):
            mask_ign = hg_pit[:, QEXT] == 0
            mask = mask & ~mask_ign
            hg_pit[mask, LOAD_VEC_BRANCHES_T] = 0
            hg_pit[mask, JAC_DERIV_DTOUT] = 1
            hg_pit[mask, JAC_DERIV_DT] = 0

        mask= hg_array[:, cls.MODE] == cls.PR_PL
        if np.any(mask):
            hg_pit[mask, LOAD_VEC_BRANCHES_T] = 0
            hg_pit[mask, JAC_DERIV_DTOUT] = 1
            hg_pit[mask, JAC_DERIV_DT] = 0

    @classmethod
    def get_component_input(cls):
        """

        Get component input.

        :return:
        :rtype:
        """

        return [("name", dtype(object)), ("return_junction", "u4"), ("flow_junction", "u4"), ("qext_w", "f8"),
                ("tflow_k", "f8"), ("controlled_mdot_kg_per_s", "f8"), ("deltat_k", "f8"), ("preturn_bar", "f8"),
                ("plift_bar", "f8"), ("in_service", "bool"), ("type", dtype(object))]
    
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
    def extract_results(cls, net, options, branch_results, mode):
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
            cls.table_name(), mode)

        node_pit = net['_pit']['node']
        branch_pit = net['_pit']['branch']
        branch_lookups = get_lookup(net, "branch", "from_to")
        f, t = branch_lookups[cls.table_name()]

        res_table = net["res_" + cls.table_name()]

        from_nodes = get_from_nodes_corrected(branch_pit[f:t])
        t_from = node_pit[from_nodes, TINIT]
        tout = branch_pit[f:t, TOUTINIT]

        hg_array = get_component_array(net, cls.table_name(), mode='heat_transfer')
        mask_prpl = hg_array[:, cls.MODE] == cls.PR_PL
        mask_other = ~mask_prpl

        res_table['deltat_k'].values[:] = t_from - tout

        if np.any(mask_prpl):
            fluid = get_fluid(net)

            cp_i = fluid.get_heat_capacity(t_from[mask_prpl])
            cp_i1 = fluid.get_heat_capacity(tout[mask_prpl])
            mass = branch_pit[f:t, MDOTINIT][mask_prpl]

            res_table['qext_w'].values[mask_prpl] = mass * (cp_i1 * tout[mask_prpl] - cp_i * t_from[mask_prpl])

            mask_reverse = (branch_pit[f:t, MDOTINIT][mask_prpl] < 0) & \
                        ~np.isclose(branch_pit[f:t, MDOTINIT][mask_prpl], 0)
            if np.any(mask_reverse):
                raise UserWarning(
                    r'Your grid is badly modelled and would lead to a direction change in circulation pump %s'
                    % str(net[cls.table_name()].index[mask_prpl][mask_reverse].tolist())
                )

        if np.any(mask_other):
            res_table['qext_w'].values[mask_other] = -branch_pit[f:t, QEXT][mask_other]
