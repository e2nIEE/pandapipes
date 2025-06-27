# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
from numpy import dtype

from pandapipes.component_models import (get_fluid, BranchWZeroLengthComponent, get_component_array,
                                         standard_branch_wo_internals_result_lookup)
from pandapipes.component_models.junction_component import Junction
from pandapipes.idx_branch import (MDOTINIT, QEXT, JAC_DERIV_DP1, JAC_DERIV_DM,
                                   JAC_DERIV_DP, LOAD_VEC_BRANCHES, TOUTINIT, JAC_DERIV_DT,
                                   JAC_DERIV_DTOUT, LOAD_VEC_BRANCHES_T, FLOW_RETURN_CONNECT)
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

class HeatConsumer(BranchWZeroLengthComponent):
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

    @classmethod
    def table_name(cls):
        return "heat_consumer"

    @classmethod
    def get_connected_node_type(cls):
        return Junction

    @classmethod
    def from_to_node_cols(cls):
        return "from_junction", "to_junction"

    @classmethod
    def active_identifier(cls):
        return "in_service"

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
        hc_pit = super().create_pit_branch_entries(net, branch_pit)
        qext = net[cls.table_name()].qext_w.values
        hc_pit[~np.isnan(qext), QEXT] = qext[~np.isnan(qext)]
        mdot = net[cls.table_name()].controlled_mdot_kg_per_s.values
        hc_pit[~np.isnan(mdot), MDOTINIT] = mdot[~np.isnan(mdot)]
        treturn = net[cls.table_name()].treturn_k.values
        mask_tr = ~np.isnan(treturn)
        hc_pit[mask_tr, TOUTINIT] = treturn[mask_tr]
        hc_pit[:, FLOW_RETURN_CONNECT] = True
        mask_q0 = qext == 0 & np.isnan(mdot)
        if np.any(mask_q0):
            logger.warning(r'qext_w is equals to zero for heat consumers with index %s. '
                           r'Therefore, the defined temperature control cannot be maintained.' \
                    %net[cls.table_name()].index[mask_q0])
        return hc_pit

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
        consumer_array = np.zeros(shape=(len(tbl), cls.internal_cols), dtype=np.float64)
        consumer_array[:, cls.DELTAT] = tbl.deltat_k.values
        consumer_array[:, cls.TRETURN] = tbl.treturn_k.values
        consumer_array[:, cls.QEXT] = tbl.qext_w.values
        consumer_array[:, cls.MASS] = tbl.controlled_mdot_kg_per_s.values
        mf = tbl.controlled_mdot_kg_per_s.values
        tr = tbl.treturn_k.values
        dt = tbl.deltat_k.values
        qe = tbl.qext_w.values
        mf = ~np.isnan(mf)
        tr = ~np.isnan(tr)
        dt = ~np.isnan(dt)
        qe = ~np.isnan(qe)
        consumer_array[mf & dt, cls.MODE] = cls.MF_DT
        consumer_array[mf & tr, cls.MODE] = cls.MF_TR
        consumer_array[qe & mf, cls.MODE] = cls.QE_MF
        consumer_array[qe & dt, cls.MODE] = cls.QE_DT
        consumer_array[qe & tr, cls.MODE] = cls.QE_TR
        component_pits[cls.table_name()] = consumer_array

    @classmethod
    def adaption_before_derivatives_hydraulic(cls, net, branch_pit, node_pit, idx_lookups, options):
        f, t = idx_lookups[cls.table_name()]
        hc_pit = branch_pit[f:t, :]
        consumer_array = get_component_array(net, cls.table_name())

        mask = consumer_array[:, cls.MODE] == cls.QE_DT
        if np.any(mask):
            cp = get_branch_cp(get_fluid(net), node_pit, hc_pit[mask])
            deltat = consumer_array[mask, cls.DELTAT]
            mass = hc_pit[mask, QEXT] / (cp * deltat)
            hc_pit[mask, MDOTINIT] = mass

    @classmethod
    def adaption_after_derivatives_hydraulic(cls, net, branch_pit, node_pit, idx_lookups, options):
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
        f, t = idx_lookups[cls.table_name()]
        consumer_array = get_component_array(net, cls.table_name())

        hc_pit = branch_pit[f:t, :]
        hc_pit[:, JAC_DERIV_DP] = 0
        hc_pit[:, JAC_DERIV_DP1] = 0
        hc_pit[:, JAC_DERIV_DM] = 1
        hc_pit[:, LOAD_VEC_BRANCHES] = 0

        mask = consumer_array[:, cls.MODE] == cls.QE_TR
        if np.any(mask):
            cp = get_branch_cp(get_fluid(net), node_pit, hc_pit)
            from_nodes = get_from_nodes_corrected(hc_pit)
            t_in = node_pit[from_nodes, TINIT]
            t_out = hc_pit[:, TOUTINIT]

            df_dm = - cp * (t_out - t_in)
            mask_equal = t_out >= t_in
            mask_zero = hc_pit[:, QEXT] == 0
            mask_ign = mask_equal | mask_zero
            hc_pit[mask & mask_ign, MDOTINIT] = 0
            hc_pit[mask & ~mask_ign, JAC_DERIV_DM] = df_dm[mask & ~mask_ign]
            hc_pit[mask, LOAD_VEC_BRANCHES] = - hc_pit[mask, QEXT] + df_dm[mask] * hc_pit[mask, MDOTINIT]

    @classmethod
    def adaption_before_derivatives_thermal(cls, net, branch_pit, node_pit, idx_lookups, options):
        f, t = idx_lookups[cls.table_name()]
        hc_pit = branch_pit[f:t, :]
        consumer_array = get_component_array(net, cls.table_name(), mode='heat_transfer')
        mask = consumer_array[:, cls.MODE] == cls.MF_DT
        if np.any(mask):
            cp = get_branch_cp(get_fluid(net), node_pit, hc_pit)
            q_ext = cp[mask] * hc_pit[mask, MDOTINIT] * consumer_array[mask, cls.DELTAT]
            hc_pit[mask, QEXT] = q_ext

        mask = consumer_array[:, cls.MODE] == cls.MF_TR
        if np.any(mask):
            cp = get_branch_cp(get_fluid(net), node_pit, hc_pit)
            from_nodes = get_from_nodes_corrected(hc_pit[mask])
            t_in = node_pit[from_nodes, TINIT]
            t_out = consumer_array[mask, cls.TRETURN]
            q_ext = cp[mask] * hc_pit[mask, MDOTINIT] * (t_in - t_out)
            hc_pit[mask, QEXT] = q_ext

    @classmethod
    def adaption_after_derivatives_thermal(cls, net, branch_pit, node_pit, idx_lookups, options):
        f, t = idx_lookups[cls.table_name()]
        hc_pit = branch_pit[f:t, :]
        consumer_array = get_component_array(net, cls.table_name(), mode='heat_transfer')

        mask= consumer_array[:, cls.MODE] == cls.QE_TR
        if np.any(mask):
            mask_ign = hc_pit[:, QEXT] == 0
            mask = mask & ~mask_ign
            hc_pit[mask, LOAD_VEC_BRANCHES_T] = 0
            hc_pit[mask, JAC_DERIV_DTOUT] = 1
            hc_pit[mask, JAC_DERIV_DT] = 0

    @classmethod
    def get_component_input(cls):
        """

        Get component input.

        :return:
        :rtype:
        """
        return [("name", dtype(object)), ("from_junction", "u4"), ("to_junction", "u4"), ("qext_w", "f8"),
                ("controlled_mdot_kg_per_s", "f8"), ("deltat_k", "f8"), ("treturn_k", "f8"),
                ("in_service", "bool"), ("type", dtype(object))]

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

        res_table['qext_w'].values[:] = branch_pit[f:t, QEXT]
        from_nodes = get_from_nodes_corrected(branch_pit[f:t])
        t_from = node_pit[from_nodes, TINIT]
        tout = branch_pit[f:t, TOUTINIT]
        res_table['deltat_k'].values[:] = t_from - tout
