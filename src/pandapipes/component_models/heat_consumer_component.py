# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
from numpy import dtype

from pandapipes.component_models import get_fluid, \
    BranchWZeroLengthComponent, get_component_array, standard_branch_wo_internals_result_lookup
from pandapipes.component_models.junction_component import Junction
from pandapipes.idx_branch import D, AREA, VINIT, ALPHA, QEXT, \
    RHO, TEXT, JAC_DERIV_DP1, JAC_DERIV_DV, JAC_DERIV_DP, LOAD_VEC_BRANCHES, TL
from pandapipes.pf.result_extraction import extract_branch_results_without_internals


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

    # numbering of given parameters (for mdot, qext, deltat, treturn)
    MF = 0
    QE = 1
    DT = 2
    TR = 3

    # heat consumer modes (sum of combinations of given parameters)
    MF_QE = 1
    MF_DT = 2
    MF_TR = 4
    QE_DT = 3
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
        hs_pit = super().create_pit_branch_entries(net, branch_pit)
        hs_pit[:, D] = net[cls.table_name()].diameter_m.values
        hs_pit[:, AREA] = hs_pit[:, D] ** 2 * np.pi / 4
        hs_pit[:, VINIT] = (net[cls.table_name()].controlled_mdot_kg_per_s.values /
                            (hs_pit[:, AREA] * hs_pit[:, RHO]))
        hs_pit[:, ALPHA] = 0
        hs_pit[:, QEXT] = net[cls.table_name()].qext_w.values
        hs_pit[:, TEXT] = 293.15
        return hs_pit

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
        consumer_array[:, cls.MASS] = tbl.controlled_mdot_kg_per_s.values
        consumer_array[:, cls.QEXT] = tbl.qext_w.values
        consumer_array[:, cls.DELTAT] = tbl.deltat_k.values
        consumer_array[:, cls.TRETURN] = tbl.treturn_k.values
        mf = ~np.isnan(consumer_array[:, cls.MASS])
        qe = ~np.isnan(consumer_array[:, cls.QEXT])
        dt = ~np.isnan(consumer_array[:, cls.DELTAT])
        tr = ~np.isnan(consumer_array[:, cls.TRETURN])
        consumer_array[:, cls.MODE] = np.sum([mf, qe, dt, tr], axis=0)
        component_pits[cls.table_name()] = consumer_array

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
        fc_branch_pit = branch_pit[f:t, :]
        fc_array = get_component_array(net, cls.table_name())
        # TODO: this is more precise, but slower:
        #       np.isin(fc_array[:, cls.MODE], [cls.MF_QE, cls.MF_DT, cls.MF_TR])
        mdot_controlled = ~np.isnan(fc_array[:, cls.MASS])
        fc_branch_pit[mdot_controlled, JAC_DERIV_DP] = 0
        fc_branch_pit[mdot_controlled, JAC_DERIV_DP1] = 0
        fc_branch_pit[mdot_controlled, JAC_DERIV_DV] = 1
        fc_branch_pit[mdot_controlled, LOAD_VEC_BRANCHES] = 0

    # @classmethod
    # def adaption_before_derivatives_thermal(cls, net, branch_pit, node_pit, idx_lookups, options):
    #     f, t = idx_lookups[cls.table_name()]
    #     hs_pit = branch_pit[f:t, :]
    #     mask_t_return = ~np.isnan(hs_pit[:, TRETURN])
    #     hs_pit[mask_t_return, TINIT_OUT] = hs_pit[mask_t_return, TINIT_OUT] - hs_pit[mask_t_return, DELTAT]
    #
    #
    # @classmethod
    # def adaption_after_derivatives_thermal(cls, net, branch_pit, node_pit, idx_lookups, options):
    #     """
    #
    #     :param net:
    #     :type net:
    #     :param branch_component_pit:
    #     :type branch_component_pit:
    #     :param node_pit:
    #     :type node_pit:
    #     :return:
    #     :rtype:
    #     """
    #     # -(rho * area * cp * v_init * (-t_init_i + t_init_i1 - tl)
    #     #   - alpha * (t_amb - t_m) * length + qext)
    #
    #     f, t = idx_lookups[cls.table_name()]
    #     hs_pit = branch_pit[f:t, :]
    #     from_nodes = hs_pit[:, FROM_NODE_T].astype(np.int32)
    #
    #     mask_qext = ~np.isnan(hs_pit[:, QEXT])
    #     mask_deltat = ~np.isnan(hs_pit[:, DELTAT])
    #     mask_t_return = ~np.isnan(hs_pit[:, TRETURN])
    #     mask_mass = ~np.isnan(hs_pit[:, MASS])
    #     hs_pit[mask_t_return | mask_deltat, JAC_DERIV_DT1] = 0

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
                ("controlled_mdot_kg_per_s", "f8"),
                ("qext_w", 'f8'),
                ("deltat_k", 'f8'),
                ("treturn_k", 'f8'),
                ("diameter_m", "f8"),
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
        if get_fluid(net).is_gas:
            output = ["v_from_m_per_s", "v_to_m_per_s", "v_mean_m_per_s", "p_from_bar", "p_to_bar",
                      "t_from_k", "t_to_k", "mdot_from_kg_per_s", "mdot_to_kg_per_s",
                      "vdot_norm_m3_per_s", "reynolds", "lambda", "normfactor_from",
                      "normfactor_to"]
        else:
            output = ["v_mean_m_per_s", "p_from_bar", "p_to_bar", "t_from_k", "t_to_k",
                      "mdot_from_kg_per_s", "mdot_to_kg_per_s", "vdot_norm_m3_per_s", "reynolds",
                      "lambda"]
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

        extract_branch_results_without_internals(net, branch_results, required_results_hyd,
                                                 required_results_ht, cls.table_name(), mode)
