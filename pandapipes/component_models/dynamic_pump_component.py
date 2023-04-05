# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

from operator import itemgetter

import numpy as np
from numpy import dtype

from pandapipes.component_models.junction_component import Junction
from pandapipes.component_models.pump_component import Pump
from pandapipes.component_models.abstract_models.branch_wzerolength_models import \
    BranchWZeroLengthComponent
from pandapipes.constants import NORMAL_TEMPERATURE, NORMAL_PRESSURE, R_UNIVERSAL, P_CONVERSION, \
     GRAVITATION_CONSTANT
from pandapipes.idx_branch import STD_TYPE, VINIT, D, AREA, TL, LOSS_COEFFICIENT as LC, FROM_NODE, \
    TINIT, PL, Kv_max, ACTUAL_POS, DESIRED_MV, RHO
from pandapipes.idx_node import PINIT, PAMB, TINIT as TINIT_NODE
from pandapipes.pf.pipeflow_setup import get_fluid, get_net_option, get_lookup
from pandapipes.pf.result_extraction import extract_branch_results_without_internals

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


class DynamicPump(BranchWZeroLengthComponent):
    """
    """
    # class attributes
    kwargs = None
    prev_act_pos = None
    time_step = 0

    @classmethod
    def from_to_node_cols(cls):
        return "from_junction", "to_junction"

    @classmethod
    def table_name(cls):
        return "dynamic_pump"

    @classmethod
    def active_identifier(cls):
        return "in_service"

    @classmethod
    def get_connected_node_type(cls):
        return Junction

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
        pump_pit = super().create_pit_branch_entries(net, branch_pit)
        std_types_lookup = np.array(list(net.std_types[cls.table_name()].keys()))
        std_type, pos = np.where(net[cls.table_name()]['std_type'].values
                                 == std_types_lookup[:, np.newaxis])
        pump_pit[pos, STD_TYPE] = std_type
        pump_pit[:, D] = 0.1
        pump_pit[:, AREA] = pump_pit[:, D] ** 2 * np.pi / 4
        pump_pit[:, LC] = 0
        pump_pit[:, ACTUAL_POS] = net[cls.table_name()].actual_pos.values
        pump_pit[:, DESIRED_MV] = net[cls.table_name()].desired_mv.values

    @classmethod
    def plant_dynamics(cls, dt, desired_mv, dyn_pump_tbl):
        """
        Takes in the desired valve position (MV value) and computes the actual output depending on
        equipment lag parameters.
        Returns Actual pump position
        """

        time_const_s = dyn_pump_tbl.time_const_s.values
        a = np.divide(dt, time_const_s + dt)
        actual_pos = (1 - a) * cls.prev_act_pos + a * desired_mv
        cls.prev_act_pos = actual_pos

        return actual_pos

        # if cls.kwargs.__contains__("act_dynamics"):
        #     typ = cls.kwargs['act_dynamics']
        # else:
        #     # default to instantaneous
        #     return desired_mv
        #
        # # linear
        # if typ == "l":
        #
        #     # TODO: equation for linear
        #     actual_pos = desired_mv
        #
        # # first order
        # elif typ == "fo":
        #
        #     a = np.divide(dt, cls.kwargs['time_const_s'] + dt)
        #     actual_pos = (1 - a) * cls.prev_act_pos + a * desired_mv
        #
        #     cls.prev_act_pos = actual_pos
        #
        # # second order
        # elif typ == "so":
        #     # TODO: equation for second order
        #     actual_pos = desired_mv
        #
        # else:
        #     # instantaneous - when incorrect option selected
        #     actual_pos = desired_mv
        #
        # return actual_pos

    @classmethod
    def adaption_before_derivatives_hydraulic(cls, net, branch_pit, node_pit, idx_lookups, options):
        # calculation of pressure lift
        dt = net['_options']['dt']
        f, t = idx_lookups[cls.table_name()]
        dyn_pump_tbl = net[cls.table_name()]
        pump_pit = branch_pit[f:t, :]
        area = pump_pit[:, AREA]
        idx = pump_pit[:, STD_TYPE].astype(int)
        std_types = np.array(list(net.std_types['dynamic_pump'].keys()))[idx]
        from_nodes = pump_pit[:, FROM_NODE].astype(np.int32)
        # to_nodes = pump_pit[:, TO_NODE].astype(np.int32)
        fluid = get_fluid(net)
        p_from = node_pit[from_nodes, PAMB] + node_pit[from_nodes, PINIT]
        # p_to = node_pit[to_nodes, PAMB] + node_pit[to_nodes, PINIT]
        numerator = NORMAL_PRESSURE * pump_pit[:, TINIT]
        v_mps = pump_pit[:, VINIT]
        desired_mv = dyn_pump_tbl.desired_mv.values
        cur_actual_pos = dyn_pump_tbl.actual_pos.values
        pump_pit[:, DESIRED_MV] = dyn_pump_tbl.desired_mv.values
        vol_m3_s = v_mps * area
        vol_m3_h = vol_m3_s * 3600

        if get_net_option(net, "time_step") == cls.time_step:
            # a controller timeseries is running
            actual_pos = cls.plant_dynamics(dt, desired_mv, dyn_pump_tbl)
            # Account for nan's when FCE are in manual
            update_pos = np.where(np.isnan(actual_pos))
            actual_pos[update_pos] = cur_actual_pos[update_pos]
            pump_pit[:, ACTUAL_POS] = actual_pos
            dyn_pump_tbl.actual_pos = actual_pos
            cls.time_step += 1

        else:  # Steady state analysis
            actual_pos = dyn_pump_tbl.actual_pos.values


        fcts = itemgetter(*std_types)(net['std_types']['dynamic_pump'])
        fcts = [fcts] if not isinstance(fcts, tuple) else fcts
        m_head = np.array(list(map(lambda x, y, z: x.get_m_head(y, z), fcts, vol_m3_s, actual_pos)))  # m head
        rho= pump_pit[:, RHO]
        prsr_lift = np.divide((rho * GRAVITATION_CONSTANT * m_head), P_CONVERSION)[0]  # bar
        dyn_pump_tbl.p_lift = prsr_lift
        dyn_pump_tbl.m_head = m_head
        pump_pit[:, PL] = prsr_lift

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
        branch_component_pit[:, TL] = 0

    @classmethod
    def extract_results(cls, net, options, branch_results, nodes_connected, branches_connected):
        """
        Function that extracts certain results.
        :param nodes_connected:
        :type nodes_connected:
        :param branches_connected:
        :type branches_connected:
        :param branch_results:
        :type branch_results:
        :param net: The pandapipes network
        :type net: pandapipesNet
        :param options:
        :type options:
        :return: No Output.
        """
        calc_compr_pow = options['calc_compression_power']

        required_results = [
            ("p_from_bar", "p_from"), ("p_to_bar", "p_to"), ("t_from_k", "temp_from"),
            ("t_to_k", "temp_to"), ("mdot_to_kg_per_s", "mf_to"), ("mdot_from_kg_per_s", "mf_from"),
            ("vdot_norm_m3_per_s", "vf"), ("deltap_bar", "pl"), ("desired_mv", "desired_mv"),
            ("actual_pos", "actual_pos")
        ]

        if get_fluid(net).is_gas:
            required_results.extend([
                ("v_from_m_per_s", "v_gas_from"), ("v_to_m_per_s", "v_gas_to"),
                ("normfactor_from", "normfactor_from"), ("normfactor_to", "normfactor_to")
            ])
        else:
            required_results.extend([("v_mean_m_per_s", "v_mps")])

        extract_branch_results_without_internals(net, branch_results, required_results,
                                                 cls.table_name(), branches_connected)

        if calc_compr_pow:
            f, t = get_lookup(net, "branch", "from_to")[cls.table_name()]
            res_table = net["res_" + cls.table_name()]
            if net.fluid.is_gas:
                p_from = branch_results["p_from"][f:t]
                p_to = branch_results["p_to"][f:t]
                from_nodes = branch_results["from_nodes"][f:t]
                t0 = net["_pit"]["node"][from_nodes, TINIT_NODE]
                mf_sum_int = branch_results["mf_from"][f:t]
                # calculate ideal compression power
                compr = get_fluid(net).get_property("compressibility", p_from)
                try:
                    molar_mass = net.fluid.get_molar_mass()  # [g/mol]
                except UserWarning:
                    logger.error('Molar mass is missing in your fluid. Before you are able to '
                                 'retrieve the compression power make sure that the molar mass is'
                                 ' defined')
                else:
                    r_spec = 1e3 * R_UNIVERSAL / molar_mass  # [J/(kg * K)]
                    # 'kappa' heat capacity ratio:
                    k = 1.4  # TODO: implement proper calculation of kappa
                    w_real_isentr = (k / (k - 1)) * r_spec * compr * t0 * \
                                    (np.divide(p_to, p_from) ** ((k - 1) / k) - 1)
                    res_table['compr_power_mw'].values[:] = \
                        w_real_isentr * np.abs(mf_sum_int) / 10 ** 6
            else:
                vf_sum_int = branch_results["vf"][f:t]
                pl = branch_results["pl"][f:t]
                res_table['compr_power_mw'].values[:] = pl * P_CONVERSION * vf_sum_int / 10 ** 6

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
                ("std_type", dtype(object)),
                ("in_service", 'bool'),
                ("actual_pos", "f8"),
                ("p_lift", "f8"),
                ('m_head', "f8"),
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
        calc_compr_pow = get_net_option(net, 'calc_compression_power')

        if get_fluid(net).is_gas:
            output = ["deltap_bar",
                      "v_from_m_per_s", "v_to_m_per_s",
                      "p_from_bar", "p_to_bar",
                      "t_from_k", "t_to_k", "mdot_from_kg_per_s", "mdot_to_kg_per_s",
                      "vdot_norm_m3_per_s", "normfactor_from", "normfactor_to",  "desired_mv", "actual_pos"]
            # TODO: inwieweit sind diese Angaben bei imaginärem Durchmesser sinnvoll?
        else:
            output = ["deltap_bar", "v_mean_m_per_s", "p_from_bar", "p_to_bar", "t_from_k",
                      "t_to_k", "mdot_from_kg_per_s", "mdot_to_kg_per_s", "vdot_norm_m3_per_s",
                       "desired_mv", "actual_pos"]
        if calc_compr_pow:
            output += ["compr_power_mw"]

        return output, True