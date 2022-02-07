# Copyright (c) 2020-2022 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

from operator import itemgetter

import numpy as np
from numpy import dtype

from pandapipes.component_models.abstract_models import BranchWZeroLengthComponent, TINIT_NODE
from pandapipes.constants import NORMAL_TEMPERATURE, NORMAL_PRESSURE, R_UNIVERSAL, P_CONVERSION
from pandapipes.idx_branch import STD_TYPE, VINIT, D, AREA, TL, \
    LOSS_COEFFICIENT as LC, FROM_NODE, TINIT, PL
from pandapipes.idx_node import PINIT, PAMB
from pandapipes.pipeflow_setup import get_fluid, get_net_option


class Pump(BranchWZeroLengthComponent):
    """

    """

    @classmethod
    def from_to_node_cols(cls):
        return "from_junction", "to_junction"

    @classmethod
    def table_name(cls):
        return "pump"

    @classmethod
    def active_identifier(cls):
        return "in_service"

    @classmethod
    def create_pit_branch_entries(cls, net, pump_pit, node_name):
        """
        Function which creates pit branch entries with a specific table.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param pump_pit:
        :type pump_pit:
        :param internal_pipe_number:
        :type internal_pipe_number:
        :return: No Output.
        """
        pump_pit = super().create_pit_branch_entries(net, pump_pit, node_name)
        std_types_lookup = np.array(list(net.std_type[cls.table_name()].keys()))
        std_type, pos = np.where(net[cls.table_name()]['std_type'].values
                                 == std_types_lookup[:, np.newaxis])
        pump_pit[pos, STD_TYPE] = std_type
        pump_pit[:, D] = 0.1
        pump_pit[:, AREA] = pump_pit[:, D] ** 2 * np.pi / 4
        pump_pit[:, LC] = 0

    @classmethod
    def calculate_pressure_lift(cls, net, pump_pit, node_pit):
        """

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param pump_pit:
        :type pump_pit:
        :param node_pit:
        :type node_pit:
        :return: power stroke
        :rtype: float
        """
        area = pump_pit[:, AREA]
        idx = pump_pit[:, STD_TYPE].astype(int)
        std_types = np.array(list(net.std_type['pump'].keys()))[idx]
        from_nodes = pump_pit[:, FROM_NODE].astype(np.int32)
        # to_nodes = pump_pit[:, TO_NODE].astype(np.int32)
        fluid = get_fluid(net)
        p_from = node_pit[from_nodes, PAMB] + node_pit[from_nodes, PINIT]
        # p_to = node_pit[to_nodes, PAMB] + node_pit[to_nodes, PINIT]
        numerator = NORMAL_PRESSURE * pump_pit[:, TINIT]
        v_mps = pump_pit[:, VINIT]
        if fluid.is_gas:
            # consider volume flow at inlet
            normfactor_from = numerator * fluid.get_property("compressibility", p_from) \
                              / (p_from * NORMAL_TEMPERATURE)
            v_mean = v_mps * normfactor_from
        else:
            v_mean = v_mps
        vol = v_mean * area
        fcts = itemgetter(*std_types)(net['std_type']['pump'])
        fcts = [fcts] if not isinstance(fcts, tuple) else fcts
        pl = np.array(list(map(lambda x, y: x.get_pressure(y), fcts, vol)))
        pump_pit[:, PL] = pl

    @classmethod
    def calculate_temperature_lift(cls, net, pump_pit, node_pit):
        """

        :param net:
        :type net:
        :param pump_pit:
        :type pump_pit:
        :param node_pit:
        :type node_pit:
        :return:
        :rtype:
        """
        pump_pit[:, TL] = 0

    @classmethod
    def extract_results(cls, net, options, node_name):
        """
        Function that extracts certain results.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param options:
        :type options:
        :return: No Output.
        """
        calc_compr_pow = options['calc_compression_power']
        if calc_compr_pow:
            placement_table, res_table, pump_pit, node_pit = super().extract_results(net, options, node_name)
            p_from = res_table['p_from_bar'].values[placement_table]
            p_to = res_table['p_to_bar'].values[placement_table]
            from_nodes = pump_pit[:, FROM_NODE].astype(np.int32)
            t0 = node_pit[from_nodes, TINIT_NODE]
            vf_sum_int = res_table["vdot_norm_m3_per_s"].values[placement_table]
            mf_sum_int = res_table["mdot_from_kg_per_s"].values[placement_table]
            if net.fluid.is_gas:
                # calculate ideal compression power
                compr = get_fluid(net).get_property("compressibility", p_from)
                molar_mass = net.fluid.get_molar_mass()  # [g/mol]
                R_spec = 1e3 * R_UNIVERSAL / molar_mass  # [J/(kg * K)]
                # 'kappa' heat capacity ratio:
                k = 1.4  # TODO: implement proper calculation of kappa
                w_real_isentr = (k / (k - 1)) * R_spec * compr * t0 * \
                                (np.divide(p_to, p_from) ** ((k - 1) / k) - 1)
                res_table['compr_power_mw'].values[placement_table] = \
                    w_real_isentr * abs(mf_sum_int) / 10 ** 6
            else:
                res_table['compr_power_mw'].values[placement_table] = \
                    pump_pit[:, PL] * P_CONVERSION * vf_sum_int / 10 ** 6
        else:
            placement_table, pump_pit, res_table = super().prepare_result_tables(net, options, node_name)
        res_table['deltap_bar'].values[placement_table] = pump_pit[:, PL]

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
                      "vdot_norm_m3_per_s", "normfactor_from", "normfactor_to"]
            # TODO: inwieweit sind diese Angaben bei imaginärem Durchmesser sinnvoll?
        else:
            output = ["deltap_bar",
                      "v_mean_m_per_s", "p_from_bar", "p_to_bar", "t_from_k", "t_to_k",
                      "mdot_from_kg_per_s", "mdot_to_kg_per_s", "vdot_norm_m3_per_s"]
        if calc_compr_pow:
            output += ["compr_power_mw"]

        return output, True
