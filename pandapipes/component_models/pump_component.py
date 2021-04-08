# Copyright (c) 2020-2021 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

from operator import itemgetter

import numpy as np
from numpy import dtype
from pandapipes.component_models.abstract_models import BranchWZeroLengthComponent, TINIT_NODE
from pandapipes.constants import NORMAL_TEMPERATURE, NORMAL_PRESSURE, R_UNIVERSAL, P_CONVERSION
from pandapipes.idx_branch import STD_TYPE, VINIT, D, AREA, TL, \
    LOSS_COEFFICIENT as LC, FROM_NODE, TO_NODE, TINIT, PL, LOAD_VEC_NODES, ELEMENT_IDX
from pandapipes.idx_node import PINIT, PAMB
from pandapipes.internals_toolbox import _sum_by_group
from pandapipes.pipeflow_setup import get_net_option, get_fluid, get_lookup


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
        p_scale = get_net_option(net, "p_scale")
        from_nodes = pump_pit[:, FROM_NODE].astype(np.int32)
        # to_nodes = pump_pit[:, TO_NODE].astype(np.int32)
        fluid = get_fluid(net)
        p_from = node_pit[from_nodes, PAMB] + node_pit[from_nodes, PINIT] * p_scale
        # p_to = node_pit[to_nodes, PAMB] + node_pit[to_nodes, PINIT] * p_scale
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
        placement_table, pump_pit, res_table = super().prepare_result_tables(net, options, node_name)
        res_table['deltap_bar'].values[placement_table] = pump_pit[:, PL]

        node_pit = net["_active_pit"]["node"]
        node_active_idx_lookup = get_lookup(net, "node", "index_active")[node_name]
        junction_idx_lookup = get_lookup(net, "node", "index")[node_name]
        from_junction_nodes = node_active_idx_lookup[junction_idx_lookup[
            net[cls.table_name()]["from_junction"].values[placement_table]]]
        to_junction_nodes = node_active_idx_lookup[junction_idx_lookup[
            net[cls.table_name()]["to_junction"].values[placement_table]]]

        p_to = node_pit[to_junction_nodes, PINIT]
        p_from = node_pit[from_junction_nodes, PINIT]

        from_nodes = pump_pit[:, FROM_NODE].astype(np.int32)
        to_nodes = pump_pit[:, TO_NODE].astype(np.int32)

        t0 = node_pit[from_nodes, TINIT_NODE]
        t1 = node_pit[to_nodes, TINIT_NODE]
        mf = pump_pit[:, LOAD_VEC_NODES]
        vf = pump_pit[:, LOAD_VEC_NODES] / get_fluid(net).get_density((t0 + t1) / 2)

        idx_active = pump_pit[:, ELEMENT_IDX]
        idx_sort, mf_sum, vf_sum, internal_pipes = \
            _sum_by_group(idx_active, mf, vf, np.ones_like(idx_active))
        res_table["mdot_to_kg_per_s"].values[placement_table] = -mf_sum / internal_pipes
        res_table["mdot_from_kg_per_s"].values[placement_table] = mf_sum / internal_pipes
        res_table["vdot_norm_m3_per_s"].values[placement_table] = vf_sum / internal_pipes

        if net.fluid.is_gas:
            # calculate ideal compression power
            compr = get_fluid(net).get_property("compressibility", p_from)
            molar_mass = net.fluid.get_molar_mass()  # [g/mol]
            R_spec = 1e3 * R_UNIVERSAL / molar_mass  # [J/(kg * K)]
            # 'kappa' heat capacity ratio:
            k = 1.4  # TODO: implement proper calculation of kappa
            w_real_isentr = (k / (k - 1)) * R_spec * compr * t0 * \
                            (np.divide(p_to, p_from) ** ((k - 1) / k) - 1)
            res_table['compr_power_w'].values[placement_table] = \
                w_real_isentr * abs(mf_sum / internal_pipes)
        else:
            res_table['compr_power_w'].values[placement_table] = \
                pump_pit[:, PL] * P_CONVERSION * vf_sum / internal_pipes

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
        result_columns = ["deltap_bar", "compr_power_w", "mdot_from_kg_per_s", "mdot_to_kg_per_s",
                          "vdot_norm_m3_per_s"]
        return result_columns, True
