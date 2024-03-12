# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

from warnings import warn

import numpy as np
from numpy import dtype

from pandapipes.component_models.abstract_models.node_models import NodeComponent
from pandapipes.component_models.component_toolbox import p_correction_height_air
from pandapipes.constants import NORMAL_TEMPERATURE, NORMAL_PRESSURE
from pandapipes.pf.pipeflow_setup import add_table_lookup, \
    get_lookup, get_table_number
from pandapipes.properties.fluids import get_fluid, get_mixture_density, get_mixture_compressibility
from pandapipes.pf.derivative_toolbox import get_derivative_density_same, get_derivative_density_diff


class Junction(NodeComponent):
    """

    """

    @classmethod
    def table_name(cls):
        return "junction"

    @classmethod
    def create_node_lookups(cls, net, ft_lookups, table_lookup, idx_lookups, current_start,
                            current_table, internal_nodes_lookup):
        """
        Function which creates node lookups.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param ft_lookups:
        :type ft_lookups:
        :param table_lookup:
        :type table_lookup:
        :param idx_lookups:
        :type idx_lookups:
        :param current_start:
        :type current_start:
        :param current_table:
        :type current_table:
        :param internal_nodes_lookup:
        :type internal_nodes_lookup:
        :return:
        :rtype:
        """
        table_indices = net[cls.table_name()].index
        table_len = len(table_indices)
        end = current_start + table_len
        ft_lookups[cls.table_name()] = (current_start, end)
        add_table_lookup(table_lookup, cls.table_name(), current_table)
        if not table_len:
            idx_lookups[cls.table_name()] = np.array([], dtype=np.int32)
            idx_lookups[cls.table_name()][table_indices] = np.arange(table_len) + current_start
        else:
            idx_lookups[cls.table_name()] = -np.ones(table_indices.max() + 1, dtype=np.int32)
            idx_lookups[cls.table_name()][table_indices] = np.arange(table_len) + current_start
        return end, current_table + 1

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
        ft_lookup = get_lookup(net, "node", "from_to")
        table_nr = get_table_number(get_lookup(net, "node", "table"), cls.table_name())
        f, t = ft_lookup[cls.table_name()]

        junctions = net[cls.table_name()]
        junction_pit = node_pit[f:t, :]
        junction_pit[:, :] = np.array([table_nr, 0, net['_idx_node']['L']] + [0] * (net['_idx_node']['node_cols'] - 3))
        junction_pit[:, net['_idx_node']['ELEMENT_IDX']] = junctions.index.values
        junction_pit[:, net['_idx_node']['HEIGHT']] = junctions.height_m.values
        junction_pit[:, net['_idx_node']['PINIT']] = junctions.pn_bar.values
        junction_pit[:, net['_idx_node']['TINIT']] = junctions.tfluid_k.values
        junction_pit[:, net['_idx_node']['PAMB']] = p_correction_height_air(junction_pit[:, net['_idx_node']['HEIGHT']])
        junction_pit[:, net['_idx_node']['ACTIVE']] = junctions.in_service.values
        w = get_lookup(net, 'node', 'w')
        junction_pit[:, w] = 1 / len(net._fluid)


    @classmethod
    def adaption_before_derivatives_hydraulic(cls, net, branch_pit, node_pit, branch_lookups, node_lookups, options):
        f, t = node_lookups[cls.table_name()]
        junction_pit = node_pit[f:t, :]
        if len(net._fluid) == 1:
            junction_pit[:, net['_idx_node']['RHO']] = \
                get_fluid(net, net._fluid[0]).get_density(junction_pit[:, net['_idx_node']['TINIT']])
        else:
            for fluid in net._fluid:
                junction_pit[:, net['_idx_node'][fluid + '_RHO']] = \
                    get_fluid(net, fluid).get_density(junction_pit[:, net['_idx_node']['TINIT']])
            w = get_lookup(net, 'node', 'w')
            rho = get_lookup(net, 'node', 'rho')
            der_rho_same = get_lookup(net, 'node', 'deriv_rho_same')
            der_rho_diff = get_lookup(net, 'node', 'deriv_rho_diff')
            mf = junction_pit[:, w]
            rl = junction_pit[:, rho]
            temperature = junction_pit[:, net['_idx_node']['TINIT']]
            junction_pit[:, net['_idx_node']['RHO']] = get_mixture_density(net, temperature, mf)
            junction_pit[:, der_rho_same] = get_derivative_density_same(mf, rl)
            junction_pit[:, der_rho_diff] = get_derivative_density_diff(mf, rl)


    @classmethod
    def extract_results(cls, net, options, branch_results, mode):
        """
        Function that extracts certain results.

        :param mode:
        :type mode:
        :param net: The pandapipes network
        :type net: pandapipesNet
        :param options:
        :type options:
        :param branch_results:
        :type branch_results:
        :param mode:
        :type mode:
        :return: No Output.
        """
        res_table = net["res_" + cls.table_name()]

        f, t = get_lookup(net, "node", "from_to")[cls.table_name()]
        junction_pit = net["_pit"]["node"][f:t, :]

        if mode in ["hydraulics", "all"]:
            junctions_connected_hydraulic = get_lookup(net, "node", "active_hydraulics")[f:t]

            if np.any(junction_pit[junctions_connected_hydraulic, net['_idx_node']['PINIT']] < 0):
                warn(UserWarning('Pipeflow converged, however, the results are physically incorrect '
                                 'as pressure is negative at nodes %s'
                                 % junction_pit[
                                     junction_pit[:, net['_idx_node']['PINIT']] < 0, net['_idx_node']['ELEMENT_IDX']]))

        res_table["p_bar"].values[:] = junction_pit[:, net['_idx_node']['PINIT']]
        res_table["t_k"].values[:] = junction_pit[:, net['_idx_node']['TINIT']]
        numerator = NORMAL_PRESSURE * junction_pit[:, net['_idx_node']['TINIT']]
        if len(net._fluid) == 1:
            p = junction_pit[:, net['_idx_node']['PAMB']] + junction_pit[:, net['_idx_node']['PINIT']]
            normfactor = numerator * get_fluid(net, net._fluid[0]).get_compressibility(p) / (p * NORMAL_TEMPERATURE)
            res_table["rho_kg_per_m3"].values[:] = junction_pit[:,
                                                                  net['_idx_node']['RHO']] / normfactor
        else:
            w = get_lookup(net, 'node', 'w')
            mf = junction_pit[:, w]
            p = junction_pit[:, net['_idx_node']['PAMB']] + junction_pit[:, net['_idx_node']['PINIT']]

            normfactor = numerator * get_mixture_compressibility(
                net, junction_pit[:, net['_idx_node']['PINIT']], mf, junction_pit[:, net['_idx_node']['TINIT']]) / (p * NORMAL_TEMPERATURE)
            rho = get_mixture_density(net, junction_pit[:, net['_idx_node']['TINIT']], mf) / normfactor
            res_table["rho_kg_per_m3"].values[:] = rho
            for i, fluid in enumerate(net._fluid):
                normfactor = numerator * get_fluid(net, fluid).get_compressibility(p) / (p * NORMAL_TEMPERATURE)
                rho_fluid = get_fluid(net, fluid).get_density(junction_pit[:, net['_idx_node']['TINIT']]) / normfactor

                res_table["rho_kg_per_m3_%s" % fluid].values[:] = rho_fluid
                res_table["w_%s" % fluid].values[:] = np.round(junction_pit[:, w[i]], 6)

    @classmethod
    def get_component_input(cls):
        """

        :return:
        :rtype:
        """
        return [('name', dtype(object)),
                ('pn_bar', 'f8'),
                ("tfluid_k", 'f8'),
                ("height_m", 'f8'),
                ('in_service', 'bool'),
                ('type', dtype(object))]

    @classmethod
    def geodata(cls):
        """

        :return:
        :rtype:
        """
        return [("x", "f8"), ("y", "f8")]

    @classmethod
    def get_result_table(cls, net):
        """

        :param net: The pandapipes network
        :type net: pandapipesNet
        :return: (columns, all_float) - the column names and whether they are all float type. Only
                if False, returns columns as tuples also specifying the dtypes
        :rtype: (list, bool)
        """
        if len(net._fluid) == 1:
            return ["p_bar", "t_k", "rho_kg_per_m3"], True
        else:
            default = ["p_bar", "t_k", "rho_kg_per_m3"]

            add = ["rho_kg_per_m3_%s" % fluid for fluid in net._fluid]
            add = ["w_%s" % fluid for fluid in net._fluid] + add
            return default + add, True
