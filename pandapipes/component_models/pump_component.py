# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

from operator import itemgetter

import numpy as np
from numpy import dtype

from pandapipes.component_models.abstract_models.branch_wzerolength_models import \
    BranchWZeroLengthComponent
from pandapipes.component_models.component_toolbox import get_component_array
from pandapipes.component_models.junction_component import Junction
from pandapipes.constants import NORMAL_TEMPERATURE, NORMAL_PRESSURE, R_UNIVERSAL, P_CONVERSION
from pandapipes.pf.pipeflow_setup import get_fluid, get_net_option, get_lookup
from pandapipes.pf.result_extraction import extract_branch_results_without_internals
from pandapipes.properties.fluids import is_fluid_gas, get_mixture_molar_mass, get_mixture_compressibility, get_fluid

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)

class Pump(BranchWZeroLengthComponent):
    """

    """
    STD_TYPE = 0

    internal_cols = 1

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
        pump_pit[:, net['_idx_branch']['D']] = 0.1
        pump_pit[:, net['_idx_branch']['AREA']] = pump_pit[:, net['_idx_branch']['D']] ** 2 * np.pi / 4
        pump_pit[:, net['_idx_branch']['LOSS_COEFFICIENT']] = 0

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
        pump_array = np.zeros(shape=(len(tbl), cls.internal_cols), dtype=np.float64)
        std_types_lookup = get_std_type_lookup(net, cls.table_name())
        std_type, pos = np.where(net[cls.table_name()]['std_type'].values
                                 == std_types_lookup[:, np.newaxis])
        pump_array[pos, cls.STD_TYPE] = std_type
        component_pits[cls.table_name()] = pump_array

    @classmethod
    def adaption_before_derivatives_hydraulic(cls, net, branch_pit, node_pit, idx_lookups, options):
        super().adaption_before_derivatives_hydraulic(net, branch_pit, node_pit, idx_lookups, options)
        # calculation of pressure lift
        f, t = idx_lookups[cls.table_name()]
        pump_branch_pit = branch_pit[f:t, :]
        area = pump_branch_pit[:, net['_idx_branch']['AREA']]

        pump_array = get_component_array(net, cls.table_name())
        idx = pump_array[:, cls.STD_TYPE].astype(np.int32)
        std_types = get_std_type_lookup(net, cls.table_name())[idx]

        from_nodes = pump_branch_pit[:, net['_idx_branch']['FROM_NODE']].astype(np.int32)
        # to_nodes = pump_branch_pit[:, TO_NODE].astype(np.int32)
        fluid = get_fluid(net)
        p_from = node_pit[from_nodes, net['_idx_node']['PAMB']] + node_pit[from_nodes, PINIT]
        # p_to = node_pit[to_nodes, PAMB] + node_pit[to_nodes, PINIT]
        numerator = NORMAL_PRESSURE * pump_branch_pit[:, net['_idx_branch']['TINIT']]
        v_mps = pump_branch_pit[:, net['_idx_branch']['VINIT']]
        if is_fluid_gas(net):
            # consider volume flow at inlet
            if len(net._fluid) == 1:
                fluid = net._fluid[0]
                normfactor_from = numerator * get_fluid(net, fluid).get_compressibility(p_from) \
                                  / (p_from * NORMAL_TEMPERATURE)
            else:
                w = get_lookup(net, 'branch', 'w')
                mass_fraction = pump_pit[:, w]
                normfactor_from = numerator * get_mixture_compressibility(net, p_from, mass_fraction, node_pit[from_nodes, net['_idx_node']['TINIT']]) \
                                  / (p_from * NORMAL_TEMPERATURE)
            v_mean = v_mps * normfactor_from
        else:
            v_mean = v_mps
        vol = v_mean * area
        if len(std_types):
            fcts = itemgetter(*std_types)(net['std_types']['pump'])
            fcts = [fcts] if not isinstance(fcts, tuple) else fcts
            pl = np.array(list(map(lambda x, y: x.get_pressure(y), fcts, vol)))
            pump_branch_pit[:, net['_idx_branch']['PL']] = pl

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
        branch_component_pit[:, net['_idx_branch']['TL']] = 0

    @classmethod
    def extract_results(cls, net, options, branch_results, mode):
        """
        Function that extracts certain results.

        :param branch_results:
        :type branch_results:
        :param net: The pandapipes network
        :type net: pandapipesNet
        :param options:
        :type options:
        :param mode:
        :type mode:
        :return: No Output.
        """
        calc_compr_pow = options['calc_compression_power']

        required_results_hyd = [
            ("p_from_bar", "p_from"), ("p_to_bar", "p_to"), ("mdot_to_kg_per_s", "mf_to"),
            ("mdot_from_kg_per_s", "mf_from"), ("vdot_norm_m3_per_s", "vf"), ("deltap_bar", "pl"),
        ]
        required_results_ht = [("t_from_k", "temp_from"), ("t_to_k", "temp_to")]

        if is_fluid_gas(net):
            required_results_hyd.extend([
                ("v_from_m_per_s", "v_gas_from"), ("v_to_m_per_s", "v_gas_to"),
                ("normfactor_from", "normfactor_from"), ("normfactor_to", "normfactor_to")
            ])
        else:
            required_results_hyd.extend([("v_mean_m_per_s", "v_mps")])

        extract_branch_results_without_internals(net, branch_results, required_results_hyd,
                                                 required_results_ht, cls.table_name(), mode)

        if calc_compr_pow:
            f, t = get_lookup(net, "branch", "from_to")[cls.table_name()]
            from_nodes = branch_results["from_nodes"][f:t]

            res_table = net["res_" + cls.table_name()]
            pl = branch_results["pl"][f:t]
            if is_fluid_gas(net):
                p_from = branch_results["p_abs_from"][f:t]
                p_to = branch_results["p_abs_to"][f:t]
                t0 = net["_pit"]["node"][from_nodes, net['_idx_node']['TINIT']]
                mf_sum_int = branch_results["mf_from"][f:t]
                # calculate ideal compression power
                if len(net._fluid) == 1:
                    fluid = net._fluid[0]
                    compr = get_fluid(net, fluid).get_compressibility(p_from)
                    try:
                        molar_mass = get_fluid(net, fluid).get_molar_mass()  # [g/mol]
                    except UserWarning:
                        logger.error('Molar mass is missing in your fluid. Before you are able to retrieve '
                                     'the compression power make sure that the molar mass is defined')
                        molar_mass_given=False
                    else:
                        molar_mass_given=True
                else:
                    node_pit = net["_active_pit"]["node"]
                    w = get_lookup(net, 'node', 'w')
                    mass_fraction = node_pit[:, w]
                    compr = get_mixture_compressibility(net, p_from, mass_fraction, node_pit[from_nodes, net['_idx_node']['TINIT']])
                    try:
                        molar_mass = get_mixture_molar_mass(net, mass_fraction)  # [g/mol]
                    except UserWarning:
                        logger.error('Molar mass is missing in your fluid. Before you are able to retrieve '
                                     'the compression power make sure that the molar mass is defined')
                        molar_mass_given=False
                    else:
                        molar_mass_given=True
                if molar_mass_given:
                    r_spec = 1e3 * R_UNIVERSAL / molar_mass  # [J/(kg * K)]
                    if len(net._fluid) == 1:
                        fluid = net._fluid[0]
                        compr = get_fluid(net, fluid).get_heat_capacity(t0)
                    else:
                        node_pit = net["_active_pit"]["node"]
                        w = get_lookup(net, 'node', 'w')
                        mass_fraction = node_pit[:, w]
                        compr = get_mixture_heat_capacity(net, p_from, mass_fraction, t0)
                    cp = net.fluid.get_heat_capacity(t0)
                    cv = cp - r_spec
                    k = cp/cv  # 'kappa' heat capacity ratio
                    w_real_isentr = (k / (k - 1)) * r_spec * compr * t0 * \
                                    (np.divide(p_to, p_from) ** ((k - 1) / k) - 1)
                    res_table['compr_power_mw'].values[:] = w_real_isentr * np.abs(mf_sum_int) / 10 ** 6
            else:
                vf_sum_int = branch_results["vf"][f:t]
                res_table['compr_power_mw'].values[:] = pl * P_CONVERSION * vf_sum_int / 10 ** 6
            res_table['deltap_bar'].values[:] = pl

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

        if is_fluid_gas(net):
            output = ["deltap_bar",
                      "v_from_m_per_s", "v_to_m_per_s",
                      "p_from_bar", "p_to_bar",
                      "t_from_k", "t_to_k", "mdot_from_kg_per_s", "mdot_to_kg_per_s",
                      "vdot_norm_m3_per_s", "normfactor_from", "normfactor_to"]
            # TODO: inwieweit sind diese Angaben bei imaginärem Durchmesser sinnvoll?
        else:
            output = ["deltap_bar", "v_mean_m_per_s", "p_from_bar", "p_to_bar", "t_from_k",
                      "t_to_k", "mdot_from_kg_per_s", "mdot_to_kg_per_s", "vdot_norm_m3_per_s"]
        if calc_compr_pow:
            output += ["compr_power_mw"]

        return output, True


def get_std_type_lookup(net, table_name):
    return np.array(list(net.std_types[table_name].keys()))
