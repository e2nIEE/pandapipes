# Copyright (c) 2020-2021 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

from operator import itemgetter

import numpy as np
from numpy import dtype

from pandapipes.component_models.abstract_models.branch_wzerolength_models import \
    BranchWZeroLengthComponent
from pandapipes.constants import NORMAL_TEMPERATURE, NORMAL_PRESSURE
from pandapipes.idx_branch import STD_TYPE, VINIT, D, AREA, TL, LOSS_COEFFICIENT as LC, FROM_NODE,\
    TO_NODE, TINIT, PL
from pandapipes.idx_node import PINIT, PAMB
from pandapipes.pf.pipeflow_setup import get_net_option, get_fluid


# the Compressor class is an adapted pump (mainly copied pump code)
class Compressor(BranchWZeroLengthComponent):
    """

    """

    @classmethod
    def table_name(cls):
        return "compressor"

    @classmethod
    def active_identifier(cls):
        return "in_service"

    @classmethod
    def create_pit_branch_entries(cls, net, branch_pit):
        """
        Function which creates pit branch entries with a specific table.
        :param net: The pandapipes network
        :type net: pandapipesNet
        :param internal_pipe_number:
        :type internal_pipe_number:
        :return: No Output.
        """
        compressor_pit = super().create_pit_branch_entries(net, compressor_pit)
        std_types_lookup = np.array(list(net.std_type[cls.table_name()].keys()))
        std_type, pos = np.where(net[cls.table_name()]['std_type'].values
                                 == std_types_lookup[:, np.newaxis])
        compressor_pit[pos, STD_TYPE] = std_type
        compressor_pit[:, D] = 0.9  # TODO: what is this -> Dummy
        compressor_pit[:, AREA] = compressor_pit[:, D] ** 2 * np.pi / 4
        compressor_pit[:, LC] = 0

    @classmethod
    def adaption_before_derivatives(cls, net, branch_pit, node_pit, idx_lookups, options):
        f, t = idx_lookups[cls.table_name()]
        compressor_pit = branch_pit[f:t, :]
        fluid = get_fluid(net)

        # calculate the 'real' velocity and volumen flow:

        # get necessary parameters from pandapipes internal table (pit):
        area = compressor_pit[:, AREA] # TODO: what is this? -> (dummy) only relevant for v
        idx = compressor_pit[:, STD_TYPE].astype(int) # TODO: what is this? -> lookup, numeric ID
        # of std type
        std_types = np.array(list(net.std_type['compressor'].keys()))[idx]
        from_nodes = compressor_pit[:, FROM_NODE].astype(np.int32)
        to_nodes = compressor_pit[:, TO_NODE].astype(np.int32)
        v_mps = compressor_pit[:, VINIT]

        # get absolute pressure in Pa:
        p_scale = get_net_option(net, "p_scale") # TODO: what is this? -> DLo fragen
        p_from = node_pit[from_nodes, PAMB] + node_pit[from_nodes, PINIT] * p_scale
        p_to = node_pit[to_nodes, PAMB] + node_pit[to_nodes, PINIT] * p_scale
        numerator = NORMAL_PRESSURE * compressor_pit[:, TINIT] # TODO: what is this? -> normfactor
        if fluid.is_gas: # TODO: what is happening here?
            # consider volume flow at inlet
            normfactor_from = numerator * fluid.get_property("compressibility", p_from) \
                              / (p_from * NORMAL_TEMPERATURE)
            v_mean = v_mps * normfactor_from
        else:
            raise UserWarning("Compressors do not work with liquid fluids.")
        vol = v_mean * area

        # get the standard type for each compressor
        fcts = itemgetter(*std_types)(net['std_type']['compressor'])
        fcts = [fcts] if not isinstance(fcts, tuple) else fcts

        # use the get_pressure function of the standard type to calculate the pressure lift from
        # the volume flow
        pl = np.array(list(map(lambda x, y: x.get_pressure(y), fcts, vol)))
        compressor_pit[:, PL] = pl
        # TODO: add mass flow in result table
        # TODO: add pressure at from_junction and to_junction to result table

    @classmethod
    def calculate_temperature_lift(cls, net, compressor_pit, node_pit):
        """

        :param net:
        :type net:
        :param compressor_pit:
        :type compressor_pit:
        :param node_pit:
        :type node_pit:
        :return:
        :rtype:
        """
        compressor_pit[:, TL] = 0

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
        placement_table, compressor_pit, res_table = \
            super().extract_results(net, options, None, nodes_connected, branches_connected)
        res_table['deltap_bar'].values[placement_table] = compressor_pit[:, PL]

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
        return ["deltap_bar"], True
