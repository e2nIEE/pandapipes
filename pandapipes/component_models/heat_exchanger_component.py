# Copyright (c) 2020-2022 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
from numpy import dtype
from pandapipes.component_models.abstract_models import BranchWZeroLengthComponent
from pandapipes.idx_branch import PL, TL, ALPHA, \
    TEXT, QEXT, T_OUT, D, AREA, LOSS_COEFFICIENT as LC
from pandapipes.pipeflow_setup import get_fluid

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class HeatExchanger(BranchWZeroLengthComponent):

    @classmethod
    def from_to_node_cols(cls):
        return "from_junction", "to_junction"

    @classmethod
    def table_name(cls):
        return "heat_exchanger"

    @classmethod
    def active_identifier(cls):
        return "in_service"

    @classmethod
    def create_pit_branch_entries(cls, net, heat_exchanger_pit, node_name):
        """
        Function which creates pit branch entries with a specific table.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param heat_exchanger_pit:
        :type heat_exchanger_pit:
        :param node_name:
        :type node_name:
        :return: No Output.
        """
        heat_exchanger_pit = super().create_pit_branch_entries(net, heat_exchanger_pit, node_name)
        heat_exchanger_pit[:, D] = net[cls.table_name()].diameter_m.values
        heat_exchanger_pit[:, AREA] = heat_exchanger_pit[:, D] ** 2 * np.pi / 4
        heat_exchanger_pit[:, LC] = net[cls.table_name()].loss_coefficient.values
        heat_exchanger_pit[:, ALPHA] = 0
        heat_exchanger_pit[:, QEXT] = net[cls.table_name()].qext_w.values
        heat_exchanger_pit[:, TEXT] = 293.15
        heat_exchanger_pit[:, T_OUT] = 307

    @classmethod
    def calculate_pressure_lift(cls, net, he_pit, node_pit):
        """

        :param net:
        :type net:
        :param he_pit:
        :type he_pit:
        :param node_pit:
        :type node_pit:
        :return:
        :rtype:
        """
        he_pit[:, PL] = 0

    @classmethod
    def calculate_temperature_lift(cls, net, he_pit, node_pit):
        """

        :param net:
        :type net:
        :param he_pit:
        :type he_pit:
        :param node_pit:
        :type node_pit:
        :return:
        :rtype:
        """
        he_pit[:, TL] = 0

    @classmethod
    def get_component_input(cls):
        """

        :return:
        :rtype:
        """
        return [("name", dtype(object)),
                ("from_junction", "u4"),
                ("to_junction", "u4"),
                ("diameter_m", "f8"),
                ("qext_w", 'f8'),
                ("loss_coefficient", "f8"),
                ("in_service", 'bool'),
                ("type", dtype(object))]

    @classmethod
    def get_result_table(cls, net):
        """

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
