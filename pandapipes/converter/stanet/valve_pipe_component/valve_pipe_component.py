# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
from numpy import dtype

from pandapipes.component_models.junction_component import Junction
from pandapipes.component_models.pipe_component import Pipe
from pandapipes.idx_branch import LENGTH, K, D, AREA, LOSS_COEFFICIENT as LC
from pandapipes.properties.fluids import get_fluid


class ValvePipe(Pipe):
    @classmethod
    def from_to_node_cols(cls):
        return "from_junction", "to_junction"

    @classmethod
    def table_name(cls):
        return "valve_pipe"

    @classmethod
    def internal_node_name(cls):
        return "valve_pipe_nodes"

    @classmethod
    def active_identifier(cls):
        return "opened"

    @classmethod
    def get_connected_node_type(cls):
        return Junction

    @classmethod
    def create_pit_branch_entries_table_specific(cls, net, comp_pit, internal_pipe_number):
        """

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param comp_pit:
        :type comp_pit:
        :param internal_pipe_number:
        :type internal_pipe_number:
        :return:
        :rtype:
        """
        comp_pit[:, LENGTH] = np.repeat(net[cls.table_name].length_km.values * 1000 /
                                        internal_pipe_number, internal_pipe_number)
        comp_pit[:, K] = np.repeat(net[cls.table_name].k_mm.values / 1000,
                                   internal_pipe_number)
        comp_pit[:, D] = np.repeat(net[cls.table_name].diameter_m.values, internal_pipe_number)
        comp_pit[:, AREA] = comp_pit[:, D] ** 2 * np.pi / 4
        comp_pit[:, LC] = np.repeat(net[cls.table_name].loss_coefficient.values,
                                    internal_pipe_number)

    @classmethod
    def get_component_input(cls):
        return [("name", dtype(object)),
                ("from_junction", "u4"),
                ("to_junction", "u4"),
                ("std_type", dtype(object)),
                ("length_km", "f8"),
                ("diameter_m", "f8"),
                ("k_mm", "f8"),
                ("opened", "bool"),
                ("loss_coefficient", "f8"),
                ("sections", "u4"),
                ("max_vdot_m3_per_s", 'f8'),
                ("max_v_m_per_s", 'f8'),
                ("in_service", 'bool'),
                ("alpha_w_per_m2k", 'f8'),
                ("qext_w", 'f8'),
                ("type", dtype(object)),
                ('index', 'u4')]

    @classmethod
    def geodata(cls):
        """

        :return:
        :rtype:
        """
        return [("coords", dtype(object))]

    @classmethod
    def get_result_table(cls, net):
        if get_fluid(net).is_gas:
            output = ["v_from_m_per_s",
                      "v_to_m_per_s",
                      "v_mean_m_per_s",
                      "p_from_bar",
                      "p_to_bar",
                      "t_from_k",
                      "t_to_k",
                      "p_from_mw",
                      "p_to_mw",
                      "mdot_from_kg_per_s",
                      "mdot_to_kg_per_s",
                      "vdot_norm_m3_per_s",
                      "reynolds",
                      "lambda",
                      "normfactor_from",
                      "normfactor_to",
                      "loading_percent"]
        else:

            output = ["v_mean_m_per_s",
                      "p_from_bar",
                      "p_to_bar",
                      "t_from_k",
                      "t_to_k",
                      "p_from_mw",
                      "p_to_mw",
                      "mdot_from_kg_per_s",
                      "mdot_to_kg_per_s",
                      "vdot_norm_m3_per_s",
                      "reynolds",
                      "lambda",
                      "loading_percent"]
        return output, True
