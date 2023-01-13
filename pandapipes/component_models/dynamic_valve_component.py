# Copyright (c) 2020-2022 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
from numpy import dtype
from operator import itemgetter

from pandapipes.component_models.abstract_models.branch_wzerolength_models import \
    BranchWZeroLengthComponent
from pandapipes.component_models.junction_component import Junction
from pandapipes.idx_node import PINIT, PAMB, TINIT as TINIT_NODE, NODE_TYPE, P, ACTIVE as ACTIVE_ND, LOAD
from pandapipes.idx_branch import D, AREA, TL, KV, ACTUAL_POS, STD_TYPE, R_TD, FROM_NODE, TO_NODE, \
    VINIT, RHO, PL, LOSS_COEFFICIENT as LC, DESIRED_MV, ACTIVE
from pandapipes.pf.result_extraction import extract_branch_results_without_internals
from pandapipes.properties.fluids import get_fluid


class DynamicValve(BranchWZeroLengthComponent):
    """
    Dynamic Valves are branch elements that can separate two junctions.
    They have a length of 0, but can introduce a lumped pressure loss.
    The equation is based on the standard valve dynamics: q = Kv(h) * sqrt(Delta_P).
    """
    fcts = None

    @classmethod
    def set_function(cls, net):
        std_types_lookup = np.array(list(net.std_types[cls.table_name()].keys()))
        std_type, pos = np.where(net[cls.table_name()]['std_type'].values
                                 == std_types_lookup[:, np.newaxis])
        std_types = np.array(list(net.std_types['dynamic_valve'].keys()))[pos]
        fcts = itemgetter(*std_types)(net['std_types']['dynamic_valve'])
        cls.fcts = [fcts] if not isinstance(fcts, tuple) else fcts

    @classmethod
    def from_to_node_cols(cls):
        return "from_junction", "to_junction"

    @classmethod
    def table_name(cls):
        return "dynamic_valve"

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
        valve_grids = net[cls.table_name()]
        valve_pit = super().create_pit_branch_entries(net, branch_pit)
        valve_pit[:, D] = net[cls.table_name()].diameter_m.values
        valve_pit[:, AREA] = valve_pit[:, D] ** 2 * np.pi / 4
        valve_pit[:, KV] = net[cls.table_name()].Kv.values
        valve_pit[:, ACTUAL_POS] = net[cls.table_name()].actual_pos.values
        valve_pit[:, DESIRED_MV] = net[cls.table_name()].desired_mv.values
        valve_pit[:, R_TD] = net[cls.table_name()].r_td.values

        # Update in_service status if valve actual position becomes 0%
        if valve_pit[:, ACTUAL_POS] > 0:
            valve_pit[:, ACTIVE] = True
        else:
            valve_pit[:, ACTIVE] = False

        # TODO: is this std_types necessary here when we have already set the look up function?
        std_types_lookup = np.array(list(net.std_types[cls.table_name()].keys()))
        std_type, pos = np.where(net[cls.table_name()]['std_type'].values
                                 == std_types_lookup[:, np.newaxis])
        valve_pit[pos, STD_TYPE] = std_type

    @classmethod
    def adaption_before_derivatives_hydraulic(cls, net, branch_pit, node_pit, idx_lookups, options):
        # calculation of valve flow via

        #  we need to know which element is fixed, i.e Dynamic_Valve P_out, Dynamic_Valve P_in, Dynamic_Valve Flow_rate
        f, t = idx_lookups[cls.table_name()]
        valve_pit = branch_pit[f:t, :]
        area = valve_pit[:, AREA]
        #idx = valve_pit[:, STD_TYPE].astype(int)
        #std_types = np.array(list(net.std_types['dynamic_valve'].keys()))[idx]
        from_nodes = valve_pit[:, FROM_NODE].astype(np.int32)
        to_nodes = valve_pit[:, TO_NODE].astype(np.int32)
        p_from = node_pit[from_nodes, PAMB] + node_pit[from_nodes, PINIT]
        p_to = node_pit[to_nodes, PAMB] + node_pit[to_nodes, PINIT]

        # look up travel (TODO: maybe a quick check if travel has changed instead of calling poly function each cycle?
        #fcts = itemgetter(*std_types)(net['std_types']['dynamic_valve'])
        #fcts = [fcts] if not isinstance(fcts, tuple) else fcts

        lift = np.divide(valve_pit[:, ACTUAL_POS], 100)
        relative_flow = np.array(list(map(lambda x, y: x.get_relative_flow(y), cls.fcts, lift)))

        kv_at_travel = relative_flow * valve_pit[:, KV] # m3/h.Bar
        delta_p = p_from - p_to  # bar
        q_m3_h = kv_at_travel * np.sqrt(delta_p)
        q_m3_s = np.divide(q_m3_h, 3600)
        v_mps = np.divide(q_m3_s, area)
        #v_mps = valve_pit[:, VINIT]
        rho = valve_pit[:, RHO]
        #rho = 1004
        zeta = np.divide(q_m3_h**2 * 2 * 100000, kv_at_travel**2 * rho * v_mps**2)
        valve_pit[:, LC] = zeta
        #node_pit[:, LOAD] = q_m3_s * RHO # mdot_kg_per_s # we can not change the velocity nor load here!!
        #valve_pit[:, VINIT] = v_mps

    @classmethod
    def calculate_temperature_lift(cls, net, valve_pit, node_pit):
        """

        :param net:
        :type net:
        :param valve_pit:
        :type valve_pit:
        :param node_pit:
        :type node_pit:
        :return:
        :rtype:
        """
        valve_pit[:, TL] = 0

    @classmethod
    def get_component_input(cls):
        """

        :return:
        :rtype:
        """
        return [("name", dtype(object)),
                ("from_junction", "i8"),
                ("to_junction", "i8"),
                ("diameter_m", "f8"),
                ("actual_pos", "f8"),
                ("std_type", dtype(object)),
                ("Kv", "f8"),
                ("r_td", "f8"),
                ("type", dtype(object))]

    @classmethod
    def extract_results(cls, net, options, branch_results, nodes_connected, branches_connected):
        required_results = [
            ("p_from_bar", "p_from"), ("p_to_bar", "p_to"), ("t_from_k", "temp_from"),
            ("t_to_k", "temp_to"), ("mdot_to_kg_per_s", "mf_to"), ("mdot_from_kg_per_s", "mf_from"),
            ("vdot_norm_m3_per_s", "vf"), ("lambda", "lambda"), ("reynolds", "reynolds"), ("desired_mv", "desired_mv"),
            ("actual_pos", "actual_pos")
        ]

        if get_fluid(net).is_gas:
            required_results.extend([
                ("v_from_m_per_s", "v_gas_from"), ("v_to_m_per_s", "v_gas_to"),
                ("v_mean_m_per_s", "v_gas_mean"), ("normfactor_from", "normfactor_from"),
                ("normfactor_to", "normfactor_to")
            ])
        else:
            required_results.extend([("v_mean_m_per_s", "v_mps")])

        extract_branch_results_without_internals(net, branch_results, required_results,
                                                 cls.table_name(), branches_connected)

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
                      "normfactor_to", "desired_mv", "actual_pos"]
        else:
            output = ["v_mean_m_per_s", "p_from_bar", "p_to_bar", "t_from_k", "t_to_k",
                      "mdot_from_kg_per_s", "mdot_to_kg_per_s", "vdot_norm_m3_per_s", "reynolds",
                      "lambda", "desired_mv", "actual_pos"]
        return output, True
