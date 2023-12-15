# Copyright (c) 2020-2022 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
from numpy import dtype
from operator import itemgetter
from pandapipes.constants import NORMAL_TEMPERATURE, NORMAL_PRESSURE, R_UNIVERSAL, P_CONVERSION
from pandapipes.component_models.abstract_models.branch_wzerolength_models import \
    BranchWZeroLengthComponent
from pandapipes.component_models.junction_component import Junction
from pandapipes.pf.pipeflow_setup import get_net_option, get_net_options
from pandapipes.idx_node import PINIT, PAMB, TINIT as TINIT_NODE, NODE_TYPE, P, ACTIVE as ACTIVE_ND, LOAD
from pandapipes.idx_branch import D, AREA, TL, Kv_max, ACTUAL_POS, STD_TYPE, FROM_NODE, TO_NODE, \
    VINIT, RHO, PL, LOSS_COEFFICIENT as LC, DESIRED_MV, ACTIVE, LOAD_VEC_NODES
from pandapipes.pf.result_extraction import extract_branch_results_without_internals
from pandapipes.properties.fluids import get_fluid


class DynamicValve(BranchWZeroLengthComponent):
    """
    Dynamic Valves are branch elements that can separate two junctions.
    They have a length of 0, but can introduce a lumped pressure loss.
    The equation is based on the standard valve dynamics: q = Kv(h) * sqrt(Delta_P).
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
        valve_pit[:, D] = 0.05 #0.1 #net[cls.table_name()].diameter_m.values
        valve_pit[:, AREA] = valve_pit[:, D] ** 2 * np.pi / 4
        valve_pit[:, Kv_max] = net[cls.table_name()].Kv_max.values
        valve_pit[:, ACTUAL_POS] = net[cls.table_name()].actual_pos.values
        valve_pit[:, DESIRED_MV] = net[cls.table_name()].desired_mv.values


        # # # Update in_service status if valve actual position becomes 0%
        # vlv_status = valve_pit[:, ACTIVE]
        # in_active_pos = np.where(valve_pit[:, ACTUAL_POS] == 0)
        # vlv_status[in_active_pos] = 0
        # valve_pit[:, ACTIVE] = vlv_status

        std_types_lookup = np.array(list(net.std_types[cls.table_name()].keys()))
        std_type, pos = np.where(net[cls.table_name()]['std_type'].values
                                 == std_types_lookup[:, np.newaxis])
        valve_pit[pos, STD_TYPE] = std_type

    @classmethod
    def plant_dynamics(cls, dt, desired_mv, dyn_valve_tbl):
        """
        Takes in the desired valve position (MV value) and computes the actual output depending on
        equipment lag parameters.
        Returns Actual valve position
        """

        time_const_s = dyn_valve_tbl.time_const_s.values
        a = np.divide(dt, time_const_s + dt)
        actual_pos = (1 - a) * cls.prev_act_pos + a * desired_mv
        cls.prev_act_pos = actual_pos

        return actual_pos

        # Issue with getting array values for different types of valves!! Assume all First Order!
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
        dt = net['_options']['dt']
        f, t = idx_lookups[cls.table_name()]
        dyn_valve_tbl = net[cls.table_name()]
        valve_pit = branch_pit[f:t, :]
        area = valve_pit[:, AREA]
        idx = valve_pit[:, STD_TYPE].astype(int)
        std_types = np.array(list(net.std_types['dynamic_valve'].keys()))[idx]
        from_nodes = valve_pit[:, FROM_NODE].astype(np.int32)
        to_nodes = valve_pit[:, TO_NODE].astype(np.int32)
        p_from = node_pit[from_nodes, PAMB] + node_pit[from_nodes, PINIT]
        p_to = node_pit[to_nodes, PAMB] + node_pit[to_nodes, PINIT]
        valve_pit[:, DESIRED_MV] = net[cls.table_name()].desired_mv.values
        desired_mv = valve_pit[:, DESIRED_MV]
        cur_actual_pos = valve_pit[:, ACTUAL_POS]


        if get_net_option(net, "time_step") == cls.time_step:
            # a controller timeseries is running
            actual_pos = cls.plant_dynamics(dt, desired_mv, dyn_valve_tbl)
            # Account for nan's when FCE are in manual
            update_pos = np.where(np.isnan(actual_pos))
            actual_pos[update_pos] = cur_actual_pos[update_pos]
            valve_pit[:, ACTUAL_POS] = actual_pos
            dyn_valve_tbl.actual_pos = actual_pos
            cls.time_step += 1

        else: # Steady state analysis - recycle for Newton-Raphson loop
            actual_pos = valve_pit[:, ACTUAL_POS]

        fcts = itemgetter(*std_types)(net['std_types']['dynamic_valve'])
        fcts = [fcts] if not isinstance(fcts, tuple) else fcts

        lift = np.divide(actual_pos, 100)
        relative_flow = np.array(list(map(lambda x, y: x.get_relative_flow(y), fcts, lift)))

        kv_at_travel = relative_flow * valve_pit[:, Kv_max] # m3/h.Bar

        delta_p = np.abs(p_from - p_to)  # bar
        #if get_net_option(net, "time_step") == None or get_net_option(net, "time_step") == cls.time_step:
            # On first loop initialise delta P to 0.1 if delta is zero
            #delta_p = np.where(delta_p == 0, 0.1, delta_p)
        #delta_p = np.where(np.ma.masked_where(delta_p == 0, lift == 1.0).mask, 0.1, delta_p)
        positions_delta_p = np.where(delta_p == 0.0)
        positions_lift = np.where(lift != 1.0)
        # get common element values (indexes that appear in both position arrays)
        intersect = np.intersect1d(positions_delta_p, positions_lift)
        # Set delta_p equal to 0.1 where lift is not equal 1 and delta-p equals zero
        delta_p[intersect] = 0.1

        q_m3_h = kv_at_travel * np.sqrt(delta_p)
        q_m3_s = np.divide(q_m3_h, 3600)
        v_mps = np.divide(q_m3_s, area)
        rho = valve_pit[:, RHO]
        # only calculate at valid entries of v_mps, else error handling is required
        update_pos = np.where(v_mps != 0)
        valid_zetas = np.divide(q_m3_h[update_pos] ** 2 * 2 * 100000, kv_at_travel[update_pos] ** 2 * \
                                rho[update_pos] * v_mps[update_pos] ** 2)
        zeta = np.zeros_like(v_mps)
        zeta[update_pos] = valid_zetas
        zeta_reset_2 = np.where(lift == 0.0)
        zeta[zeta_reset_2] = 20e+20
        valve_pit[:, LC] = zeta

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
                ("Kv_max", "f8"),
                ("type", dtype(object))]

    @classmethod
    def extract_results(cls, net, options, branch_results, nodes_connected, branches_connected):
        required_results = [
            ("p_from_bar", "p_from"), ("p_to_bar", "p_to"), ("t_from_k", "temp_from"),
            ("t_to_k", "temp_to"), ("mdot_to_kg_per_s", "mf_to"), ("mdot_from_kg_per_s", "mf_from"),
            ("vdot_norm_m3_per_s", "vf"), ("lambda", "lambda"), ("reynolds", "reynolds"), ("desired_mv", "desired_mv"),
            ("actual_pos", "actual_pos"), ("LC", "LC"),
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
                      "lambda", "desired_mv", "actual_pos", "LC"]
        return output, True
