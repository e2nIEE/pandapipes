# Copyright (c) 2020-2022 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
from numpy import dtype
from operator import itemgetter
from pandapipes.component_models.junction_component import Junction
from pandapipes.component_models.abstract_models.circulation_pump import CirculationPump
from pandapipes.component_models.component_toolbox import set_fixed_node_entries, update_fixed_node_entries
from pandapipes.idx_node import PINIT, NODE_TYPE, P, EXT_GRID_OCCURENCE
from pandapipes.pf.pipeflow_setup import get_lookup, get_net_option
from pandapipes.idx_branch import STD_TYPE, VINIT, D, AREA, ACTIVE, LOSS_COEFFICIENT as LC, FROM_NODE, \
    TINIT, PL, ACTUAL_POS, DESIRED_MV, RHO, TO_NODE, JAC_DERIV_DP, JAC_DERIV_DP1, JAC_DERIV_DV, LOAD_VEC_BRANCHES
from pandapipes.idx_node import PINIT, PAMB, TINIT as TINIT_NODE, HEIGHT, RHO as RHO_node
from pandapipes.constants import NORMAL_TEMPERATURE, NORMAL_PRESSURE, P_CONVERSION, GRAVITATION_CONSTANT
from pandapipes.properties.fluids import get_fluid
from pandapipes.component_models.component_toolbox import p_correction_height_air
from pandapipes.component_models.component_toolbox import set_fixed_node_entries, \
    get_mass_flow_at_nodes
from pandapipes.pf.result_extraction import extract_branch_results_without_internals

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


class DynamicCirculationPump(CirculationPump):

    # class attributes
    kwargs = None
    prev_act_pos = None
    time_step = 0

    @classmethod
    def table_name(cls):
        return "dyn_circ_pump"

    @classmethod
    def get_connected_node_type(cls):
        return Junction

    @classmethod
    def active_identifier(cls):
        return "in_service"

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
        # Sets the discharge pressure, otherwise known as the starting node in the system
        dyn_circ_pump, press = super().create_pit_node_entries(net, node_pit)

        # SET SUCTION PRESSURE
        junction = dyn_circ_pump[cls.from_to_node_cols()[0]].values
        p_in = dyn_circ_pump.p_static_bar.values
        set_fixed_node_entries(net, node_pit, junction, dyn_circ_pump.type.values, p_in, None,
                               cls.get_connected_node_type(), "p")


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
        dyn_circ_pump_pit = super().create_pit_branch_entries(net, branch_pit)
        dyn_circ_pump_pit[:, ACTIVE] = False

    @classmethod
    def plant_dynamics(cls, dt, desired_mv, dyn_pump_tbl):
        """
        Takes in the desired valve position (MV value) and computes the actual output depending on
        equipment lag parameters.
        Returns Actual valve position
        """

        """
        Takes in the desired valve position (MV value) and computes the actual output depending on
        equipment lag parameters.
        Returns Actual valve position
        """

        if dyn_pump_tbl.__contains__("time_const_s"):
            time_const_s = dyn_pump_tbl.time_const_s.values
        else:
            print("No actuator time constant set, default lag is now 5s.")
            time_const_s = 5

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
        circ_pump_tbl = net[cls.table_name()]
        junction_lookup = get_lookup(net, "node", "index")[ cls.get_connected_node_type().table_name()]
        fn_col, tn_col = cls.from_to_node_cols()
        # get indices in internal structure for flow_junctions in circ_pump tables which are
        # "active"
        return_junctions = circ_pump_tbl[fn_col].values
        return_node = junction_lookup[return_junctions]
        rho = node_pit[return_node, RHO_node]
        flow_junctions = circ_pump_tbl[tn_col].values
        flow_nodes = junction_lookup[flow_junctions]
        in_service = circ_pump_tbl.in_service.values
        p_grids = np.isin(circ_pump_tbl.type.values, ["p", "pt"]) & in_service
        sum_mass_flows, inverse_nodes, counts = get_mass_flow_at_nodes(net, node_pit, branch_pit,
                                                                       flow_nodes[p_grids], cls)
        q_kg_s = - (sum_mass_flows / counts)[inverse_nodes]
        vol_m3_s = np.divide(q_kg_s, rho)
        vol_m3_h = vol_m3_s * 3600
        desired_mv = circ_pump_tbl.desired_mv.values
        cur_actual_pos = circ_pump_tbl.actual_pos.values

        #if not np.isnan(desired_mv) and get_net_option(net, "time_step") == cls.time_step:
        if get_net_option(net, "time_step") == cls.time_step:
            # a controller timeseries is running
            actual_pos = cls.plant_dynamics(dt, desired_mv, circ_pump_tbl)
            # Account for nan's when FCE are in manual
            update_pos = np.where(np.isnan(actual_pos))
            actual_pos[update_pos] = cur_actual_pos[update_pos]
            circ_pump_tbl.actual_pos = actual_pos
            cls.time_step += 1

        else: # Steady state analysis
            actual_pos = circ_pump_tbl.actual_pos.values

        std_types_lookup = np.array(list(net.std_types['dynamic_pump'].keys()))
        std_type, pos = np.where(net[cls.table_name()]['std_type'].values
                                 == std_types_lookup[:, np.newaxis])
        std_types = np.array(list(net.std_types['dynamic_pump'].keys()))[std_type]
        fcts = itemgetter(*std_types)(net['std_types']['dynamic_pump'])
        fcts = [fcts] if not isinstance(fcts, tuple) else fcts
        m_head = np.array(list(map(lambda x, y, z: x.get_m_head(y, z), fcts, vol_m3_s, actual_pos))) # m head
        prsr_lift = np.divide((rho * GRAVITATION_CONSTANT * m_head), P_CONVERSION)[0] # bar
        circ_pump_tbl.p_lift = prsr_lift
        circ_pump_tbl.m_head = m_head

        # Now: Update the Discharge pressure node (Also known as the starting PT node)
        # And the discharge temperature from the suction temperature (neglecting pump temp)
        circ_pump_tbl = net[cls.table_name()][net[cls.table_name()][cls.active_identifier()].values]

        junction = net[cls.table_name()][cls.from_to_node_cols()[1]].values

        # TODO: there should be a warning, if any p_bar value is not given or any of the types does
        #       not contain "p", as this should not be allowed for this component

        t_flow_k = node_pit[return_node, TINIT_NODE]
        p_static = circ_pump_tbl.p_static_bar.values

        # update the 'FROM' node i.e: discharge node temperature and pressure lift updates
        update_fixed_node_entries(net, node_pit, junction, circ_pump_tbl.type.values, (prsr_lift + p_static), t_flow_k,
                                  cls.get_connected_node_type(), "pt")


    @classmethod
    def get_result_table(cls, net):
        """

        :param net: The pandapipes network
        :type net: pandapipesNet
        :return: (columns, all_float) - the column names and whether they are all float type. Only
                if False, returns columns as tuples also specifying the dtypes
        :rtype: (list, bool)
        """
        return ["mdot_flow_kg_per_s", "deltap_bar", "desired_mv", "actual_pos", "p_lift", "m_head", "rho", "t_from_k",
                "t_to_k", "p_static_bar", "p_flow_bar"], True

    @classmethod
    def get_component_input(cls):
        """

        :return:
        :rtype:
        """
        return [("name", dtype(object)),
                ("return_junction", "u4"),
                ("flow_junction", "u4"),
                ("p_flow_bar", "f8"),
                ("t_flow_k", "f8"),
                ("p_lift", "f8"),
                ('m_head', "f8"),
                ("p_static_bar", "f8"),
                ("actual_pos", "f8"),
                ("in_service", 'bool'),
                ("std_type", dtype(object)),
                ("type", dtype(object))]

    @classmethod
    def calculate_temperature_lift(cls, net, pipe_pit, node_pit):
        pass


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
        circ_pump_tbl = net[cls.table_name()]

        if len(circ_pump_tbl) == 0:
            return

        res_table = net["res_" + cls.table_name()]

        branch_pit = net['_pit']['branch']
        node_pit = net["_pit"]["node"]

        junction_lookup = get_lookup(net, "node", "index")[
            cls.get_connected_node_type().table_name()]
        fn_col, tn_col = cls.from_to_node_cols()
        # get indices in internal structure for flow_junctions in circ_pump tables which are
        # "active"
        flow_junctions = circ_pump_tbl[tn_col].values
        flow_nodes = junction_lookup[flow_junctions]
        in_service = circ_pump_tbl.in_service.values
        p_grids = np.isin(circ_pump_tbl.type.values, ["p", "pt"]) & in_service
        sum_mass_flows, inverse_nodes, counts = get_mass_flow_at_nodes(net, node_pit, branch_pit,
                                                                       flow_nodes[p_grids], cls)

        # positive results mean that the circ_pump feeds in, negative means that the ext grid
        # extracts (like a load)
        res_table["mdot_flow_kg_per_s"].values[p_grids] = - (sum_mass_flows / counts)[inverse_nodes]

        return_junctions = circ_pump_tbl[fn_col].values
        return_node = junction_lookup[return_junctions]


        #res_table["vdot_norm_m3_per_s"] = np.divide(- (sum_mass_flows / counts)[inverse_nodes], rho)

        return_junctions = circ_pump_tbl[fn_col].values
        return_nodes = junction_lookup[return_junctions]

        deltap_bar = node_pit[flow_nodes, PINIT] - node_pit[return_nodes, PINIT]
        res_table["p_static_bar"].values[in_service] = circ_pump_tbl.p_static_bar.values
        res_table["p_flow_bar"].values[in_service] = node_pit[flow_nodes, PINIT]
        res_table["deltap_bar"].values[in_service] = deltap_bar[in_service]
        res_table["t_from_k"].values[p_grids] = node_pit[return_node, TINIT]
        res_table["t_to_k"].values[p_grids] = node_pit[flow_nodes, TINIT]
        res_table["rho"].values[p_grids] = node_pit[return_node, RHO_node]
        res_table["p_lift"].values[p_grids] = circ_pump_tbl.p_lift.values
        res_table["m_head"].values[p_grids] = circ_pump_tbl.m_head.values
        res_table["actual_pos"].values[p_grids] = circ_pump_tbl.actual_pos.values
        res_table["desired_mv"].values[p_grids] = circ_pump_tbl.desired_mv.values
