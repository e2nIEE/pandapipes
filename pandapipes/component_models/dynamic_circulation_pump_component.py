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
from pandapipes.idx_node import PINIT, PAMB, TINIT as TINIT_NODE, HEIGHT
from pandapipes.constants import NORMAL_TEMPERATURE, NORMAL_PRESSURE, P_CONVERSION, GRAVITATION_CONSTANT
from pandapipes.properties.fluids import get_fluid
from pandapipes.component_models.component_toolbox import p_correction_height_air

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


class DynamicCirculationPump(CirculationPump):

    # class attributes
    prev_mvlag = 0
    kwargs = None
    prev_act_pos = 0
    time_step = 0
    sink_index_p= None
    source_index_p = None

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
    def plant_dynamics(cls, dt, desired_mv):
        """
        Takes in the desired valve position (MV value) and computes the actual output depending on
        equipment lag parameters.
        Returns Actual valve position
        """

        if cls.kwargs.__contains__("act_dynamics"):
            typ = cls.kwargs['act_dynamics']
        else:
            # default to instantaneous
            return desired_mv

        # linear
        if typ == "l":

            # TODO: equation for linear
            actual_pos = desired_mv

        # first order
        elif typ == "fo":

            a = np.divide(dt, cls.kwargs['time_const_s'] + dt)
            actual_pos = (1 - a) * cls.prev_act_pos + a * desired_mv

            cls.prev_act_pos = actual_pos

        # second order
        elif typ == "so":
            # TODO: equation for second order
            actual_pos = desired_mv

        else:
            # instantaneous - when incorrect option selected
            actual_pos = desired_mv

        return actual_pos


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
        p_in = dyn_circ_pump.p_static_circuit.values
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
        dyn_circ_pump_pit[:, ACTIVE] = True
        dyn_circ_pump_pit[:, LC] = 0
        dyn_circ_pump_pit[:, ACTUAL_POS] = net[cls.table_name()].actual_pos.values
        dyn_circ_pump_pit[:, DESIRED_MV] = net[cls.table_name()].desired_mv.values

        std_types_lookup = np.array(list(net.std_types['dynamic_pump'].keys()))
        std_type, pos = np.where(net[cls.table_name()]['std_type'].values
                                 == std_types_lookup[:, np.newaxis])
        dyn_circ_pump_pit[pos, STD_TYPE] = std_type
        dyn_circ_pump_pit[:, VINIT] = 0#.1 #0.1

    @classmethod
    def adaption_before_derivatives_hydraulic(cls, net, branch_pit, node_pit, idx_lookups, options):

        # calculation of pressure lift
        f, t = idx_lookups[cls.table_name()]
        dyn_circ_pump_pit = branch_pit[f:t, :]
        dt = options['dt']
        area = dyn_circ_pump_pit[:, AREA]
        idx = dyn_circ_pump_pit[:, STD_TYPE].astype(int)
        std_types = np.array(list(net.std_types['dynamic_pump'].keys()))[idx]
        from_nodes = dyn_circ_pump_pit[:, FROM_NODE].astype(np.int32)
        to_nodes = dyn_circ_pump_pit[:, TO_NODE].astype(np.int32)
        fluid = get_fluid(net)
        p_from = node_pit[from_nodes, PAMB] + node_pit[from_nodes, PINIT]
        p_to = node_pit[to_nodes, PAMB] + node_pit[to_nodes, PINIT]
        numerator = NORMAL_PRESSURE * dyn_circ_pump_pit[:, TINIT]
        v_mps = dyn_circ_pump_pit[:, VINIT]
        desired_mv = dyn_circ_pump_pit[:, DESIRED_MV]

        if fluid.is_gas:
            # consider volume flow at inlet
            normfactor_from = numerator * fluid.get_property("compressibility", p_from) \
                              / (p_from * NORMAL_TEMPERATURE)
            v_mean = v_mps * normfactor_from
        else:
            v_mean = v_mps
        vol = v_mean * area

        if not np.isnan(desired_mv) and get_net_option(net, "time_step") == cls.time_step:
            # a controller timeseries is running
            actual_pos = cls.plant_dynamics(dt, desired_mv)
            dyn_circ_pump_pit[:, ACTUAL_POS] = actual_pos
            cls.time_step+= 1

        else: # Steady state analysis
            actual_pos = dyn_circ_pump_pit[:, ACTUAL_POS]

        fcts = itemgetter(*std_types)(net['std_types']['dynamic_pump'])
        fcts = [fcts] if not isinstance(fcts, tuple) else fcts

        hl = np.array(list(map(lambda x, y, z: x.get_m_head(y, z), fcts, vol, actual_pos))) # m head
        pl = np.divide((dyn_circ_pump_pit[:, RHO] * GRAVITATION_CONSTANT * hl), P_CONVERSION)[0] # bar



        # Now: Update the Discharge pressure node (Also known as the starting PT node)
        # And the discharge temperature from the suction temperature (neglecting pump temp)
        circ_pump_tbl = net[cls.table_name()][net[cls.table_name()][cls.active_identifier()].values]

        dyn_circ_pump_pit[:, PL] = pl # -(pl - circ_pump_tbl.p_static_circuit)

        junction = net[cls.table_name()][cls.from_to_node_cols()[1]].values

        # TODO: there should be a warning, if any p_bar value is not given or any of the types does
        #       not contain "p", as this should not be allowed for this component

        t_flow_k = node_pit[from_nodes, TINIT_NODE]
        p_static = node_pit[from_nodes, PINIT]

        # update the 'FROM' node i.e: discharge node
        update_fixed_node_entries(net, node_pit, junction, circ_pump_tbl.type.values, pl + p_static,
                                  t_flow_k, cls.get_connected_node_type())

    @classmethod
    def adaption_after_derivatives_hydraulic(cls, net, branch_pit, node_pit, idx_lookups, options):
        # set all PC branches to derivatives to 0
        f, t = idx_lookups[cls.table_name()]
        dyn_circ_pump_pit = branch_pit[f:t, :]
        v_mps = dyn_circ_pump_pit[:, VINIT]
        area = dyn_circ_pump_pit[:, AREA]
        # function at 100% speed hardcoded
        P_const = np.divide((dyn_circ_pump_pit[:, RHO] * GRAVITATION_CONSTANT), P_CONVERSION)[0]
        df_dv = - P_const * (2 * v_mps * -1.2028 * area**2 + 0.2417 * area)
        dyn_circ_pump_pit[:, JAC_DERIV_DV] = df_dv

        from_nodes = dyn_circ_pump_pit[:, FROM_NODE].astype(np.int32)
        to_nodes = dyn_circ_pump_pit[:, TO_NODE].astype(np.int32)
        p_from = node_pit[from_nodes, PAMB] + node_pit[from_nodes, PINIT]
        p_to = node_pit[to_nodes, PAMB] + node_pit[to_nodes, PINIT]
        pl = P_const * (-1.2028 * v_mps**2 ** area**2 + 0.2417 * v_mps * area + 49.252)
        pl = dyn_circ_pump_pit[:, PL]
        load_vec = p_to - p_from - pl
        dyn_circ_pump_pit[:, LOAD_VEC_BRANCHES] = load_vec

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
                ("p_static_circuit", "f8"),
                ("actual_pos", "f8"),
                ("in_service", 'bool'),
                ("std_type", dtype(object)),
                ("type", dtype(object))]

    @classmethod
    def calculate_temperature_lift(cls, net, pipe_pit, node_pit):
        pass
