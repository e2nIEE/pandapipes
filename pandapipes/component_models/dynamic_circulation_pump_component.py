# Copyright (c) 2020-2022 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
from numpy import dtype
from operator import itemgetter
from pandapipes.component_models.junction_component import Junction
from pandapipes.component_models.abstract_models.circulation_pump import CirculationPump
from pandapipes.idx_node import PINIT, NODE_TYPE, P, EXT_GRID_OCCURENCE
from pandapipes.pf.internals_toolbox import _sum_by_group
from pandapipes.pf.pipeflow_setup import get_lookup, get_net_option
from pandapipes.idx_branch import STD_TYPE, VINIT, D, AREA, LOSS_COEFFICIENT as LC, FROM_NODE, \
    TINIT, PL, ACTUAL_POS, DESIRED_MV, RHO
from pandapipes.idx_node import PINIT, PAMB, HEIGHT
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
    fcts = None
    prev_mvlag = 0
    kwargs = None
    prev_act_pos = 0
    time_step = 0
    sink_index_p= None
    source_index_p = None

    @classmethod
    def set_function(cls, net, actual_pos, **kwargs):
        std_types_lookup = np.array(list(net.std_types['dynamic_pump'].keys()))
        std_type, pos = np.where(net[cls.table_name()]['std_type'].values
                                 == std_types_lookup[:, np.newaxis])
        std_types = np.array(list(net.std_types['dynamic_pump'].keys()))[pos]
        fcts = itemgetter(*std_types)(net['std_types']['dynamic_pump'])
        cls.fcts = [fcts] if not isinstance(fcts, tuple) else fcts

        # Initial config
        cls.prev_act_pos = actual_pos
        cls.kwargs = kwargs


    @classmethod
    def table_name(cls):
        return "dyn_circ_pump"

    @classmethod
    def get_connected_node_type(cls):
        return Junction

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
        # Sets Source (discharge pressure), temp and junction types
        circ_pump, press = super().create_pit_node_entries(net, node_pit)

        junction_idx_lookups = get_lookup(net, "node", "index")[
            cls.get_connected_node_type().table_name()]

        # Calculates the suction pressure from: (source minus p_lift)  and indicates which junction node to set this value
        juncts_p, press_sum, number = _sum_by_group(get_net_option(net, "use_numba"), circ_pump.to_junction.values,
            p_correction_height_air(node_pit[:, HEIGHT]), np.ones_like(press, dtype=np.int32))

        # Sets sink (suction pressure) pressure and type values
        cls.sink_index_p = junction_idx_lookups[juncts_p]
        node_pit[cls.sink_index_p, PINIT] = press_sum / number
        node_pit[cls.sink_index_p, NODE_TYPE] = P
        node_pit[cls.sink_index_p, EXT_GRID_OCCURENCE] += number

        net["_lookups"]["ext_grid"] = \
            np.array(list(set(np.concatenate([net["_lookups"]["ext_grid"], cls.sink_index_p]))))


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
    def adaption_before_derivatives_hydraulic(cls, net, branch_pit, node_pit, idx_lookups, options):
        # calculation of pressure lift
        f, t = idx_lookups[cls.table_name()]
        pump_pit = branch_pit[f:t, :]
        dt = options['dt']

        desired_mv = net[cls.table_name()].desired_mv.values

        D = 0.1
        area = D ** 2 * np.pi / 4

        from_nodes = pump_pit[:, FROM_NODE].astype(np.int32)
        fluid = get_fluid(net)

        p_from = node_pit[from_nodes, PAMB] + node_pit[from_nodes, PINIT]

        numerator = NORMAL_PRESSURE * pump_pit[:, TINIT]
        v_mps = pump_pit[:, VINIT]

        if not np.isnan(desired_mv) and get_net_option(net, "time_step") == cls.time_step: # a controller timeseries is running
            actual_pos = cls.plant_dynamics(dt, desired_mv)
            valve_pit[:, ACTUAL_POS] = actual_pos
            cls.time_step+= 1


        else: # Steady state analysis
            actual_pos = valve_pit[:, ACTUAL_POS]


        if fluid.is_gas:
            # consider volume flow at inlet
            normfactor_from = numerator * fluid.get_property("compressibility", p_from) \
                              / (p_from * NORMAL_TEMPERATURE)
            v_mean = v_mps * normfactor_from
        else:
            v_mean = v_mps
        vol = v_mean * area

        speed = net[cls.table_name()].actual_pos.values
        hl = np.array(list(map(lambda x, y, z: x.get_m_head(y, z), cls.fcts, vol, speed)))
        pl = np.divide((pump_pit[:, RHO] * GRAVITATION_CONSTANT * hl), P_CONVERSION)  # bar


        ##### Add Pressure Lift To Extgrid Node  ########
        ext_grids = net[cls.table_name()]
        ext_grids = ext_grids[ext_grids.in_service.values]

        p_mask = np.where(np.isin(ext_grids.type.values, ["p", "pt"]))
        press = pl #ext_grids.p_bar.values[p_mask]
        junction_idx_lookups = get_lookup(net, "node", "index")[
            cls.get_connected_node_type().table_name()]
        junction = cls.get_connected_junction(net)
        juncts_p, press_sum, number = _sum_by_group(
            get_net_option(net, "use_numba"), junction.values[p_mask], press,
            np.ones_like(press, dtype=np.int32))
        index_p = junction_idx_lookups[juncts_p]

        node_pit[index_p, PINIT] = press_sum / number
        #node_pit[index_p, NODE_TYPE] = P
        #node_pit[index_p, EXT_GRID_OCCURENCE] += number


    @classmethod
    def get_component_input(cls):
        """

        :return:
        :rtype:
        """
        return [("name", dtype(object)),
                ("from_junction", "u4"),
                ("to_junction", "u4"),
                ("p_bar", "f8"),
                ("t_k", "f8"),
                ("plift_bar", "f8"),
                ("actual_pos", "f8"),
                ("std_type", dtype(object)),
                ("in_service", 'bool'),
                ("type", dtype(object))]
