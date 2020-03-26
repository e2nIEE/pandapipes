# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
from pandapipes.component_models.abstract_models import CirculationPump
from pandapipes.idx_node import PINIT, NODE_TYPE, P, EXT_GRID_OCCURENCE
from pandapipes.pipeflow_setup import get_lookup
from numpy import dtype

from pandapipes.toolbox import _sum_by_group

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


class CirculationPumpPressure(CirculationPump):

    @classmethod
    def table_name(cls):
        return "circ_pump_pressure"

    @classmethod
    def create_pit_node_entries(cls, net, node_pit, node_name):
        """
        Function which creates pit node entries.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param node_pit:
        :type node_pit:
        :return: No Output.
        """
        circ_pump, press = super().create_pit_node_entries(net, node_pit, node_name)

        junction_idx_lookups = get_lookup(net, "node", "index")[node_name]
        juncts_p, press_sum, number = _sum_by_group(circ_pump.to_junction.values,
                                                    press - circ_pump.plift_bar.values,
                                                    np.ones_like(press, dtype=np.int32))

        index_p = junction_idx_lookups[juncts_p]
        node_pit[index_p, PINIT] = press_sum / number
        node_pit[index_p, NODE_TYPE] = P
        node_pit[index_p, EXT_GRID_OCCURENCE] += number

        net["_lookups"]["ext_grid"] = \
            np.array(list(set(np.concatenate([net["_lookups"]["ext_grid"], index_p]))))

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
                ("in_service", 'bool'),
                ("type", dtype(object))]
