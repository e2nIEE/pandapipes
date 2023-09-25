# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

from pandapipes.component_models.junction_component import Junction
from pandapipes.component_models.abstract_models.const_flow_models import ConstFlow
from pandapipes.pf.pipeflow_setup import get_lookup


class Sink(ConstFlow):
    """

    """

    @classmethod
    def table_name(cls):
        return "sink"

    @classmethod
    def sign(cls):
        return 1

    @classmethod
    def get_connected_node_type(cls):
        return Junction

    @classmethod
    def active_identifier(cls):
        return "in_service"

    @classmethod
    def node_element_relevant(cls, net):
        return False

    @classmethod
    def create_pit_node_entries(cls, net, node_pit):
        super().create_pit_node_entries(net, node_pit)
        fluids = net._fluid
        if len(fluids) != 1:
            sinks = net[cls.table_name()]
            juncts = sinks.loc[sinks.mdot_kg_per_s.values == 0, 'junction']
            junction_idx_lookups = get_lookup(net, "node", "index")[cls.get_connected_node_type().table_name()]
            index = junction_idx_lookups[juncts]
            node_pit[index, net['_idx_node']['LOAD']] += 0#10 ** -16