# Copyright (c) 2020-2026 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np

from pandapipes.component_models.abstract_models.branch_models import BranchComponent
from pandapipes.component_models.component_toolbox import build_pit_entries
from pandapipes.idx_branch import IdxBranch
from pandapipes.pf.pipeflow_setup import add_table_lookup, get_net_option, get_lookup
from pandapipes.pf.system_index import PitEntries

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


class BranchWOInternalsComponent(BranchComponent):

    @classmethod
    def table_name(cls):
        raise NotImplementedError

    @classmethod
    def active_identifier(cls):
        raise NotImplementedError

    @classmethod
    def get_connected_node_type(cls):
        raise NotImplementedError

    @classmethod
    def from_to_node_cols(cls):
        raise NotImplementedError

    @classmethod
    def get_component_input(cls):
        raise NotImplementedError

    @classmethod
    def create_branch_lookups(cls, net, ft_lookups, table_lookup, idx_lookups, current_start, current_table, internals):
        """Function which creates branch lookups.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param ft_lookups:
        :type ft_lookups:
        :param table_lookup:
        :type table_lookup:
        :param idx_lookups:
        :type idx_lookups:
        :param current_table:
        :type current_table:
        :param current_start:
        :type current_start:
        :param internals:
        :type internals:
        :return:
        :rtype:
        """
        end = current_start + len(net[cls.table_name()])
        ft_lookups[cls.table_name()] = (current_start, end)
        add_table_lookup(table_lookup, cls.table_name(), current_table)
        return end, current_table + 1

    @classmethod
    def register_pit_branch_entries(cls, net, branch_pit, node_pit, registry) -> None:
        super().register_pit_branch_entries(net, branch_pit, node_pit, registry)

        f, t = get_lookup(net, "branch", "from_to")[cls.table_name()]
        tbl = net[cls.table_name()]
        if not len(tbl):
            return

        rows = np.arange(f, t, dtype=np.int32)
        junction_table_name = cls.get_connected_node_type().table_name()
        junction_idx_lookup = get_lookup(net, "node", "index")[junction_table_name]
        fn_col, tn_col = cls.from_to_node_cols()
        from_junctions = tbl[fn_col].values
        to_junctions = tbl[tn_col].values
        from_nodes = junction_idx_lookup[from_junctions]
        to_nodes = junction_idx_lookup[to_junctions]

        if not get_net_option(net, "transient") or get_net_option(net, "simulation_time_step") == 0:
            toutinit_vals = cls._toutinit_vals(net, to_junctions, junction_table_name)
            ambient_t = get_net_option(net, 'ambient_temperature')
            d_val = 0.1
            registry.add(PitEntries(*build_pit_entries(
                rows,
                [IdxBranch.FROM_NODE, IdxBranch.TO_NODE, IdxBranch.TOUTINIT, IdxBranch.ELEMENT_IDX,
                 IdxBranch.ACTIVE, IdxBranch.LENGTH, IdxBranch.K, IdxBranch.TEXT, IdxBranch.ALPHA,
                 IdxBranch.D, IdxBranch.DO],
                [from_nodes.astype(float), to_nodes.astype(float), toutinit_vals,
                 tbl.index.values.astype(float), tbl[cls.active_identifier()].values.astype(float),
                 0., 1e-3, float(ambient_t), 0., d_val, d_val],
            )))

    @classmethod
    def _toutinit_vals(cls, net, to_junctions, junction_table_name):
        return net[junction_table_name].loc[to_junctions, "tfluid_k"].values

    @classmethod
    def calculate_temperature_lift(cls, net, branch_component_pit, node_pit):
        raise NotImplementedError

    @classmethod
    def extract_results(cls, net, options, branch_results, mode):
        raise NotImplementedError

    @classmethod
    def get_result_table(cls, net):
        raise NotImplementedError
