# Copyright (c) 2016-2022 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np
import pandas as pd
from pandapower.auxiliary import get_free_id, write_to_net

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


class CollectorController:
    """
    Class for logic selector controllers, for use with multiple external reset PID controllers

    """

    controller_mv_table = pd.DataFrame(data=[], columns=['fc_element', 'fc_index', 'fc_variable',
                                                         'ctrl_values', 'logic_typ', 'write_flag'])
    collect_ctrl_active = None

    @classmethod
    def write_to_ctrl_collector(cls, net, ctrl_element, ctrl_index, ctrl_variable, ctrl_values, logic_typ, write_flag):
        """
        Collector table for controllers on the same level and order.
        Common final control elements are collected are held until all controllers are finalised.
        """

        if cls.controller_mv_table[(cls.controller_mv_table['fc_element'] == ctrl_element) &
                                   (cls.controller_mv_table['fc_index'] == ctrl_index.item()) &
                                   (cls.controller_mv_table['fc_variable'] == ctrl_variable)].empty:

            idx = get_free_id(cls.controller_mv_table)
            cls.controller_mv_table.loc[idx] = \
                [ctrl_element, ctrl_index, ctrl_variable, [ctrl_values.item()], [logic_typ], [write_flag]]

            cls.collect_ctrl_active = True

        else:
            r_idx = int(cls.controller_mv_table[(cls.controller_mv_table.fc_element == ctrl_element) &
                                                (cls.controller_mv_table.fc_index == ctrl_index) &
                                                (cls.controller_mv_table.fc_variable == ctrl_variable)].index.values)

            cls.controller_mv_table.loc[r_idx].ctrl_values.append(ctrl_values.item())
            cls.controller_mv_table.loc[r_idx].logic_typ.append(logic_typ)
            cls.controller_mv_table.loc[r_idx].write_flag.append(write_flag)

    @classmethod
    def consolidate_logic(cls, net):
        """
        Combines controllers that are on the same level and order, so that controller logic type can be evaluated on all
        common FCE elements. The final desired MV value is written to the FCE element.
        """

        for fc_element, fc_index, fc_variable, ctrl_values, logic_typ, write_flag in cls.controller_mv_table[
            ['fc_element', 'fc_index', 'fc_variable', 'ctrl_values', 'logic_typ', 'write_flag']].apply(tuple, axis=1):

            ctrl_typ = np.array(logic_typ)
            values, counts = np.unique(ctrl_typ, return_counts=True)

            if len(values) == 1:
                if values.item() == 'low':
                    write_to_net(net, fc_element, fc_index, fc_variable, min(ctrl_values), write_flag[0])
                elif values.item() == 'high':
                    write_to_net(net, fc_element, fc_index, fc_variable, max(ctrl_values), write_flag[0])
                else:
                    logger.warning("Override logic selector type not yet implemented for common final control element")
            else:
                logger.warning("Multiple override logic selectors implemented for common final control element"
                               " at " + str(fc_element) + ', ' + str(fc_index) + ', ' + str(fc_variable))

        cls.controller_mv_table.drop(cls.controller_mv_table.index, inplace=True)
        cls.collect_ctrl_active = False
