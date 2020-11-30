# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
import pandapipes as ppipes
import pandapower as pp
import pandas as pd
from pandapipes.control.run_control import prepare_run_ctrl as prepare_run_ctrl_ppipes
from pandapower.control.run_control import prepare_run_ctrl as prepare_run_ctrl_pp, \
    net_initialization, get_recycle, control_initialization, control_finalization, \
    _evaluate_net as _evaluate_net, control_implementation, get_controller_order, NetCalculationNotConverged

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def _evaluate_multinet(multinet, levelorder, ctrl_variables, **kwargs):
    levelorder = np.array(levelorder)
    multinet_converged = []
    rel_nets = _relevant_nets(multinet, levelorder)
    for net_name in multinet['nets'].keys():
        net = multinet['nets'][net_name]
        rel_levelorder = levelorder[rel_nets[net_name]]
        ctrl_variables[net_name] = _evaluate_net(net, rel_levelorder, ctrl_variables[net_name], **kwargs) if np.any(
            rel_nets[net_name]) else ctrl_variables[net_name]
        multinet_converged += [ctrl_variables[net_name]['converged']]
    ctrl_variables['converged'] = np.all(multinet_converged)
    return ctrl_variables


def _relevant_nets(multinet, levelorder):
    net_names = dict()

    nns = []
    levelorder = np.array(levelorder)
    rel_levelorder_multi = levelorder[:, 1].__eq__(multinet)
    controller = levelorder[rel_levelorder_multi, 0]
    nns += [ctrl.get_all_net_names() for ctrl in controller]

    for net_name in multinet['nets'].keys():
        net = multinet['nets'][net_name]
        rel_levelorder = levelorder[:, 1].__eq__(net)
        level_excl = [False if not net_name in nn else True for nn in nns]
        rel_levelorder_multi[rel_levelorder_multi] = level_excl
        net_names[net_name] = np.maximum(rel_levelorder_multi, rel_levelorder)

    return net_names


def net_initialization_multinet(multinet, ctrl_variables, **kwargs):
    ctrl_variables['converged'] = False

    for net_name in multinet['nets'].keys():
        net = multinet['nets'][net_name]
        kwargs['recycle'], kwargs['only_v_results'] = \
            ctrl_variables[net_name]['recycle'], ctrl_variables[net_name]['only_v_results']
        ctrl_variables[net_name] = net_initialization(net, ctrl_variables[net_name], **kwargs)
        ctrl_variables['converged'] = max(ctrl_variables['converged'], ctrl_variables[net_name]['converged'])


def run_control(multinet, ctrl_variables=None, max_iter=30, continue_on_lf_divergence=False, **kwargs):
    ctrl_variables = prepare_run_ctrl(multinet, ctrl_variables)
    controller_order = ctrl_variables['controller_order']

    # initialize each controller prior to the first power flow
    control_initialization(controller_order)

    # initial run (takes time, but is not needed for every kind of controller)
    net_initialization_multinet(multinet, ctrl_variables, **kwargs)

    # run each controller step in given controller order
    control_implementation(multinet, controller_order, ctrl_variables, max_iter,
                           evaluate_net_fct=_evaluate_multinet, **kwargs)

    # call finalize function of each controller
    control_finalization(controller_order)


def get_controller_order_multinet(multinet):
    """
    Defining the controller order per level
    Takes the order and level columns from net.controller
    If levels are specified, the levels and orders are executed in ascending order.
    :param multinet:
    :return: level_list - list of levels to be run
    :return: controller_order - list of controller order lists per level
    """

    net_list = []
    controller_list = []

    if hasattr(multinet, "controller") and len(multinet.controller[multinet.controller.in_service]) != 0:
        # if no controllers are in the net, we have no levels and no order lists
        multinets = [multinet] * len(multinet.controller)
        net_list += multinets
        controller_list += [multinet.controller]

    for net_name in multinet['nets'].keys():
        net = multinet['nets'][net_name]
        if not (hasattr(net, 'controller') and len(net.controller[net.controller.in_service]) != 0):
            # if no controllers are in the net, we have no levels and no order lists
            continue
        nets = [net] * len(net.controller)
        net_list += nets
        controller_list += [net.controller]

    controller_list = pd.concat(controller_list).reset_index(drop=True)

    if not controller_list.size:
        # if no controllers are in the net, we have no levels and no order lists
        return [0], [[]]
    else:
        return get_controller_order(net_list, controller_list)


def prepare_run_ctrl(multinet, ctrl_variables):
    """
    Prepares run control functions. Internal variables needed:

    **controller_order** (list) - Order in which controllers in net.controller will be called
    **runpp** (function) - the runpp function (for time series a faster version is possible)
    **initial_powerflow** (bool) - some controllers need an initial powerflow prior to the control step

    """
    # sort controller_order by order if not already done
    if ctrl_variables is None:
        ctrl_variables = dict()

    excl_net = []

    if hasattr(multinet, "controller") and len(multinet.controller[multinet.controller.in_service]) != 0:
        for _, c in multinet['controller'].iterrows():
            net_names = c.object.get_all_net_names()
            for net_name in net_names:
                if net_name not in ctrl_variables.keys():
                    ctrl_variables[net_name] = {'run': None, 'initial_run': None, 'errors': ()}
                net = multinet['nets'][net_name]
                if isinstance(net, ppipes.pandapipesNet):
                    ctrl_variables_net = prepare_run_ctrl_ppipes(net, None)
                elif isinstance(net, pp.pandapowerNet):
                    ctrl_variables_net = prepare_run_ctrl_pp(net, None)
                else:
                    raise ValueError('the given nets are neither pandapipes nor pandapower nets')

                ctrl_variables[net_name]['run'] = ctrl_variables_net['run']
                ctrl_variables[net_name]['errors'] = ctrl_variables[net_name]['errors'] if ctrl_variables[net_name][
                    'errors'] else ctrl_variables_net['errors']
                ctrl_variables[net_name]['initial_run'] = ctrl_variables[net_name]['initial_run'] if \
                    ctrl_variables[net_name]['initial_run'] is not None else ctrl_variables_net['initial_run']
                ctrl_variables[net_name]['only_v_results'], ctrl_variables[net_name]['recycle'] = \
                    get_recycle(ctrl_variables_net)
                excl_net += [net_name]

    for net_name in multinet['nets'].keys():
        if net_name in excl_net:
            continue
        if net_name not in ctrl_variables.keys():
            ctrl_variables[net_name] = {'run': None, 'initial_run': False, 'errors': ()}
        net = multinet['nets'][net_name]
        if isinstance(net, ppipes.pandapipesNet):
            ctrl_variables_net = prepare_run_ctrl_ppipes(net, None)
        elif isinstance(net, pp.pandapowerNet):
            ctrl_variables_net = prepare_run_ctrl_pp(net, None)
        else:
            raise ValueError('the given nets are neither pandapipes nor pandapower nets')
        ctrl_variables[net_name]['run'] = ctrl_variables_net['run']
        ctrl_variables[net_name]['errors'] = ctrl_variables[net_name]['errors'] if ctrl_variables[net_name][
            'errors'] else ctrl_variables_net['errors']
        ctrl_variables[net_name]['initial_run'] = ctrl_variables[net_name]['initial_run'] if \
            ctrl_variables[net_name]['initial_run'] is not None else ctrl_variables_net['initial_run']

    ctrl_variables['errors'] = (NetCalculationNotConverged,)

    ctrl_variables['level'], ctrl_variables['controller_order'] = get_controller_order_multinet(multinet)

    return ctrl_variables
