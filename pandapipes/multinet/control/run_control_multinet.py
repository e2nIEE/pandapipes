# Copyright (c) 2020-2022 by Fraunhofer Institute for Energy Economics
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
    """
    Within a control loop after all controllers applied their their action "_evaluate_multinet" checks if all
    nets affectd in one level did converge or not

    :param multinet: multinet with multinet controllers, net distinct controllers and several pandapipes/pandapower nets
    :type multinet: pandapipes.Multinet
    :param levelorder: list of tuples given the correct order of the different controllers within one level
    :type levelorder: list
    :param ctrl_variables: contains all relevant information and boundaries required for a successful control run
    :type ctrl_variables: dict
    :param kwargs: additional keyword arguments handed to each run function
    :type kwargs: dict
    :return: as the ctrl_variables are adapted they are returned
    :rtype: dict
    """
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
    """
    This function determines the relevant nets in each level, i.e. only the nets affected in each level are investigated
    and checked.

    :param multinet: multinet with multinet controllers, net distinct controllers and several pandapipes/pandapower nets
    :type multinet: pandapipes.Multinet
    :param levelorder: list of tuples given the correct order of the different controllers within one level
    :type levelorder: list
    :return: dictionary of all nets in multinet.nets. Entries are booleans. True means net is in the
    corresponding level, False means it is not.
    :rtype: dict
    """
    net_names = dict()

    nns = []
    levelorder = np.array(levelorder)
    rel_levelorder_multi = levelorder[:, 1].__eq__(multinet)
    controller = levelorder[rel_levelorder_multi, 0]
    nns += [ctrl.get_all_net_names() for ctrl in controller]

    for net_name in multinet['nets'].keys():
        net = multinet['nets'][net_name]
        rel_levelorder = levelorder[:, 1].__eq__(net)
        level_excl = [False if net_name not in nn else True for nn in nns]
        rel_levelorder_multi[rel_levelorder_multi] = level_excl
        net_names[net_name] = np.maximum(rel_levelorder_multi, rel_levelorder)

    return net_names


def net_initialization_multinet(multinet, ctrl_variables, **kwargs):
    """
    If one controller affecting a net requires an initial_run, a loadflow/pipeflow is conducted.

    :param multinet: multinet with multinet controllers, net distinct controllers and several pandapipes/pandapower nets
    :type multinet: pandapipes.Multinet
    :param ctrl_variables: contains all relevant information and boundaries required for a successful control run
    :type ctrl_variables: dict
    :param kwargs: additional keyword arguments handed to each run function
    :type kwargs: dict
    :return: as the ctrl_variables are adapted they are returned
    :rtype: dict
    """
    ctrl_variables['converged'] = False

    for net_name in multinet['nets'].keys():
        net = multinet['nets'][net_name]
        kwargs['recycle'], kwargs['only_v_results'] = \
            ctrl_variables[net_name]['recycle'], ctrl_variables[net_name]['only_v_results']
        ctrl_variables[net_name] = net_initialization(net, ctrl_variables[net_name], **kwargs)
        ctrl_variables['converged'] = max(ctrl_variables['converged'], ctrl_variables[net_name]['converged'])
    return ctrl_variables


def run_control(multinet, ctrl_variables=None, max_iter=30, **kwargs):
    """
    Main function to call a multnet with controllers.

    Function is running control loops for the controllers specified in net.controller
    Runs controller until each one converged or max_iter is hit.

    1. Call initialize_control() on each controller
    2. Calculate an inital run (if it is enabled, i.e. setting the initial_run veriable to True)
    3. Repeats the following steps in ascending order of controller_order until total convergence of all
       controllers for each level:
       a) Evaluate individual convergence for all controllers in the level
       b) Call control_step() for all controllers in the level on diverged controllers
       c) Fire run function (or optionally another function like runopf or whatever you defined)
    4. Call finalize_control() on each controller

    :param multinet: multinet with multinet controllers, net distinct controllers and several pandapipes/pandapower nets
    :type multinet: pandapipes.Multinet
    :param ctrl_variables: contains all relevant information and boundaries required for a successful control run. To \
           define ctrl_variables yourself, following entries for each net are required:\n
           - level (list): gives a list of levels to be investigated \n
           - controller_order (list): nested list of tuples given the correct order of the
             different controllers within one level\
           - run (funct, e.g. pandapower.runpp, pandapipes.pipeflow): function to be used to
             conduct a loadflow/pipeflow \n
           - initial_run (boolean): Is a initial_run for a net required or not\n
           - continue_on_divergence (boolean): What to do if loadflow/pipeflow is not converging, fires control_repair
    :type ctrl_variables: dict, default: None
    :param max_iter: number of iterations for each controller to converge
    :type max_iter: int, default: 30
    :param kwargs: additional keyword arguments handed to each run function
    :type kwargs: dict
    :return: runs an entire control loop
    :rtype: None
    """
    ctrl_variables = prepare_run_ctrl(multinet, ctrl_variables)
    controller_order = ctrl_variables['controller_order']

    # initialize each controller prior to the first power flow
    control_initialization(controller_order)

    # initial run (takes time, but is not needed for every kind of controller)
    ctrl_variables = net_initialization_multinet(multinet, ctrl_variables, **kwargs)

    # run each controller step in given controller order
    control_implementation(multinet, controller_order, ctrl_variables, max_iter,
                           evaluate_net_fct=_evaluate_multinet, **kwargs)

    # call finalize function of each controller
    control_finalization(controller_order)


def get_controller_order_multinet(multinet):
    """
    Defining the controller order per level.

    Takes the order and level columns from net.controller.
    If levels are specified, the levels and orders are executed in ascending order.

    :param multinet: multinet with multinet controllers, net distinct controllers and several pandapipes/pandapower nets
    :type multinet: pandapipes.Multinet
    :return: nested list of tuples given the correct order of the controllers, respectively for each level
    :rtype: list
    """

    net_list = []
    controller_list = []

    if hasattr(multinet, "controller") and len(multinet.controller[multinet.controller.in_service]) != 0:
        # if no controllers are in the net, we have no levels and no order lists
        multinets = [multinet] * len(multinet.controller)
        net_list += multinets
        controller_list += [multinet.controller.values]

    for net_name in multinet['nets'].keys():
        net = multinet['nets'][net_name]
        if not (hasattr(net, 'controller') and len(net.controller[net.controller.in_service]) != 0):
            # if no controllers are in the net, we have no levels and no order lists
            continue
        nets = [net] * len(net.controller)
        net_list += nets
        controller_list += [net.controller.values]

    if not len(controller_list):
        # if no controllers are in the net, we have no levels and no order lists
        return [0], [[]]
    else:
        controller_list = pd.DataFrame(np.concatenate(controller_list), columns=multinet.controller.columns)
        controller_list = controller_list.astype(multinet.controller.dtypes)
        return get_controller_order(net_list, controller_list)


def prepare_run_ctrl(multinet, ctrl_variables, **kwargs):
    """
    Prepares run control functions.

    Internal variables needed:
        - level (list): gives a list of levels to be investigated
        - controller_order (list): nested list of tuples given the correct order of the different controllers
        within one level
        - run (funct, e.g. pandapower.runpp, pandapipes.pipeflow): function to be used to conduct a loadflow/pipeflow
        - initial_run (boolean): Is a initial_run for a net required or not
        - continue_on_divergence (boolean): What to do if loadflow/pipeflow is not converging, fires control_repair

    You don't need to define it for each net. If one net is not defined, the default settings are used.

    :param multinet: multinet with multinet controllers, net distinct controllers and several pandapipes/pandapower nets
    :type multinet: pandapipes.Multinet
    :param ctrl_variables: contains all relevant information and boundaries required for a successful control run.
    :type ctrl_variables: dict
    :return: adapted ctrl_variables for all nets with all required boundary informaiton
    :rtype: dict
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
                    ctrl_variables[net_name] = {'run': None, 'initial_run': None,
                                                'continue_on_divergence': None, 'errors': ()}
                net = multinet['nets'][net_name]
                if isinstance(net, ppipes.pandapipesNet):
                    ctrl_variables_net = prepare_run_ctrl_ppipes(net, None, **kwargs)
                elif isinstance(net, pp.pandapowerNet):
                    ctrl_variables_net = prepare_run_ctrl_pp(net, None, **kwargs)
                else:
                    raise ValueError('the given nets are neither pandapipes nor pandapower nets')

                ctrl_variables[net_name]['run'] = ctrl_variables_net['run']
                ctrl_variables[net_name]['errors'] = ctrl_variables_net['errors']
                ctrl_variables[net_name]['initial_run'] = ctrl_variables[net_name]['initial_run'] if \
                    ctrl_variables[net_name]['initial_run'] is not None else ctrl_variables_net['initial_run']
                ctrl_variables[net_name]['only_v_results'], ctrl_variables[net_name]['recycle'] = \
                    get_recycle(ctrl_variables_net)
                ctrl_variables[net_name]['continue_on_divergence'] = \
                    ctrl_variables[net_name]['continue_on_divergence'] if \
                    ctrl_variables[net_name]['continue_on_divergence'] is not None else \
                    ctrl_variables_net['continue_on_divergence']
                excl_net += [net_name]

    for net_name in multinet['nets'].keys():
        if net_name in excl_net:
            continue
        if net_name not in ctrl_variables.keys():
            ctrl_variables[net_name] = {'run': None, 'initial_run': False,
                                        'continue_on_divergence': None, 'errors': ()}
        net = multinet['nets'][net_name]
        if isinstance(net, ppipes.pandapipesNet):
            ctrl_variables_net = prepare_run_ctrl_ppipes(net, None, **kwargs)
        elif isinstance(net, pp.pandapowerNet):
            ctrl_variables_net = prepare_run_ctrl_pp(net, None, **kwargs)
        else:
            raise ValueError('the given nets are neither pandapipes nor pandapower nets')
        ctrl_variables[net_name]['run'] = ctrl_variables_net['run']
        ctrl_variables[net_name]['errors'] = ctrl_variables_net['errors']
        ctrl_variables[net_name]['initial_run'] = ctrl_variables[net_name]['initial_run'] if \
            ctrl_variables[net_name]['initial_run'] is not None else ctrl_variables_net['initial_run']
        ctrl_variables[net_name]['only_v_results'], ctrl_variables[net_name]['recycle'] = \
            get_recycle(ctrl_variables_net)
        ctrl_variables[net_name]['continue_on_divergence'] = \
            ctrl_variables[net_name]['continue_on_divergence'] if \
            ctrl_variables[net_name]['continue_on_divergence'] is not None else \
            ctrl_variables_net['continue_on_divergence']

    if ('check_each_level') in kwargs:
        check = kwargs.pop('check_each_level')
        ctrl_variables['check_each_level'] = check
    else:
        ctrl_variables['check_each_level'] = True

    ctrl_variables['errors'] = (NetCalculationNotConverged,)

    ctrl_variables['level'], ctrl_variables['controller_order'] = get_controller_order_multinet(multinet)

    return ctrl_variables
