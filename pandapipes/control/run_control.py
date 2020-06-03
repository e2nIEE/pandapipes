# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

from pandapower.control import *
import pandapipes as ppipe


def run_control_ppipe(net, ctrl_variables=None, max_iter=30, continue_on_lf_divergence=False,
                      **kwargs):
    """
    Function to run a control of the pandapipes network.

    :param net: The pandapipes network
    :type net: pandapipesNet
    :param ctrl_variables: used control variables. If None, default control variables are used.
    :type ctrl_variables: ?, default None
    :param max_iter: maximal amount of iterations
    :type max_iter: int, default 30
    :param continue_on_lf_divergence: ?
    :type continue_on_lf_divergence: bool, default False
    :param kwargs: additional key word arguments
    :return: No Output.
    """
    if ctrl_variables is None:
        ctrl_variables = ctrl_variables_ppipe_default(net)
    else:
        ctrl_variables["initial_powerflow"] = ctrl_variables["initial_pipeflow"]
    run_control(net, ctrl_variables=ctrl_variables, max_iter=max_iter,
                continue_on_lf_divergence=continue_on_lf_divergence, **kwargs)


def ctrl_variables_ppipe_default(net):
    """
    Function that defines default control variables.

    :param net: The pandapipes network
    :type net: pandapipesNet
    :return: ctrl_variables
    :rtype: ?
    """
    ctrl_variables = dict()
    ctrl_variables["level"], ctrl_variables["controller_order"] = get_controller_order(net)
    ctrl_variables["run"] = ppipe.pipeflow
    ctrl_variables["initial_pipeflow"] = check_for_initial_pipeflow(
        ctrl_variables["controller_order"])
    ctrl_variables["initial_powerflow"] = ctrl_variables["initial_pipeflow"]
    return ctrl_variables


def check_for_initial_pipeflow(controllers):
    """
    Function checking if any of the controllers need an initial pipe flow
    If net has no controllers, an initial pipe flow is done by default.
    :param controllers:
    :type controllers:
    :return:
    :rtype:
    """
    if not len(controllers[0]):
        return True

    for order in controllers:
        for ctrl in order:
            if ctrl.initial_pipeflow:
                return True
    return False
