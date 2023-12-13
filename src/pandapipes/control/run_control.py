# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

from pandapower.control import run_control as run_control_pandapower, \
    prepare_run_ctrl as prepare_run_control_pandapower
import pandapipes as ppipe
from pandapipes.pipeflow import PipeflowNotConverged


def run_control(net, ctrl_variables=None, max_iter=30, **kwargs):
    """
    Function to run a control of the pandapipes network.

    :param net: The pandapipes network
    :type net: pandapipesNet
    :param ctrl_variables: Used control variables. If None, default control variables are used.
    :type ctrl_variables: dict, default None
    :param max_iter: Maximal amount of iterations
    :type max_iter: int, default 30
    :param kwargs: Additional keyword arguments
    :type kwargs: dict
    :return: No output
    """
    if ctrl_variables is None:
        ctrl_variables = prepare_run_ctrl(net, None, **kwargs)

    run_control_pandapower(net, ctrl_variables=ctrl_variables, max_iter=max_iter, **kwargs)


def prepare_run_ctrl(net, ctrl_variables, **kwargs):
    """
    Function that defines default control variables.

    :param net: The pandapipes network
    :type net: pandapipesNet
    :return: ctrl_variables
    :rtype: dict
    """
    if ctrl_variables is None:
        ctrl_variables = prepare_run_control_pandapower(net, None, **kwargs)
        ctrl_variables["run"] = ppipe.pipeflow

    ctrl_variables["errors"] = (PipeflowNotConverged,)  # has to be a tuple

    return ctrl_variables



