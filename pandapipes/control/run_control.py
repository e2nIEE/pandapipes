# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

from pandapower.control import run_control as run_control_pandapower, prepare_run_ctrl as prepare_run_control_pandapower
import pandapipes as ppipe
from pandapipes.pipeflow import PipeflowNotConverged

def run_control(net, ctrl_variables=None, max_iter=30, continue_on_lf_divergence=False,
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
        ctrl_variables = prepare_run_ctrl(net, None)


    run_control_pandapower(net, ctrl_variables=ctrl_variables, max_iter=max_iter,
                           continue_on_lf_divergence=continue_on_lf_divergence, **kwargs)


def prepare_run_ctrl(net, ctrl_variables):
    """
    Function that defines default control variables.

    :param net: The pandapipes network
    :type net: pandapipesNet
    :return: ctrl_variables
    :rtype: ?
    """
    if ctrl_variables is None:
        ctrl_variables = prepare_run_control_pandapower(net, None)
        ctrl_variables["run"] = ppipe.pipeflow

    if not "error_repair" in ctrl_variables:
        ctrl_variables["error_repair"] = (PipeflowNotConverged)
    return ctrl_variables



