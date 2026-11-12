# Copyright (c) 2020-2026 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

from pandapipes.pf.calculation import execute_heat, execute_hydraulics, execute_bidirectional
from pandapipes.pf.pipeflow_setup import (
    get_net_option, init_options, create_lookups, initialize_pit, init_all_result_tables,

)
from pandapipes.pf.result_extraction import extract_all_results

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def set_logger_level_pipeflow(level):
    """Set logger level from outside to reduce/extend pipeflow() printout.

    :param level: levels according to 'logging' (i.e. DEBUG, INFO, WARNING, ERROR and CRITICAL)
    :type level: str
    :return: No output

    Example
    -------
        set_logger_level_pipeflow('WARNING')

    """
    logger.setLevel(level)


def pipeflow(net, **kwargs):
    """The main method used to start the solver to calculate the velocity, pressure and temperature\
    distribution for a given net. Different options can be entered for \\**kwargs, which control\
    the solver behaviour (see function :func:`init_options` for more information).

    :param net: The pandapipes net for which to perform the pipeflow
    :type net: pandapipesNet
    :param kwargs: A list of options controlling the solver behaviour
    :return: No output

    :Example:
        >>> pipeflow(net, mode="hydraulics")

    """
    init_pipeflow(net, **kwargs)

    execute_pipeflow(net)

    extract_all_results(net)


def  init_pipeflow(net, **kwargs):
    """Inputs & initialization of variables: physical constants/options, result tables, lookups and the internal PIT (pandapipes internal tables) arrays.

    :param net: The pandapipes net for which to perform the pipeflow
    :type net: pandapipesNet
    :param kwargs: A list of options controlling the solver behaviour
    :return: No output
    """
    init_options(net, **kwargs)
    init_all_result_tables(net)
    create_lookups(net)
    initialize_pit(net)

def execute_pipeflow(net):
    calculation_mode = get_net_option(net, "mode")
    calculate_hydraulics = calculation_mode in ["hydraulics", "sequential"]
    calculate_heat = calculation_mode in ["heat", "sequential"]
    calculate_bidrect = calculation_mode == "bidirectional"

    if not (calculate_hydraulics | calculate_heat | calculate_bidrect):
        raise UserWarning("No proper calculation mode chosen.")
    elif calculate_bidrect:
        execute_bidirectional(net)
    else:
        if calculate_hydraulics:
            execute_hydraulics(net)
        if calculate_heat:
            execute_heat(net)
