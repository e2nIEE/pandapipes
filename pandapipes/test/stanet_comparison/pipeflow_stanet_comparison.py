# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
import pandapipes
import pandas as pd
from pandapipes.properties.fluids import get_fluid

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def pipeflow_stanet_comparison(net, log_results=True, friction_model='nikuradse',
                               only_update_hydraulic_matrix=False, **kwargs):
    """

    :param net:
    :type net:
    :param log_results:
    :type log_results:
    :param plot_net:
    :type plot_net:
    :param friction_model:
    :type friction_model:
    :param only_update_hydraulic_matrix:
    :type only_update_hydraulic_matrix:
    :return:
    :rtype:
    """
    pandapipes.pipeflow(net, mode='hydraulics', stop_condition="tol", iter=100, tol_p=1e-7,
                        tol_v=1e-7, friction_model=friction_model,
                        only_update_hydraulic_matrix=only_update_hydraulic_matrix, **kwargs)

    p_stanet = net.junction.p_stanet
    p_valid = pd.notnull(p_stanet)
    p_stanet = p_stanet.loc[p_valid]

    if get_fluid(net).is_gas:
        if 'pipe' in net:
            v_diff_from_pipe, v_diff_to_pipe, v_diff_mean_pipe, v_diff_abs_pipe, \
            v_mean_pandapipes_pipe, v_stanet_pipe = retrieve_velocity_gas(net, 'pipe')
        else:
            v_diff_abs_pipe = pd.Series()
            v_stanet_pipe = pd.Series()
            v_mean_pandapipes_pipe = pd.Series()
            v_diff_from_pipe = pd.Series()
            v_diff_to_pipe = pd.Series()
            v_diff_mean_pipe = pd.Series()

        diff_results_v_pipe = pd.DataFrame(
            {"diff_v_from_pipe": v_diff_from_pipe, "diff_v_to_pipe": v_diff_to_pipe,
             "diff_v_mean_pipe": v_diff_mean_pipe, "diff_v_abs_pipe": v_diff_abs_pipe})

        if 'valve' in net:
            v_diff_from_valve, v_diff_to_valve, v_diff_mean_valve, v_diff_abs_valve, \
            v_mean_pandapipes_valve, v_stanet_valve = retrieve_velocity_gas(net, 'valve')
        else:
            v_diff_abs_valve = pd.Series()
            v_stanet_valve = pd.Series()
            v_mean_pandapipes_valve = pd.Series()
            v_diff_from_valve = pd.Series()
            v_diff_to_valve = pd.Series()
            v_diff_mean_valve = pd.Series()

        diff_results_v_valve = pd.DataFrame(
            {"diff_v_from_valve": v_diff_from_valve, "diff_v_to_valve": v_diff_to_valve,
             "diff_v_mean_valve": v_diff_mean_valve, "diff_v_abs_valve": v_diff_abs_valve})

    else:
        if 'pipe' in net:
            v_diff_mean_pipe, v_diff_abs_pipe, v_mean_pandapipes_pipe, v_stanet_pipe = \
                retrieve_velocity_liquid(net, 'pipe')
        else:
            v_diff_abs_pipe = pd.Series()
            v_stanet_pipe = pd.Series()
            v_mean_pandapipes_pipe = pd.Series()
            v_diff_mean_pipe = pd.Series()

        if 'valve' in net:
            v_diff_mean_valve, v_diff_abs_valve, v_mean_pandapipes_valve, v_stanet_valve = \
                retrieve_velocity_liquid(net, 'valve')
        else:
            v_diff_abs_valve = pd.Series()
            v_stanet_valve = pd.Series()
            v_mean_pandapipes_valve = pd.Series()
            v_diff_mean_valve = pd.Series()

        diff_results_v_pipe = pd.DataFrame({"diff_v_mean_pipe": v_diff_mean_pipe,
                                            "diff_v_abs_pipe": v_diff_abs_pipe})
        diff_results_v_valve = pd.DataFrame({"diff_v_mean_valve": v_diff_mean_valve,
                                             "diff_v_abs_valve": v_diff_abs_valve})

    p_pandapipes = net.res_junction.p_bar.loc[p_valid].values.astype(np.float64).round(4)
    p_diff = np.abs(1 - p_pandapipes / p_stanet)
    v_diff_abs = v_diff_abs_pipe.append(v_diff_abs_valve, ignore_index=True)
    v_diff_abs.dropna(inplace=True)

    # Avoiding division by zero

    if log_results:
        logger.info("p_sta %s" % p_stanet)
        logger.info("p_PP %s" % p_pandapipes)
        logger.info("v_sta_pipe %s" % v_stanet_pipe)
        logger.info("v_sta_valve %s" % v_stanet_valve)
        logger.info("v_PP_pipe %s" % v_mean_pandapipes_pipe)
        logger.info("v_PP_valve %s" % v_mean_pandapipes_valve)

        logger.info("Druckdifferenz: %s" % p_diff)
        logger.info("Geschwindigkeitsdifferenz Rohr: \n %s" % diff_results_v_pipe)
        logger.info("Geschwindigkeitsdifferenz Ventil: \n %s" % diff_results_v_valve)

    return p_diff, v_diff_abs


def retrieve_velocity_gas(net, element='pipe'):
    """

    :param net:
    :type net:
    :param element:
    :type element:
    :return:
    :rtype:
    """
    if 'v_stanet' not in net[element]:
        net[element]['v_stanet'] = []
    v_stanet = net[element].v_stanet
    v_valid = pd.notnull(v_stanet)
    v_stanet = v_stanet.loc[v_valid]

    res_element = net['res_' + element].loc[v_valid, :]
    v_from_pandapipes = res_element.v_from_m_per_s.values.astype(np.float64).round(4)
    v_to_pandapipes = res_element.v_to_m_per_s.values.astype(np.float64).round(4)
    v_mean_pandapipes = res_element.v_mean_m_per_s.values.astype(np.float64).round(4)

    v_stanet[v_stanet == 0] += 0.0001
    v_mean_pandapipes[v_mean_pandapipes == 0] += 0.0001
    v_from_pandapipes[v_from_pandapipes == 0] += 0.0001
    v_to_pandapipes[v_to_pandapipes == 0] += 0.0001

    v_diff_from = np.abs(1 - v_from_pandapipes / v_stanet)
    v_diff_to = np.abs(1 - v_to_pandapipes / v_stanet)
    v_diff_mean = np.abs(1 - v_mean_pandapipes / v_stanet)
    v_diff_abs = np.abs(v_stanet - v_mean_pandapipes)

    return v_diff_from, v_diff_to, v_diff_mean, v_diff_abs, v_mean_pandapipes, v_stanet


def retrieve_velocity_liquid(net, element='pipe'):
    """

    :param net:
    :type net:
    :param element:
    :type element:
    :return:
    :rtype:
    """
    if 'v_stanet' not in net[element]:
        net[element]['v_stanet'] = []
    v_stanet = net.pipe.v_stanet
    v_valid = pd.notnull(v_stanet)
    v_stanet = v_stanet.loc[v_valid]

    v_stanet[v_stanet == 0] += 0.0001
    v_mean_pandapipes = (net.res_pipe.v_mean_m_per_s.loc[v_valid]
                         * 1).values.astype(np.float64).round(4)

    v_mean_pandapipes[v_mean_pandapipes == 0] += 0.0001

    v_diff_mean = np.abs(1 - v_mean_pandapipes / v_stanet)
    v_diff_abs = np.abs(v_stanet - v_mean_pandapipes)

    return v_diff_mean, v_diff_abs, v_mean_pandapipes, v_stanet
