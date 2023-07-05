# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import statistics as st

import numpy as np
import pandas as pd

import pandapipes as pp
from pandapipes.component_models.pipe_component import Pipe
from pandapipes.properties.fluids import get_fluid

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def pipeflow_openmodelica_comparison(net, log_results=True, friction_model='colebrook',
                                     mode='hydraulics', only_update_hydraulic_matrix=False,
                                     use_numba=True):
    """
        Comparison of the calculations of OpenModelica and pandapipes.

        :param net: converted OpenModelica network
        :type net: pandapipesNet
        :param log_results:
        :type log_results: bool, True
        :param friction_model:
        :type friction_model: str, "colebrook"
        :param mode:
        :type mode: str, "nomral"
        :param only_update_hydraulic_matrix:
        :type only_update_hydraulic_matrix: bool, False
        :param use_numba: whether to use numba for pipeflow calculations
        :type use_numba: bool, True
        :return: p_diff, v_diff_abs
        :rtype: one-dimensional ndarray with axis labels
    """
    pp.pipeflow(net, stop_condition="tol", iter=100, tol_p=1e-7, tol_v=1e-7,
                friction_model=friction_model, mode=mode, use_numba=use_numba,
                only_update_hydraulic_matrix=only_update_hydraulic_matrix)

    logger.debug(net.res_junction)
    logger.debug(net.res_pipe)

    p_om = net.junction.p_om
    p_valid = pd.notnull(p_om)
    p_om = p_om.loc[p_valid]

    if get_fluid(net).is_gas:
        if 'pipe' in net:
            v_diff_from_pipe, v_diff_to_pipe, v_diff_mean_pipe, v_diff_abs_pipe, \
                v_mean_pandapipes_pipe, v_om_pipe = retrieve_velocity_gas(net, 'pipe')
        else:
            v_diff_abs_pipe = pd.Series(dtype='float64')
            v_om_pipe = pd.Series(dtype='float64')
            v_mean_pandapipes_pipe = pd.Series(dtype='float64')
            v_diff_from_pipe = pd.Series(dtype='float64')
            v_diff_to_pipe = pd.Series(dtype='float64')
            v_diff_mean_pipe = pd.Series(dtype='float64')

        diff_results_v_pipe = pd.DataFrame(
            {"diff_v_from_pipe": v_diff_from_pipe, "diff_v_to_pipe": v_diff_to_pipe,
             "diff_v_mean_pipe": v_diff_mean_pipe, "diff_v_abs_pipe": v_diff_abs_pipe})

        if 'valve' in net:
            v_diff_from_valve, v_diff_to_valve, v_diff_mean_valve, v_diff_abs_valve, \
                v_mean_pandapipes_valve, v_om_valve = retrieve_velocity_gas(net, 'valve')
        else:
            v_diff_abs_valve = pd.Series(dtype='float64')
            v_om_valve = pd.Series(dtype='float64')
            v_mean_pandapipes_valve = pd.Series(dtype='float64')
            v_diff_from_valve = pd.Series(dtype='float64')
            v_diff_to_valve = pd.Series(dtype='float64')
            v_diff_mean_valve = pd.Series(dtype='float64')

        diff_results_v_valve = pd.DataFrame(
            {"diff_v_from_valve": v_diff_from_valve, "diff_v_to_valve": v_diff_to_valve,
             "diff_v_mean_valve": v_diff_mean_valve, "diff_v_abs_valve": v_diff_abs_valve})
    else:
        if 'pipe' in net:
            v_diff_mean_pipe, v_diff_abs_pipe, v_mean_pandapipes_pipe, v_om_pipe = \
                retrieve_velocity_liquid(net, element="pipe")

            if mode != "hydraulics":
                T_diff_mean_pipe, T_diff_abs_pipe, T_mean_pandapipes_pipe, T_om_pipe = \
                    retrieve_temperature_liquid(net)
        else:
            v_diff_abs_pipe = pd.Series(dtype='float64')
            v_om_pipe = pd.Series(dtype='float64')
            v_mean_pandapipes_pipe = pd.Series(dtype='float64')
            v_diff_mean_pipe = pd.Series(dtype='float64')

            if mode != "hydraulics":
                T_diff_abs_pipe = pd.Series(dtype='float64')
                T_om_pipe = pd.Series(dtype='float64')
                T_mean_pandapipes_pipe = pd.Series(dtype='float64')
                T_diff_mean_pipe = pd.Series(dtype='float64')

        if 'valve' in net:
            v_diff_mean_valve, v_diff_abs_valve, v_mean_pandapipes_valve, v_om_valve = \
                retrieve_velocity_liquid(net, element="valve")
        else:
            v_diff_abs_valve = pd.Series(dtype='float64')
            v_om_valve = pd.Series(dtype='float64')
            v_mean_pandapipes_valve = pd.Series(dtype='float64')
            v_diff_mean_valve = pd.Series(dtype='float64')

        diff_results_v_pipe = pd.DataFrame({"diff_v_mean_pipe": v_diff_mean_pipe,
                                            "diff_v_abs_pipe": v_diff_abs_pipe})
        diff_results_v_valve = pd.DataFrame({"diff_v_mean_valve": v_diff_mean_valve,
                                             "diff_v_abs_valve": v_diff_abs_valve})

    v_diff_abs = pd.concat([v_diff_abs_pipe, v_diff_abs_valve], ignore_index=True)
    v_diff_abs.dropna(inplace=True)

    p_pandapipes = net.res_junction.p_bar.loc[p_valid].values.astype(np.float64).round(4)
    p_diff = np.abs(1 - p_pandapipes / p_om)

    p_diff = pd.Series(p_diff, range(len(p_diff)), dtype='float64')
    v_diff_abs = pd.Series(v_diff_abs, range(len(v_diff_abs)), dtype='float64')

    '''
    print("\n p_diff = \n", p_diff)
    print("\n v_diff_abs = \n", v_diff_abs)

    print("\n p_diff < 0.01 = \n", p_diff < 0.01)
    print("\n v_diff_abs < 0.05 = \n", v_diff_abs < 0.05)
    '''

    if log_results:
        logger.info("p_om %s" % p_om)
        logger.info("p_PP %s" % p_pandapipes)
        logger.info("v_om_pipe %s" % v_om_pipe)
        logger.info("v_PP_valve %s" % v_om_valve)
        logger.info("v_PP_pipe %s" % v_mean_pandapipes_pipe)
        logger.info("v_PP_valve %s" % v_mean_pandapipes_valve)

        logger.info("pressure difference: %s" % p_diff)
        logger.info("velocity difference pipe: \n %s" % diff_results_v_pipe)
        logger.info("velocity difference valve: \n %s" % diff_results_v_valve)

    if mode == "hydraulics":
        return p_diff, v_diff_abs
    else:
        return p_diff, v_diff_abs, T_diff_mean_pipe


def retrieve_velocity_liquid(net, element="pipe"):
    """
        Get the calculated velocities for a liquid fluid in pandapipes and OpenModelica and
        calculate the absolute and relative errors.

        :param net: converted OpenModelica network
        :type net: pandapipesNet
        :param element: either pipe or valve
        :type element: str, default "pipe"
        :return: relative and absolute error (v_diff_mean, v_diff_abs) of average velocities,
        calculated velocities from pandapipes and OpenModelica (v_mean_pandapipes, v_om)
        :rtype: one-dimensional ndarray with axis labels
    """
    if 'v_om' not in net[element]:
        net[element]['v_om'] = []
    v_om = net[element].v_om
    v_valid = pd.notnull(v_om)
    v_om = v_om.loc[v_valid]
    v_om[v_om == 0] += 0.0001

    if element == "pipe":
        v_mean_pandapipes = net.res_pipe.v_mean_m_per_s.loc[v_valid].values.astype(
            np.float64).round(4)

    if element == "valve":
        v_mean_pandapipes = net.res_valve.v_mean_m_per_s.loc[v_valid].values.astype(
            np.float64).round(4)

    v_mean_pandapipes[v_mean_pandapipes == 0] += 0.0001

    v_diff_mean = np.abs(1 - v_mean_pandapipes / v_om)
    v_diff_abs = np.abs(v_om - v_mean_pandapipes)

    v_diff_mean = pd.Series(v_diff_mean, range(len(v_diff_mean)), dtype='float64')
    v_diff_abs = pd.Series(v_diff_abs, range(len(v_diff_abs)), dtype='float64')
    v_om = pd.Series(v_om, range(len(v_om)), dtype='float64')

    return v_diff_mean, v_diff_abs, v_mean_pandapipes, v_om


def retrieve_velocity_gas(net, element='pipe'):
    """
        Get the calculated velocities for a gaseous fluid in pandapipes and OpenModelica and
        calculate the absolute and relative errors.

        :param net: converted OpenModelica network
        :type net: pandapipesNet
        :param element: either pipe or valve
        :type element: str, default "pipe"
        :return: relative and absolute error (v_diff_mean, v_diff_abs) of average velocities,
        calculated velocities from pandapipes and OpenModelica (v_mean_pandapipes, v_om)
        :rtype: one-dimensional ndarray with axis labels
    """
    if 'v_om' not in net[element]:
        net[element]['v_om'] = []
    v_om = net[element].v_om
    v_valid = pd.notnull(v_om)
    v_om = v_om.loc[v_valid]

    res_element = net['res_' + element].loc[v_valid, :]
    v_from_pandapipes = res_element.v_from_m_per_s.values.astype(np.float64).round(4)
    v_to_pandapipes = res_element.v_to_m_per_s.values.astype(np.float64).round(4)
    v_mean_pandapipes = res_element.v_mean_m_per_s.values.astype(np.float64).round(4)

    v_om[v_om == 0] += 0.0001
    v_mean_pandapipes[v_mean_pandapipes == 0] += 0.0001
    v_from_pandapipes[v_from_pandapipes == 0] += 0.0001
    v_to_pandapipes[v_to_pandapipes == 0] += 0.0001

    v_diff_from = np.abs(1 - v_from_pandapipes / v_om)
    v_diff_to = np.abs(1 - v_to_pandapipes / v_om)
    v_diff_mean = np.abs(1 - v_mean_pandapipes / v_om)
    v_diff_abs = np.abs(v_om - v_mean_pandapipes)

    v_diff_mean = pd.Series(v_diff_mean, range(len(v_diff_mean)), dtype='float64')
    v_diff_from = pd.Series(v_diff_from, range(len(v_diff_from)), dtype='float64')
    v_diff_to = pd.Series(v_diff_to, range(len(v_diff_to)), dtype='float64')
    v_diff_abs = pd.Series(v_diff_abs, range(len(v_diff_abs)), dtype='float64')
    v_om = pd.Series(v_om, range(len(v_om)), dtype='float64')

    return v_diff_from, v_diff_to, v_diff_mean, v_diff_abs, v_mean_pandapipes, v_om


def retrieve_temperature_liquid(net):
    """
        Get the calculated temperatures for a liquid fluid in pandapipes and OpenModelica and
        calculate the absolute and relative errors.

        :param net: converted OpenModelica network
        :type net: pandapipesNet
        :return: relative and absolute error (T_diff_mean, T_diff_abs) of average velocities,
        calculated velocities from pandapipes and OpenModelica (T_mean_pandapipes, T_om)
        :rtype: one-dimensional ndarray with axis labels
    """
    T_om = net.pipe["T_om"]
    num_of_pipes = len(T_om)
    T_mean_om = np.arange(num_of_pipes)
    T_mean_pandapipes = np.arange(num_of_pipes)

    for i in range(num_of_pipes):
        T_mean_om[i] = st.mean(T_om[i])

    for j in range(num_of_pipes):
        pipe_res = Pipe.get_internal_results(net, [j])
        T_mean_pandapipes[j] = st.mean(pipe_res["TINIT"][:, 1])

    T_diff_mean = np.abs(1 - T_mean_pandapipes / T_mean_om)
    T_diff_abs = np.abs(T_mean_om - T_mean_pandapipes)

    T_diff_mean = pd.Series(T_diff_mean, range(len(T_diff_mean)), dtype='float64')
    T_diff_abs = pd.Series(T_diff_abs, range(len(T_diff_abs)), dtype='float64')

    T_om = pd.Series(T_om, range(len(T_om)), dtype='float64')
    T_mean_pandapipes = pd.Series(T_mean_pandapipes, range(len(T_mean_pandapipes)), dtype='float64')

    return T_diff_mean, T_diff_abs, T_mean_pandapipes, T_om
