# Copyright (c) 2020-2021 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

from numpy import linalg
import numpy as np
from numba import njit


def calc_lambda(v, eta, rho, d, k, gas_mode, friction_model, dummy, options):
    """
    Function calculates the friction factor of a pipe. Turbulence is calculated based on
    Nikuradse. If v equals 0, a value of 0.001 is used in order to avoid division by zero.
    This should not be a problem as the pressure loss term will equal zero (lambda * u^2).

    :param v:
    :type v:
    :param eta:
    :type eta:
    :param rho:
    :type rho:
    :param d:
    :type d:
    :param k:
    :type k:
    :param gas_mode:
    :type gas_mode:
    :param friction_model:
    :type friction_model:
    :param dummy:
    :type dummy:
    :param options:
    :type options:
    :return:
    :rtype:
    """
    if gas_mode:
        re, lambda_laminar, lambda_nikuradse = calc_lambda_nikuradse_comp(v, d, k, eta, rho)
    else:
        re, lambda_laminar, lambda_nikuradse = calc_lambda_nikuradse_incomp(v, d, k, eta, rho)

    if friction_model == "colebrook":
        # TODO: move this import to top level if possible
        from pandapipes.pipeflow import PipeflowNotConverged
        max_iter = options.get("max_iter_colebrook", 100)
        converged, lambda_colebrook = colebrook_nb(re, d, k, lambda_nikuradse, dummy, max_iter)
        if not converged:
            raise PipeflowNotConverged(
                "The Colebrook-White algorithm did not converge. There might be model "
                "inconsistencies. The maximum iterations can be given as 'max_iter_colebrook' "
                "argument to the pipeflow.")
        return lambda_colebrook, re
    elif friction_model == "swamee-jain":
        lambda_swamee_jain = 0.25 / ((np.log10(k/(3.7*d) + 5.74/(re**0.9)))**2)
        return lambda_swamee_jain, re
    else:
        # lambda_tot = np.where(re > 2300, lambda_laminar + lambda_nikuradse, lambda_laminar)
        lambda_tot = lambda_laminar + lambda_nikuradse
        return lambda_tot, re


def calc_der_lambda(v, eta, rho, d, k, friction_model, lambda_pipe):
    """
    Function calculates the derivative of lambda with respect to v. Turbulence is calculated based
    on Nikuradse. This should not be a problem as the pressure loss term will equal zero
    (lambda * u^2).

    :param v:
    :type v:
    :param eta:
    :type eta:
    :param rho:
    :type rho:
    :param d:
    :type d:
    :param k:
    :type k:
    :param friction_model:
    :type friction_model:
    :param lambda_pipe:
    :type lambda_pipe:
    :return:
    :rtype:
    """

    v_corr = np.where(v == 0, 0.00001, v)

    if friction_model == "colebrook":
        b_term = 2.51 * eta / (rho * d * np.sqrt(lambda_pipe) * v_corr) + k / (3.71 * d)

        df_dv = -2 * (2.51 * eta / (rho * np.sqrt(lambda_pipe) * v_corr ** 2)) \
                / (np.log(10) * b_term)

        df_dlambda = -0.5 * lambda_pipe ** (-3 / 2) - ((2.51 * eta) / (rho * d * v_corr)) \
                     * lambda_pipe ** (-3 / 2) * 1 / (np.log(10) * b_term)

        lambda_colebrook_der = df_dv / df_dlambda

        return lambda_colebrook_der
    elif friction_model == "swamee-jain":
        param = k/(3.7*d) + 5.74 * (np.abs(eta))**0.9 / ((np.abs(rho*v_corr*d))**0.9)
        lambda_swamee_jain_der = 0.5/np.log(10) * 1/(np.log(param) ** 3) * 1/param * 5.166 \
                                 * (np.abs(eta))**0.9/(((np.abs((rho*d)))**0.9) * (np.abs(v_corr))**1.9)
        return lambda_swamee_jain_der
    else:
        lambda_laminar_der = -(64 * eta) / (rho * v_corr ** 2 * d)
        return lambda_laminar_der


def colebrook(re, d, k, lambda_nikuradse, dummy, max_iter):
    """

    :param re:
    :type re:
    :param d:
    :type d:
    :param k:
    :type k:
    :param lambda_nikuradse:
    :type lambda_nikuradse:
    :param dummy:
    :type dummy:
    :param max_iter:
    :type max_iter:
    :return: lambda_cb
    :rtype:
    """
    lambda_cb = lambda_nikuradse
    converged = False
    error_lambda = []
    niter = 0

    # Inner Newton-loop for calculation of lambda
    while not converged and niter < max_iter:
        f = lambda_cb ** (-1 / 2) + 2 * np.log10(2.51 / (re * np.sqrt(lambda_cb)) + k / (3.71 * d))

        df_dlambda_cb = (-1 / 2 * lambda_cb ** (-3 / 2)) - (2.51 / re) * lambda_cb ** (-3 / 2) \
                        / (np.log(10) * 2.51 / (re * np.sqrt(lambda_cb) + k / (3.71 * d)))

        x = - f / df_dlambda_cb

        lambda_cb_old = lambda_cb
        lambda_cb = lambda_cb + x

        dx = np.abs(lambda_cb - lambda_cb_old) * dummy
        error_lambda.append(linalg.norm(dx) / (len(dx)))

        if error_lambda[niter] <= 1e-4:
            converged = True

        niter += 1

    return converged, lambda_cb


@njit
def calc_lambda_nikuradse_comp(v, d, k, eta, rho):
    lambda_nikuradse = np.empty_like(v)
    lambda_laminar = np.zeros_like(v)
    re = np.empty_like(v)
    v_abs = np.abs(v)
    for i in range(v.shape[0]):
        if v_abs[i] < 1e-6:
            re[i] = np.divide(rho[i] * 1e-6 * d[i], eta[i])
        else:
            re[i] = np.divide(rho[i] * v_abs[i] * d[i], eta[i])
        if v[i] != 0:
            lambda_laminar[i] = 64 / re[i]
        lambda_nikuradse[i] = np.divide(1, (2 * np.log10(d[i] / k[i]) + 1.14) ** 2)
    return re, lambda_laminar, lambda_nikuradse


@njit
def calc_lambda_nikuradse_incomp(v, d, k, eta, rho):
    lambda_nikuradse = np.empty_like(v)
    lambda_laminar = np.zeros_like(v)
    re = np.empty_like(v)
    v_abs = np.abs(v)
    for i in range(v.shape[0]):
        if v_abs[i] < 1e-6:
            re[i] = np.divide(rho[i] * 1e-6 * d[i], eta[i])
        else:
            re[i] = np.divide(rho[i] * v_abs[i] * d[i], eta[i])
        if v[i] != 0:
            lambda_laminar[i] = 64 / re[i]
        lambda_nikuradse[i] = np.divide(1, (-2 * np.log10(k[i] / (3.71 * d[i]))) ** 2)
    return re, lambda_laminar, lambda_nikuradse


@njit
def colebrook_nb(re, d, k, lambda_nikuradse, dummy, max_iter):
    lambda_cb = lambda_nikuradse.copy()
    lambda_cb_old = np.empty_like(lambda_nikuradse)
    converged = False
    niter = 0
    factor = np.log(10) * 2.51

    # Inner Newton-loop for calculation of lambda
    while not converged and niter < max_iter:
        for i in range(lambda_cb.shape[0]):
            sqt = np.sqrt(lambda_cb[i])
            add_val = np.divide(k[i], (3.71 * d[i]))
            sqt_div = np.divide(1, sqt)
            re_div = np.divide(1, re[i])
            sqt_div3 = sqt_div ** 3

            f = sqt_div + 2 * np.log10(2.51 * re_div * sqt_div + add_val)
            df_dlambda_cb = - 0.5 * sqt_div3 - 2.51 * re_div * sqt_div3 * np.divide(
                re[i] * sqt + add_val, factor)
            x = - f / df_dlambda_cb

            lambda_cb_old[i] = lambda_cb[i]
            lambda_cb[i] += x

        dx = np.abs(lambda_cb - lambda_cb_old) * dummy
        error_lambda = linalg.norm(dx) / dx.shape[0]

        if error_lambda <= 1e-4:
            converged = True

        niter += 1

    return converged, lambda_cb
