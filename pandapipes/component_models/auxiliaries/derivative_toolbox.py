# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

from numpy import linalg
import numpy as np


def calc_lambda(v, eta, rho, d, k, gas_mode, friction_model, dummy):
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
    :return:
    :rtype:
    """
    v_corr = np.where(np.abs(v) < 1e-6, 1e-6 * np.sign(v), v)
    lambda_laminar = np.zeros_like(v)

    re = np.abs(rho * v_corr * d / eta)
    lambda_laminar[v != 0] = 64 / re[v != 0]

    if gas_mode:
        lambda_nikuradse = 1 / (2 * np.log10(d / k) + 1.14) ** 2
    else:
        lambda_nikuradse = 1 / (-2 * np.log10(k / (3.71 * d))) ** 2

    if friction_model == "colebrook":
        lambda_colebrook = colebrook(re, d, k, lambda_nikuradse, dummy)
        return lambda_colebrook, re
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

    lambda_laminar_der = -(64 * eta) / (rho * v_corr ** 2 * d)

    if friction_model == "colebrook":
        b_term = 2.51 * eta / (rho * d * np.sqrt(lambda_pipe) * v_corr) + k / (3.71 * d)

        df_dv = -2 * (2.51 * eta / (rho * np.sqrt(lambda_pipe) * v_corr ** 2)) \
                / (np.log(10) * b_term)

        df_dlambda = -0.5 * lambda_pipe ** (-3 / 2) - ((2.51 * eta) / (rho * d * v_corr)) \
                     * lambda_pipe ** (-3 / 2) * 1 / (np.log(10) * b_term)

        lambda_colebrook_der = df_dv / df_dlambda

        return lambda_colebrook_der

    else:
        return lambda_laminar_der


def colebrook(re, d, k, lambda_nikuradse, dummy):
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
    :return: lambda_cb
    :rtype:
    """
    lambda_cb = lambda_nikuradse
    converged = False
    error_lambda = []
    niter = 0

    # Inner Newton-loop for calculation of lambda
    while not converged:
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

    return lambda_cb
