# Copyright (c) 2020-2021 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

from numpy import linalg
import numpy as np


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
    v_corr = np.where(np.abs(v) < 1e-6, 1e-6 * np.sign(v), v)
    lambda_laminar = np.zeros_like(v)

    re = np.abs(rho * v_corr * d / eta)
    lambda_laminar[v != 0] = 64 / re[v != 0]

    if gas_mode:
        lambda_nikuradse = 1 / (2 * np.log10(d / k) + 1.14) ** 2
    else:
        lambda_nikuradse = 1 / (-2 * np.log10(k / (3.71 * d))) ** 2

    if friction_model == "colebrook":
        # TODO: move this import to top level if possible
        from pandapipes.pipeflow import PipeflowNotConverged
        max_iter = options.get("max_iter_colebrook", 100)
        converged, lambda_colebrook = colebrook(re, d, k, lambda_nikuradse, dummy, max_iter)
        if not converged:
            raise PipeflowNotConverged(
                "The Colebrook-White algorithm did not converge. There might be model "
                "inconsistencies. The maximum iterations can be given as 'max_iter_colebrook' "
                "argument to the pipeflow.")
        return lambda_colebrook, re
    elif friction_model == "explicit_colebrook":
        lambda_explicit = explicit_colebrook(re, k)
        return lambda_explicit, re
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

    lambda_laminar_der = -(64 * eta) / (rho * v_corr ** 2 * d)

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

def explicit_colebrook(re, k):
    """
    Function calculates the friction factor of a pipe, it is an explicit correlation of the colebrook equation
    could potentially enhance caclulation time

    :param re:
    :type re:
    :param k:
    :type k:
    :return: lambda_explicit
    :rtype:

    for more info :
    https://www.researchgate.net/publication/344081672_Review_of_new_flow_friction_equations_Constructing_Colebrook's_explicit_correlations_accurately
    """

    A = (re * k) / 8.0878
    B = np.log(re) - 0.7794
    x = A + B
    C = np.log(x)
    s1 = C * ((1 / x) - 1)
    s2 = (C / (2 * (x**2))) * (C - 2)
    s3 = (C / (6 * (x**3))) * ((2 * (C**2)) - ((9 * C) + 6))
    s4 = (C / (12 * (x**4))) * ((3 * (C**3)) - (22 * (C**2)) + (36 * C) - 12)
    s5 = (C / (60 * (x**5))) * ((12 * (C**4)) - (125 * (C**3)) + (350 * (C**2)) - (300 * C) + 60)
    y = s1 + s2 + s3 + s4 + s5
    lambda_explicit = (1 / (0.8686 * (B + y)))**2

    return lambda_explicit
