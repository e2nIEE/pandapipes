# Copyright (c) 2020-2026 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

"""
Regression test for ``calc_der_lambda`` (dlambda/dm): all three friction models ("nikuradse"/
default, "swamee-jain", "colebrook") previously returned the same value for m < 0 as for m > 0
instead of flipping sign (missing sign(m)/np.abs(m) chain-rule factors), and "colebrook" was
additionally missing the leading minus sign from the implicit function theorem
(dlambda/dm = -dF/dm / dF/dlambda). See git history for the derivations.
"""

import numpy as np
import pytest

from pandapipes.pf.derivative_calculation import calc_lambda, calc_der_lambda

EPS = 1e-6


@pytest.mark.parametrize("friction_model", ["nikuradse", "swamee-jain", "colebrook"])
@pytest.mark.parametrize("m0", [0.8, -0.8, 0.05, -0.05, 2.0, -2.0])
def test_calc_der_lambda_matches_finite_difference(friction_model, m0):
    d = np.array([0.12])
    l = np.array([800.])
    k = np.array([0.1e-3])
    eta = np.array([1e-3])
    opts = {"use_numba": False, "max_iter_colebrook": 100, "tolerance_colebrook": 1e-6}

    def lambd_of_m(m):
        m = np.array([m])
        area = np.pi * (d / 2) ** 2
        lambd, _ = calc_lambda(m, eta, d, k, False, friction_model, l, opts, area)
        return lambd[0]

    fd = (lambd_of_m(m0 + EPS) - lambd_of_m(m0 - EPS)) / (2 * EPS)

    m = np.array([m0])
    area = np.pi * (d / 2) ** 2
    lambd, re = calc_lambda(m, eta, d, k, False, friction_model, l, opts, area)
    der_lambda = calc_der_lambda(m, eta, d, k, friction_model, lambd, area, re, l)

    assert np.isclose(der_lambda[0], fd, rtol=1e-4)


@pytest.mark.parametrize("friction_model", ["nikuradse", "swamee-jain", "colebrook"])
def test_calc_der_lambda_flips_sign_for_negative_m(friction_model):
    # a symmetric physical setup (same |m|) must give equal-magnitude, opposite-sign derivatives -
    # this is the exact class of bug that motivated the finite-difference test above
    d = np.array([0.12])
    l = np.array([800.])
    k = np.array([0.1e-3])
    eta = np.array([1e-3])
    opts = {"use_numba": False, "max_iter_colebrook": 100, "tolerance_colebrook": 1e-6}
    area = np.pi * (d / 2) ** 2

    def der_at(m0):
        m = np.array([m0])
        lambd, re = calc_lambda(m, eta, d, k, False, friction_model, l, opts, area)
        return calc_der_lambda(m, eta, d, k, friction_model, lambd, area, re, l)[0]

    assert np.isclose(der_at(0.8), -der_at(-0.8), rtol=1e-8)
