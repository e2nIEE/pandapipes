# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
import pandapipes
import pandas as pd
import pytest
from pandapipes.test.pipeflow_internals import internals_data_path


def test_p_type():
    """

    :return:
    :rtype:
    """
    net = pandapipes.create_empty_network("net")
    d = 75e-3
    pandapipes.create_junction(net, pn_bar=5, tfluid_k=293.15)
    pandapipes.create_junction(net, pn_bar=5, tfluid_k=293.15)
    pandapipes.create_pipe_from_parameters(net, 0, 1, 10, diameter_m=d, k_mm=0.1, sections=1)
    pandapipes.create_ext_grid(net, 0, p_bar=5, t_k=285.15, type="p")
    pandapipes.create_sink(net, 1, mdot_kg_per_s=1)
    pandapipes.create_fluid_from_lib(net, name="water")
    pandapipes.pipeflow(net, stop_condition="tol", iter=70, friction_model="nikuradse",
                        transient=False, nonlinear_method="automatic", tol_p=1e-4, tol_v=1e-4)

    data = pd.read_csv(os.path.join(internals_data_path, "ext_grid_p.csv"),
                       sep=';', header=0, keep_default_na=False)
    p_comp = data["p"]
    p_pandapipes = net.res_junction["p_bar"][0]

    p_diff = np.abs(1 - p_pandapipes / p_comp.loc[0])

    assert np.all(p_diff < 0.01)


def test_t_type_single_pipe():
    """

    :return:
    :rtype:
    """
    net = pandapipes.create_empty_network("net")
    d = 75e-3

    j0 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283)
    j1 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283)
    pandapipes.create_ext_grid(net, j0, 5, 645, type="pt")
    pandapipes.create_sink(net, j1, 1)
    pandapipes.create_pipe_from_parameters(net, j0, j1, 6, diameter_m=d, k_mm=.1, sections=1,
                                           alpha_w_per_m2k=5)

    pandapipes.create_fluid_from_lib(net, "water", overwrite=True)
    pandapipes.pipeflow(net, stop_condition="tol", iter=70, friction_model="nikuradse",
                        transient=False, nonlinear_method="automatic", tol_p=1e-4,
                        tol_v=1e-4, mode="all")

    temp = net.res_junction.t_k.values

    net2 = pandapipes.create_empty_network("net")
    d = 75e-3

    j0 = pandapipes.create_junction(net2, pn_bar=5, tfluid_k=283)
    j1 = pandapipes.create_junction(net2, pn_bar=5, tfluid_k=283)
    pandapipes.create_ext_grid(net2, j0, 5, 645, type="p")
    pandapipes.create_ext_grid(net2, j1, 100, 323.15, type="t")
    pandapipes.create_sink(net2, j1, 1)

    pandapipes.create_pipe_from_parameters(net2, j0, j1, 6, diameter_m=d, k_mm=.1, sections=1,
                                           alpha_w_per_m2k=5)
    pandapipes.create_fluid_from_lib(net2, "water", overwrite=True)
    pandapipes.pipeflow(net2, stop_condition="tol", iter=70, friction_model="nikuradse",
                        transient=False, nonlinear_method="automatic", tol_p=1e-4, tol_v=1e-4,
                        mode="all")

    temp2 = net2.res_junction.t_k.values

    temp_diff = np.abs(1 - temp / temp2)

    assert np.all(temp_diff < 0.01)


def test_t_type_tee():
    """

    :return:
    :rtype:
    """
    net = pandapipes.create_empty_network("net")
    d = 75e-3

    j0 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=300)
    j1 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=300)
    j2 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=300)
    j3 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=300)
    pandapipes.create_ext_grid(net, j0, 5, 645, type="p")
    pandapipes.create_sink(net, j2, 1)
    pandapipes.create_sink(net, j3, 1)
    pandapipes.create_ext_grid(net, j2, 5, 310, type="t")

    pandapipes.create_pipe_from_parameters(net, j0, j1, 6, diameter_m=d, k_mm=.1, sections=1,
                                           alpha_w_per_m2k=5)
    pandapipes.create_pipe_from_parameters(net, j1, j2, 2.5, diameter_m=d, k_mm=.1, sections=1,
                                           alpha_w_per_m2k=5)
    pandapipes.create_pipe_from_parameters(net, j1, j3, 2.5, diameter_m=d, k_mm=.1, sections=1,
                                           alpha_w_per_m2k=5)

    pandapipes.create_fluid_from_lib(net, "water", overwrite=True)
    pandapipes.pipeflow(net, stop_condition="tol", iter=70, friction_model="nikuradse",
                        transient=False, nonlinear_method="automatic", tol_p=1e-4,
                        tol_v=1e-4, mode="all")

    temp = net.res_junction.t_k.values

    net2 = pandapipes.create_empty_network("net")
    d = 75e-3

    j0 = pandapipes.create_junction(net2, pn_bar=5, tfluid_k=300)
    j1 = pandapipes.create_junction(net2, pn_bar=5, tfluid_k=300)
    j2 = pandapipes.create_junction(net2, pn_bar=5, tfluid_k=300)
    j3 = pandapipes.create_junction(net2, pn_bar=5, tfluid_k=300)
    pandapipes.create_ext_grid(net2, j0, 5, 380.445, type="pt")
    pandapipes.create_sink(net2, j2, 1)
    pandapipes.create_sink(net2, j3, 1)

    pandapipes.create_pipe_from_parameters(net2, j0, j1, 6, diameter_m=d, k_mm=.1, sections=1,
                                           alpha_w_per_m2k=5)
    pandapipes.create_pipe_from_parameters(net2, j1, j2, 2.5, diameter_m=d, k_mm=.1, sections=1,
                                           alpha_w_per_m2k=5)
    pandapipes.create_pipe_from_parameters(net2, j1, j3, 2.5, diameter_m=d, k_mm=.1, sections=1,
                                           alpha_w_per_m2k=5)

    pandapipes.create_fluid_from_lib(net2, "water", overwrite=True)
    pandapipes.pipeflow(net2, stop_condition="tol", iter=70, friction_model="nikuradse",
                        transient=False, nonlinear_method="automatic", tol_p=1e-4, tol_v=1e-4,
                        mode="all")

    temp2 = net2.res_junction.t_k.values

    temp_diff = np.abs(1 - temp / temp2)

    assert np.all(temp_diff < 0.01)


def test_t_type_tee_2zu_2ab():
    """

    :return:
    :rtype:
    """
    net = pandapipes.create_empty_network("net")
    d = 75e-3

    j0 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=300)
    j1 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=300)
    j2 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=300)
    j3 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=300)
    j4 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=300)
    pandapipes.create_ext_grid(net, j0, 5, 645, type="p")
    pandapipes.create_ext_grid(net, j1, 5, 645, type="p")
    pandapipes.create_sink(net, j3, 1)
    pandapipes.create_sink(net, j4, 1)
    pandapipes.create_ext_grid(net, j1, 5, 645, type="t")
    pandapipes.create_ext_grid(net, j0, 5, 645, type="t")

    pandapipes.create_pipe_from_parameters(net, j0, j2, 6, diameter_m=d, k_mm=.1, sections=1,
                                           alpha_w_per_m2k=5)
    pandapipes.create_pipe_from_parameters(net, j1, j2, 2.5, diameter_m=d, k_mm=.1, sections=5,
                                           alpha_w_per_m2k=5)
    pandapipes.create_pipe_from_parameters(net, j2, j3, 2.5, diameter_m=d, k_mm=.1, sections=5,
                                           alpha_w_per_m2k=5)
    pandapipes.create_pipe_from_parameters(net, j2, j4, 2.5, diameter_m=d, k_mm=.1, sections=5,
                                           alpha_w_per_m2k=5)

    pandapipes.create_fluid_from_lib(net, "water", overwrite=True)
    pandapipes.pipeflow(net, stop_condition="tol", iter=3, friction_model="nikuradse",
                        transient=False, nonlinear_method="automatic", tol_p=1e-4,
                        tol_v=1e-4, mode="all")

    temp = net.res_junction.t_k.values

    net2 = pandapipes.create_empty_network("net")
    d = 75e-3

    j0 = pandapipes.create_junction(net2, pn_bar=5, tfluid_k=300)
    j1 = pandapipes.create_junction(net2, pn_bar=5, tfluid_k=300)
    j2 = pandapipes.create_junction(net2, pn_bar=5, tfluid_k=300)
    j3 = pandapipes.create_junction(net2, pn_bar=5, tfluid_k=300)
    j4 = pandapipes.create_junction(net2, pn_bar=5, tfluid_k=300)
    pandapipes.create_ext_grid(net2, j0, 5, 645, type="pt")
    pandapipes.create_ext_grid(net2, j1, 5, 645, type="pt")
    pandapipes.create_sink(net2, j3, 1)
    pandapipes.create_sink(net2, j4, 1)

    pandapipes.create_pipe_from_parameters(net2, j0, j2, 6, diameter_m=d, k_mm=.1, sections=1,
                                           alpha_w_per_m2k=5)
    pandapipes.create_pipe_from_parameters(net2, j1, j2, 2.5, diameter_m=d, k_mm=.1, sections=5,
                                           alpha_w_per_m2k=5)
    pandapipes.create_pipe_from_parameters(net2, j2, j3, 2.5, diameter_m=d, k_mm=.1, sections=5,
                                           alpha_w_per_m2k=5)
    pandapipes.create_pipe_from_parameters(net2, j2, j4, 2.5, diameter_m=d, k_mm=.1, sections=5,
                                           alpha_w_per_m2k=5)

    pandapipes.create_fluid_from_lib(net2, "water", overwrite=True)
    pandapipes.pipeflow(net2, stop_condition="tol", iter=3, friction_model="nikuradse",
                        transient=False, nonlinear_method="automatic", tol_p=1e-4,
                        tol_v=1e-4, mode="all")

    temp2 = net2.res_junction.t_k.values

    temp_diff = np.abs(1 - temp / temp2)

    assert np.all(temp_diff < 0.01)


def test_t_type_tee_2zu_2ab2():
    """

    :return:
    :rtype:
    """
    net = pandapipes.create_empty_network("net")
    d = 75e-3

    j0 = pandapipes.create_junction(net, pn_bar=3, tfluid_k=300)
    j1 = pandapipes.create_junction(net, pn_bar=3, tfluid_k=300)
    j2 = pandapipes.create_junction(net, pn_bar=3, tfluid_k=300)
    j3 = pandapipes.create_junction(net, pn_bar=3, tfluid_k=300)
    j4 = pandapipes.create_junction(net, pn_bar=3, tfluid_k=300)
    pandapipes.create_ext_grid(net, j0, 5, 645, type="p")
    pandapipes.create_ext_grid(net, j1, 5, 645, type="p")
    pandapipes.create_sink(net, j3, 1)
    pandapipes.create_sink(net, j4, 1)
    pandapipes.create_ext_grid(net, j0, 5, 645, type="t")
    pandapipes.create_ext_grid(net, j4, 5, 378.83472, type="t")

    pandapipes.create_pipe_from_parameters(net, j0, j2, 2.5, diameter_m=d, k_mm=.1, sections=5,
                                           alpha_w_per_m2k=5)
    pandapipes.create_pipe_from_parameters(net, j1, j2, 2.5, diameter_m=d, k_mm=.1, sections=5,
                                           alpha_w_per_m2k=5)
    pandapipes.create_pipe_from_parameters(net, j2, j3, 2.5, diameter_m=d, k_mm=.1, sections=5,
                                           alpha_w_per_m2k=5)
    pandapipes.create_pipe_from_parameters(net, j2, j4, 2.5, diameter_m=d, k_mm=.1, sections=5,
                                           alpha_w_per_m2k=5)

    pandapipes.create_fluid_from_lib(net, "water", overwrite=True)
    pandapipes.pipeflow(net, stop_condition="tol", iter=20, friction_model="nikuradse",
                        transient=False, nonlinear_method="automatic", tol_p=1e-4,
                        tol_v=1e-4, mode="all")

    temp = net.res_junction.t_k.values

    net2 = pandapipes.create_empty_network("net")
    d = 75e-3

    j0 = pandapipes.create_junction(net2, pn_bar=5, tfluid_k=300)
    j1 = pandapipes.create_junction(net2, pn_bar=5, tfluid_k=300)
    j2 = pandapipes.create_junction(net2, pn_bar=5, tfluid_k=300)
    j3 = pandapipes.create_junction(net2, pn_bar=5, tfluid_k=300)
    j4 = pandapipes.create_junction(net2, pn_bar=5, tfluid_k=300)
    pandapipes.create_ext_grid(net2, j0, 5, 645, type="pt")
    pandapipes.create_ext_grid(net2, j1, 5, 645, type="pt")
    pandapipes.create_sink(net2, j3, 1)
    pandapipes.create_sink(net2, j4, 1)

    pandapipes.create_pipe_from_parameters(net2, j0, j2, 2.5, diameter_m=d, k_mm=.1, sections=5,
                                           alpha_w_per_m2k=5)
    pandapipes.create_pipe_from_parameters(net2, j1, j2, 2.5, diameter_m=d, k_mm=.1, sections=5,
                                           alpha_w_per_m2k=5)
    pandapipes.create_pipe_from_parameters(net2, j2, j3, 2.5, diameter_m=d, k_mm=.1, sections=5,
                                           alpha_w_per_m2k=5)
    pandapipes.create_pipe_from_parameters(net2, j2, j4, 2.5, diameter_m=d, k_mm=.1, sections=5,
                                           alpha_w_per_m2k=5)

    pandapipes.create_fluid_from_lib(net2, "water", overwrite=True)
    pandapipes.pipeflow(net2, stop_condition="tol", iter=20, friction_model="nikuradse",
                        transient=False, nonlinear_method="automatic", tol_p=1e-4,
                        tol_v=1e-4, mode="all")

    temp2 = net2.res_junction.t_k.values

    temp_diff = np.abs(1 - temp / temp2)

    assert np.all(temp_diff < 0.01)


def test_t_type_tee_2zu_2ab3():
    """

    :return:
    :rtype:
    """
    net = pandapipes.create_empty_network("net")
    d = 75e-3

    j0 = pandapipes.create_junction(net, pn_bar=3, tfluid_k=300)
    j1 = pandapipes.create_junction(net, pn_bar=3, tfluid_k=300)
    j2 = pandapipes.create_junction(net, pn_bar=3, tfluid_k=300)
    j3 = pandapipes.create_junction(net, pn_bar=3, tfluid_k=300)
    j4 = pandapipes.create_junction(net, pn_bar=3, tfluid_k=300)
    pandapipes.create_ext_grid(net, j0, 5, 645, type="p")
    pandapipes.create_ext_grid(net, j2, 5, 645, type="p")
    pandapipes.create_sink(net, j3, 1)
    pandapipes.create_sink(net, j4, 1)
    pandapipes.create_ext_grid(net, j2, 5, 645, type="t")
    pandapipes.create_ext_grid(net, j4, 5, 378.83472, type="t")

    pandapipes.create_pipe_from_parameters(net, j0, j1, 2.5, diameter_m=d, k_mm=.1, sections=5,
                                           alpha_w_per_m2k=5)
    pandapipes.create_pipe_from_parameters(net, j2, j1, 2.5, diameter_m=d, k_mm=.1, sections=5,
                                           alpha_w_per_m2k=5)
    pandapipes.create_pipe_from_parameters(net, j1, j3, 2.5, diameter_m=d, k_mm=.1, sections=5,
                                           alpha_w_per_m2k=5)
    pandapipes.create_pipe_from_parameters(net, j1, j4, 2.5, diameter_m=d, k_mm=.1, sections=5,
                                           alpha_w_per_m2k=5)

    pandapipes.create_fluid_from_lib(net, "water", overwrite=True)
    pandapipes.pipeflow(net, stop_condition="tol", iter=10, friction_model="nikuradse",
                        transient=False, nonlinear_method="automatic", tol_p=1e-4,
                        tol_v=1e-4, mode="all")

    temp = net.res_junction.t_k.values

    net2 = pandapipes.create_empty_network("net")
    d = 75e-3

    j0 = pandapipes.create_junction(net2, pn_bar=5, tfluid_k=300)
    j1 = pandapipes.create_junction(net2, pn_bar=5, tfluid_k=300)
    j2 = pandapipes.create_junction(net2, pn_bar=5, tfluid_k=300)
    j3 = pandapipes.create_junction(net2, pn_bar=5, tfluid_k=300)
    j4 = pandapipes.create_junction(net2, pn_bar=5, tfluid_k=300)
    pandapipes.create_ext_grid(net2, j0, 5, 645, type="pt")
    pandapipes.create_ext_grid(net2, j2, 5, 645, type="pt")
    pandapipes.create_sink(net2, j3, 1)
    pandapipes.create_sink(net2, j4, 1)

    pandapipes.create_pipe_from_parameters(net2, j0, j1, 2.5, diameter_m=d, k_mm=.1, sections=5,
                                           alpha_w_per_m2k=5)
    pandapipes.create_pipe_from_parameters(net2, j2, j1, 2.5, diameter_m=d, k_mm=.1, sections=5,
                                           alpha_w_per_m2k=5)
    pandapipes.create_pipe_from_parameters(net2, j1, j3, 2.5, diameter_m=d, k_mm=.1, sections=5,
                                           alpha_w_per_m2k=5)
    pandapipes.create_pipe_from_parameters(net2, j1, j4, 2.5, diameter_m=d, k_mm=.1, sections=5,
                                           alpha_w_per_m2k=5)

    pandapipes.create_fluid_from_lib(net2, "water", overwrite=True)
    pandapipes.pipeflow(net2, stop_condition="tol", iter=10, friction_model="nikuradse",
                        transient=False, nonlinear_method="automatic", tol_p=1e-4,
                        tol_v=1e-4, mode="all")

    temp2 = net2.res_junction.t_k.values

    temp_diff = np.abs(1 - temp / temp2)

    assert np.all(temp_diff < 0.01)
