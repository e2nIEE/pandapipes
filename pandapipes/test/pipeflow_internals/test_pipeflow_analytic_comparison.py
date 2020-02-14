# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
import pandapipes
import pandas as pd
import pytest
from pandapipes.component_models import Pipe, Junction
from pandapipes.idx_node import PINIT, TINIT
from pandapipes.pipeflow_setup import get_lookup
from pandapipes.test.pipeflow_internals import internals_data_path


def test_gas_internal_nodes():
    """

    :return:
    :rtype:
    """
    net = pandapipes.create_empty_network("net", add_stdtypes=False)
    d = 209.1e-3
    pandapipes.create_junction(net, pn_bar=51, tfluid_k=285.15)
    pandapipes.create_junction(net, pn_bar=51, tfluid_k=285.15)
    pandapipes.create_pipe_from_parameters(net, 0, 1, 12.0, d, k_mm=.5, sections=12)
    pandapipes.create_ext_grid(net, 0, p_bar=51 - 1.01325, t_k=285.15, type="pt")
    pandapipes.create_sink(net, 1, mdot_kg_per_s=0.82752 * 45000 / 3600)
    pandapipes.add_fluid_to_net(net, pandapipes.create_constant_fluid(
        name="natural_gas", fluid_type="gas", viscosity=11.93e-6, heat_capacity=2185,
        compressibility=1, der_compressibility=0, density=0.82752
    ))
    pandapipes.pipeflow(net, stop_condition="tol", iter=70, friction_model="nikuradse",
                        transient=False, nonlinear_method="automatic", tol_p=1e-4, tol_v=1e-4)

    pipe_results = Pipe.get_internal_results(net, [0])

    data = pd.read_csv(os.path.join(internals_data_path, "gas_sections_an.csv"), sep=';', header=0,
                       keep_default_na=False)
    p_an = data["p1"] / 1e5
    v_an = data["v"]
    v_an = v_an.drop([0])

    pipe_p_data_idx = np.where(pipe_results["PINIT"][:, 0] == 0)
    pipe_v_data_idx = np.where(pipe_results["VINIT"][:, 0] == 0)
    pipe_p_data = pipe_results["PINIT"][pipe_p_data_idx, 1]
    pipe_v_data = pipe_results["VINIT"][pipe_v_data_idx, 1]

    node_pit = net["_pit"]["node"]

    junction_idx_lookup = get_lookup(net, "node", "index")[Junction.table_name()]
    from_junction_nodes = junction_idx_lookup[net["pipe"]["from_junction"].values]
    to_junction_nodes = junction_idx_lookup[net["pipe"]["to_junction"].values]

    p_pandapipes = np.zeros(len(pipe_p_data[0]) + 2)
    p_pandapipes[0] = node_pit[from_junction_nodes[0], PINIT]
    p_pandapipes[1:-1] = pipe_p_data[:]
    p_pandapipes[-1] = node_pit[to_junction_nodes[0], PINIT]
    p_pandapipes = p_pandapipes + 1.01325
    v_pandapipes = pipe_v_data[0, :]

    p_diff = np.abs(1 - p_pandapipes / p_an)
    v_diff = np.abs(v_an - v_pandapipes)

    assert np.all(p_diff < 0.01)
    assert np.all(v_diff < 0.4)


def test_temperature_internal_nodes_single_pipe():
    """

    :return:
    :rtype:
    """
    net = pandapipes.create_empty_network("net", add_stdtypes=False)
    d = 75e-3
    pandapipes.create_junction(net, pn_bar=5, tfluid_k=283)
    pandapipes.create_junction(net, pn_bar=5, tfluid_k=283)
    pandapipes.create_pipe_from_parameters(net, 0, 1, 6, d, k_mm=.1, sections=6, alpha_w_per_m2k=5)
    pandapipes.create_ext_grid(net, 0, p_bar=5, t_k=330, type="pt")
    pandapipes.create_sink(net, 1, mdot_kg_per_s=1)

    pandapipes.create_fluid_from_lib(net, "water", overwrite=True)

    pandapipes.pipeflow(net, stop_condition="tol", iter=3, friction_model="nikuradse",
                        mode="all", transient=False, nonlinear_method="automatic", tol_p=1e-4,
                        tol_v=1e-4)

    pipe_results = Pipe.get_internal_results(net, [0])

    data = pd.read_csv(os.path.join(internals_data_path, "Temperature_one_pipe_an.csv"), sep=';',
                       header=0, keep_default_na=False)
    temp_an = data["T"]

    pipe_temp_data_idx = np.where(pipe_results["TINIT"][:, 0] == 0)
    pipe_temp_data = pipe_results["TINIT"][pipe_temp_data_idx, 1]

    node_pit = net["_pit"]["node"]

    junction_idx_lookup = get_lookup(net, "node", "index")[Junction.table_name()]
    from_junction_nodes = junction_idx_lookup[net["pipe"]["from_junction"].values]
    to_junction_nodes = junction_idx_lookup[net["pipe"]["to_junction"].values]

    temp_pandapipes = np.zeros(len(pipe_temp_data[0]) + 2)
    temp_pandapipes[0] = node_pit[from_junction_nodes[0], TINIT]
    temp_pandapipes[1:-1] = pipe_temp_data[:]
    temp_pandapipes[-1] = node_pit[to_junction_nodes[0], TINIT]

    temp_diff = np.abs(1 - temp_pandapipes / temp_an)

    assert np.all(temp_diff < 0.01)


def test_temperature_internal_nodes_tee_2ab_1zu():
    """

    :return:
    :rtype:
    """
    net = pandapipes.create_empty_network("net", add_stdtypes=False)
    d = 75e-3
    j0 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283)
    j1 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283)
    j2 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283)
    j3 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283)
    pandapipes.create_ext_grid(net, j0, p_bar=5, t_k=350, type="pt")
    pandapipes.create_sink(net, j2, mdot_kg_per_s=1)
    pandapipes.create_sink(net, j3, mdot_kg_per_s=1)

    pandapipes.create_pipe_from_parameters(net, j0, j1, 2.5, d, k_mm=.1, alpha_w_per_m2k=5)
    pandapipes.create_pipe_from_parameters(net, j1, j2, 2.5, d, k_mm=.1, alpha_w_per_m2k=5)
    pandapipes.create_pipe_from_parameters(net, j1, j3, 2.5, d, k_mm=.1, alpha_w_per_m2k=5)

    pandapipes.create_fluid_from_lib(net, "water", overwrite=True)

    pandapipes.pipeflow(net, stop_condition="tol", iter=70, friction_model="nikuradse",
                        mode='all', transient=False, nonlinear_method="automatic", tol_p=1e-4,
                        tol_v=1e-4)

    data = pd.read_csv(os.path.join(internals_data_path, "Temperature_tee_2ab_1zu_an.csv"),
                       sep=';', header=0, keep_default_na=False)
    temp_an = data["T"]

    temp_pandapipes = net.res_junction["t_k"]
    temp_diff = np.abs(1 - temp_pandapipes / temp_an)

    assert np.all(temp_diff < 0.01)


def test_temperature_internal_nodes_tee_2zu_1ab():
    """

    :return:
    :rtype:
    """
    net = pandapipes.create_empty_network("net", add_stdtypes=False)
    d = 75e-3
    j0 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283)
    j1 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283)
    j2 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283)
    j3 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283)

    pandapipes.create_pipe_from_parameters(net, j0, j2, 2.5, d, k_mm=.1, sections=3,
                                           alpha_w_per_m2k=5)
    pandapipes.create_pipe_from_parameters(net, j1, j2, 2.5, d, k_mm=.1, sections=3,
                                           alpha_w_per_m2k=5)
    pandapipes.create_pipe_from_parameters(net, j2, j3, 2.5, d, k_mm=.1, sections=3,
                                           alpha_w_per_m2k=5)
    pandapipes.create_ext_grid(net, j0, p_bar=5, t_k=350, type="pt")
    pandapipes.create_ext_grid(net, j1, p_bar=5, t_k=350, type="pt")
    pandapipes.create_sink(net, j3, mdot_kg_per_s=1)

    pandapipes.create_fluid_from_lib(net, "water", overwrite=True)

    pandapipes.pipeflow(net, stop_condition="tol", iter=3, friction_model="nikuradse",
                        mode='all', transient=False, nonlinear_method="automatic", tol_p=1e-4,
                        tol_v=1e-4)

    data = pd.read_csv(os.path.join(internals_data_path, "Temperature_tee_2zu_1ab_an.csv"),
                       sep=';', header=0, keep_default_na=False)
    temp_an = data["T"]

    temp_pandapipes = net.res_junction["t_k"]
    temp_diff = np.abs(1 - temp_pandapipes / temp_an)

    assert np.all(temp_diff < 0.01)


def test_temperature_internal_nodes_tee_2zu_1ab_direction_changed():
    """

    :return:
    :rtype:
    """
    net = pandapipes.create_empty_network("net", add_stdtypes=False)
    d = 75e-3
    j0 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283)
    j1 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283)
    j2 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283)
    j3 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283)
    pandapipes.create_ext_grid(net, j0, p_bar=5, t_k=350, type="pt")
    pandapipes.create_ext_grid(net, j1, p_bar=5, t_k=350, type="pt")
    pandapipes.create_sink(net, j3, mdot_kg_per_s=1)

    pandapipes.create_pipe_from_parameters(net, j0, j2, 2.5, d, k_mm=.1, sections=5,
                                           alpha_w_per_m2k=5)
    pandapipes.create_pipe_from_parameters(net, j2, j1, 2.5, d, k_mm=.1, sections=5,
                                           alpha_w_per_m2k=5)
    pandapipes.create_pipe_from_parameters(net, j2, j3, 2.5, d, k_mm=.1, sections=5,
                                           alpha_w_per_m2k=5)

    pandapipes.create_fluid_from_lib(net, "water", overwrite=True)

    pandapipes.pipeflow(net, stop_condition="tol", iter=70, friction_model="nikuradse",
                        mode='all', transient=False, nonlinear_method="automatic", tol_p=1e-4,
                        tol_v=1e-4)

    data = pd.read_csv(os.path.join(internals_data_path, "Temperature_tee_2zu_1ab_an.csv"),
                       sep=';', header=0, keep_default_na=False)
    temp_an = data["T"]

    temp_pandapipes = net.res_junction["t_k"]
    temp_diff = np.abs(1 - temp_pandapipes / temp_an)

    assert np.all(temp_diff < 0.01)


def test_temperature_internal_nodes_2zu_2ab():
    """

    :return:
    :rtype:
    """
    net = pandapipes.create_empty_network("net", add_stdtypes=False)
    d = 75e-3
    j0 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283)
    j1 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283)
    j2 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283)
    j3 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283)
    j4 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283)
    pandapipes.create_ext_grid(net, j0, p_bar=5, t_k=350, type="pt")
    pandapipes.create_ext_grid(net, j1, p_bar=5, t_k=300, type="pt")
    pandapipes.create_sink(net, j3, mdot_kg_per_s=1)
    pandapipes.create_sink(net, j4, mdot_kg_per_s=1)

    pandapipes.create_pipe_from_parameters(net, j0, j2, 2.5, d, k_mm=.1, alpha_w_per_m2k=5)
    pandapipes.create_pipe_from_parameters(net, j1, j2, 2.5, d, k_mm=.1, alpha_w_per_m2k=5)
    pandapipes.create_pipe_from_parameters(net, j2, j3, 2.5, d, k_mm=.1, alpha_w_per_m2k=5)
    pandapipes.create_pipe_from_parameters(net, j2, j4, 2.5, d, k_mm=.1, alpha_w_per_m2k=5)

    pandapipes.create_fluid_from_lib(net, "water", overwrite=True)

    pandapipes.pipeflow(net, stop_condition="tol", iter=70, friction_model="nikuradse",
                        mode='all', transient=False, nonlinear_method="automatic", tol_p=1e-4,
                        tol_v=1e-4)

    data = pd.read_csv(os.path.join(internals_data_path, "Temperature_2zu_2ab_an.csv"), sep=';',
                       header=0, keep_default_na=False)
    temp_an = data["T"]

    temp_pandapipes = net.res_junction["t_k"]
    temp_diff = np.abs(1 - temp_pandapipes / temp_an)

    assert np.all(temp_diff < 0.01)


def test_temperature_internal_nodes_masche_1load():
    """

    :return:
    :rtype:
    """
    net = pandapipes.create_empty_network("net", add_stdtypes=False)
    d = 75e-3
    j0 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283)
    j1 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283)
    j2 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283)
    j3 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283)

    pandapipes.create_pipe_from_parameters(net, j0, j1, 2.5, d, k_mm=.1, sections=6,
                                           alpha_w_per_m2k=5)
    pandapipes.create_pipe_from_parameters(net, j1, j2, 2.5, d, k_mm=.1, sections=6,
                                           alpha_w_per_m2k=5)
    pandapipes.create_pipe_from_parameters(net, j1, j3, 2.5, d, k_mm=.1, sections=6,
                                           alpha_w_per_m2k=5)
    pandapipes.create_pipe_from_parameters(net, j3, j2, 2.5, d, k_mm=.1, sections=6,
                                           alpha_w_per_m2k=5)

    pandapipes.create_ext_grid(net, j0, p_bar=5, t_k=350, type="pt")
    pandapipes.create_sink(net, j2, mdot_kg_per_s=1)

    pandapipes.create_fluid_from_lib(net, "water", overwrite=True)

    pandapipes.pipeflow(net, stop_condition="tol", iter=70, friction_model="nikuradse",
                        mode='all', transient=False, nonlinear_method="automatic", tol_p=1e-4,
                        tol_v=1e-4)

    data = pd.read_csv(os.path.join(internals_data_path, "Temperature_masche_1load_an.csv"),
                       sep=';', header=0, keep_default_na=False)
    temp_an = data["T"]

    temp_pandapipes = net.res_junction["t_k"]
    temp_diff = np.abs(1 - temp_pandapipes / temp_an)

    assert np.all(temp_diff < 0.01)


def test_temperature_internal_nodes_masche_1load_changed_direction():
    """

    :return:
    :rtype:
    """
    net = pandapipes.create_empty_network("net", add_stdtypes=False)
    d = 75e-3
    j0 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283)
    j2 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283)
    j3 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283)

    pandapipes.create_pipe_from_parameters(net, j0, j2, 2.5, d, k_mm=.1, sections=5,
                                           alpha_w_per_m2k=5)
    pandapipes.create_pipe_from_parameters(net, j0, j3, 2.5, d, k_mm=.1, sections=5,
                                           alpha_w_per_m2k=5)
    pandapipes.create_pipe_from_parameters(net, j3, j2, 2.5, d, k_mm=.1, sections=5,
                                           alpha_w_per_m2k=5)

    pandapipes.create_fluid_from_lib(net, "water", overwrite=True)

    pandapipes.create_ext_grid(net, j0, p_bar=5, t_k=350, type="pt")
    pandapipes.create_sink(net, j3, mdot_kg_per_s=1)

    pandapipes.pipeflow(net, stop_condition="tol", iter=70, friction_model="nikuradse",
                        mode='all', transient=False, nonlinear_method="automatic", tol_p=1e-4,
                        tol_v=1e-4)

    data = pd.read_csv(os.path.join(internals_data_path,
                                    "Temperature_masche_1load_direction_an.csv"),
                       sep=';', header=0, keep_default_na=False)
    temp_an = data["T"]

    temp_pandapipes = net.res_junction["t_k"]
    temp_diff = np.abs(1 - temp_pandapipes / temp_an)

    assert np.all(temp_diff < 0.01)


if __name__ == "__main__":
    pytest.main([r'pandapipes/test/pipflow_internals/test_pipeflow_analytic_comparison.py'])
