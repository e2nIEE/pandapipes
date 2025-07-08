# Copyright (c) 2020-2025 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
import pandas as pd
import pytest

import pandapipes
from pandapipes.component_models.junction_component import Junction
from pandapipes.component_models.pipe_component import Pipe
from pandapipes.idx_node import PINIT, TINIT
from pandapipes.pf.pipeflow_setup import get_lookup
from pandapipes.properties.fluids import _add_fluid_to_net
from pandapipes.test import data_path


def create_source_2pipes_sink(
    net,
    t_source_k,
    t_junctions_k,
    mdot_kg_per_s,
    length_km,
    diameter_m,
    pipe_name_suffix="",
    k_mm=0.1,
):
    j1, j2, j3 = pandapipes.create_junctions(
        net, nr_junctions=3, pn_bar=1, tfluid_k=t_junctions_k
    )
    pandapipes.create_pipe_from_parameters(
        net,
        from_junction=j1,
        to_junction=j2,
        name="pipe1" + pipe_name_suffix,
        length_km=length_km,
        diameter_m=diameter_m,
        k_mm=k_mm,
    )
    pandapipes.create_pipe_from_parameters(
        net,
        from_junction=j2,
        to_junction=j3,
        name="pipe2" + pipe_name_suffix,
        length_km=length_km,
        diameter_m=diameter_m,
        k_mm=k_mm,
    )
    pandapipes.create_ext_grid(net, junction=j1, p_bar=10, t_k=t_source_k)
    pandapipes.create_sink(net, junction=j3, mdot_kg_per_s=mdot_kg_per_s)


@pytest.mark.parametrize("use_numba", [True, False])
def test_gas_internal_nodes(use_numba):
    """

    :return:
    :rtype:
    """
    net = pandapipes.create_empty_network("net", add_stdtypes=False)
    d = 209.1e-3
    pandapipes.create_junction(net, pn_bar=51, tfluid_k=285.15)
    pandapipes.create_junction(net, pn_bar=51, tfluid_k=285.15)
    pandapipes.create_pipe_from_parameters(net, 0, 1, 12.0, d, k_mm=0.5, sections=12)
    pandapipes.create_ext_grid(net, 0, p_bar=51 - 1.01325, t_k=285.15, type="pt")
    pandapipes.create_sink(net, 1, mdot_kg_per_s=0.82752 * 45000 / 3600)
    _add_fluid_to_net(
        net,
        pandapipes.create_constant_fluid(
            name="natural_gas",
            fluid_type="gas",
            viscosity=11.93e-6,
            heat_capacity=2185,
            compressibility=1,
            der_compressibility=0,
            density=0.82752,
        ),
    )
    max_iter_hyd = 6 if use_numba else 6
    pandapipes.pipeflow(
        net,
        max_iter_hyd=max_iter_hyd,
        stop_condition="tol",
        friction_model="nikuradse",
        transient=False,
        nonlinear_method="automatic",
        tol_p=1e-4,
        tol_m=1e-4,
        use_numba=use_numba,
    )

    pipe_results = Pipe.get_internal_results(net, [0])

    data = pd.read_csv(
        os.path.join(data_path, "gas_sections_an.csv"),
        sep=";",
        header=0,
        keep_default_na=False,
    )
    p_an = data["p1"] / 1e5
    v_an = data["v"]
    v_an = v_an.drop([0])

    pipe_p_data_idx = np.where(pipe_results["PINIT"][:, 0] == 0)
    pipe_v_data_idx = np.where(pipe_results["VINIT_MEAN"][:, 0] == 0)
    pipe_p_data = pipe_results["PINIT"][pipe_p_data_idx, 1]
    pipe_v_data = pipe_results["VINIT_MEAN"][pipe_v_data_idx, 1]

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


@pytest.mark.parametrize("use_numba", [True, False])
def test_temperature_internal_nodes_single_pipe(use_numba):
    """

    :return:
    :rtype:
    """
    net = pandapipes.create_empty_network("net", add_stdtypes=False)
    d = 75e-3
    pandapipes.create_junction(net, pn_bar=5, tfluid_k=283)
    pandapipes.create_junction(net, pn_bar=5, tfluid_k=283)
    pandapipes.create_pipe_from_parameters(
        net, 0, 1, 6, d, k_mm=0.1, sections=6, u_w_per_m2k=5
    )
    pandapipes.create_ext_grid(net, 0, p_bar=5, t_k=330, type="pt")
    pandapipes.create_sink(net, 1, mdot_kg_per_s=1)

    pandapipes.create_fluid_from_lib(net, "water", overwrite=True)

    max_iter_hyd = 3 if use_numba else 3
    max_iter_therm = 4 if use_numba else 4
    pandapipes.pipeflow(
        net,
        stop_condition="tol",
        max_iter_hyd=max_iter_hyd,
        max_iter_therm=max_iter_therm,
        friction_model="nikuradse",
        mode="sequential",
        transient=False,
        nonlinear_method="automatic",
        tol_p=1e-4,
        tol_m=1e-4,
        use_numba=use_numba,
    )

    pipe_results = Pipe.get_internal_results(net, [0])

    data = pd.read_csv(
        os.path.join(data_path, "Temperature_one_pipe_an.csv"),
        sep=";",
        header=0,
        keep_default_na=False,
    )
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


@pytest.mark.parametrize("use_numba", [True, False])
def test_temperature_internal_nodes_tee_2ab_1zu(use_numba):
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

    pandapipes.create_pipe_from_parameters(net, j0, j1, 2.5, d, k_mm=0.1, u_w_per_m2k=5)
    pandapipes.create_pipe_from_parameters(net, j1, j2, 2.5, d, k_mm=0.1, u_w_per_m2k=5)
    pandapipes.create_pipe_from_parameters(net, j1, j3, 2.5, d, k_mm=0.1, u_w_per_m2k=5)

    pandapipes.create_fluid_from_lib(net, "water", overwrite=True)

    max_iter_hyd = 4 if use_numba else 4
    max_iter_therm = 4 if use_numba else 4
    pandapipes.pipeflow(
        net,
        stop_condition="tol",
        max_iter_hyd=max_iter_hyd,
        max_iter_therm=max_iter_therm,
        friction_model="nikuradse",
        mode="sequential",
        transient=False,
        nonlinear_method="automatic",
        tol_p=1e-4,
        tol_m=1e-4,
        use_numba=use_numba,
    )

    data = pd.read_csv(
        os.path.join(data_path, "Temperature_tee_2ab_1zu_an.csv"),
        sep=";",
        header=0,
        keep_default_na=False,
    )
    temp_an = data["T"]

    temp_pandapipes = net.res_junction["t_k"]
    temp_diff = np.abs(1 - temp_pandapipes / temp_an)

    assert np.all(temp_diff < 0.01)


@pytest.mark.parametrize("use_numba", [True, False])
def test_temperature_internal_nodes_tee_2zu_1ab(use_numba):
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

    pandapipes.create_pipe_from_parameters(
        net, j0, j2, 2.5, d, k_mm=0.1, sections=3, u_w_per_m2k=5
    )
    pandapipes.create_pipe_from_parameters(
        net, j1, j2, 2.5, d, k_mm=0.1, sections=3, u_w_per_m2k=5
    )
    pandapipes.create_pipe_from_parameters(
        net, j2, j3, 2.5, d, k_mm=0.1, sections=3, u_w_per_m2k=5
    )
    pandapipes.create_ext_grid(net, j0, p_bar=5, t_k=350, type="pt")
    pandapipes.create_ext_grid(net, j1, p_bar=5, t_k=350, type="pt")
    pandapipes.create_sink(net, j3, mdot_kg_per_s=1)

    pandapipes.create_fluid_from_lib(net, "water", overwrite=True)

    max_iter_hyd = 3 if use_numba else 3
    max_iter_therm = 4 if use_numba else 4
    pandapipes.pipeflow(
        net,
        stop_condition="tol",
        max_iter_hyd=max_iter_hyd,
        max_iter_therm=max_iter_therm,
        friction_model="nikuradse",
        mode="sequential",
        transient=False,
        nonlinear_method="automatic",
        tol_p=1e-4,
        tol_m=1e-4,
        use_numba=use_numba,
    )

    data = pd.read_csv(
        os.path.join(data_path, "Temperature_tee_2zu_1ab_an.csv"),
        sep=";",
        header=0,
        keep_default_na=False,
    )
    temp_an = data["T"]

    temp_pandapipes = net.res_junction["t_k"]
    temp_diff = np.abs(1 - temp_pandapipes / temp_an)

    assert np.all(temp_diff < 0.01)


@pytest.mark.parametrize("use_numba", [True, False])
def test_temperature_internal_nodes_tee_2zu_1ab_direction_changed(use_numba):
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

    pandapipes.create_pipe_from_parameters(
        net, j0, j2, 2.5, d, k_mm=0.1, sections=5, u_w_per_m2k=5
    )
    pandapipes.create_pipe_from_parameters(
        net, j2, j1, 2.5, d, k_mm=0.1, sections=5, u_w_per_m2k=5
    )
    pandapipes.create_pipe_from_parameters(
        net, j2, j3, 2.5, d, k_mm=0.1, sections=5, u_w_per_m2k=5
    )

    pandapipes.create_fluid_from_lib(net, "water", overwrite=True)

    max_iter_hyd = 6 if use_numba else 6
    max_iter_therm = 4 if use_numba else 4
    pandapipes.pipeflow(
        net,
        stop_condition="tol",
        max_iter_hyd=max_iter_hyd,
        max_iter_therm=max_iter_therm,
        friction_model="nikuradse",
        mode="sequential",
        transient=False,
        nonlinear_method="automatic",
        tol_p=1e-4,
        tol_m=1e-4,
        use_numba=use_numba,
    )

    data = pd.read_csv(
        os.path.join(data_path, "Temperature_tee_2zu_1ab_an.csv"),
        sep=";",
        header=0,
        keep_default_na=False,
    )
    temp_an = data["T"]

    temp_pandapipes = net.res_junction["t_k"]
    temp_diff = np.abs(1 - temp_pandapipes / temp_an)

    assert np.all(temp_diff < 0.01)


@pytest.mark.parametrize("use_numba", [True, False])
def test_temperature_internal_nodes_2zu_2ab(use_numba):
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

    pandapipes.create_pipe_from_parameters(net, j0, j2, 2.5, d, k_mm=0.1, u_w_per_m2k=5)
    pandapipes.create_pipe_from_parameters(net, j1, j2, 2.5, d, k_mm=0.1, u_w_per_m2k=5)
    pandapipes.create_pipe_from_parameters(net, j2, j3, 2.5, d, k_mm=0.1, u_w_per_m2k=5)
    pandapipes.create_pipe_from_parameters(net, j2, j4, 2.5, d, k_mm=0.1, u_w_per_m2k=5)

    pandapipes.create_fluid_from_lib(net, "water", overwrite=True)

    max_iter_hyd = 3 if use_numba else 3
    max_iter_therm = 5 if use_numba else 5
    pandapipes.pipeflow(
        net,
        stop_condition="tol",
        max_iter_hyd=max_iter_hyd,
        max_iter_therm=max_iter_therm,
        friction_model="nikuradse",
        mode="sequential",
        transient=False,
        nonlinear_method="automatic",
        tol_p=1e-4,
        tol_m=1e-4,
        use_numba=use_numba,
    )

    data = pd.read_csv(
        os.path.join(data_path, "Temperature_2zu_2ab_an.csv"),
        sep=";",
        header=0,
        keep_default_na=False,
    )
    temp_an = data["T"]

    temp_pandapipes = net.res_junction["t_k"]
    temp_diff = np.abs(1 - temp_pandapipes / temp_an)

    assert np.all(temp_diff < 0.01)


@pytest.mark.parametrize("use_numba", [True, False])
def test_temperature_internal_nodes_masche_1load(use_numba):
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

    pandapipes.create_pipe_from_parameters(
        net, j0, j1, 2.5, d, k_mm=0.1, sections=6, u_w_per_m2k=5
    )
    pandapipes.create_pipe_from_parameters(
        net, j1, j2, 2.5, d, k_mm=0.1, sections=6, u_w_per_m2k=5
    )
    pandapipes.create_pipe_from_parameters(
        net, j1, j3, 2.5, d, k_mm=0.1, sections=6, u_w_per_m2k=5
    )
    pandapipes.create_pipe_from_parameters(
        net, j3, j2, 2.5, d, k_mm=0.1, sections=6, u_w_per_m2k=5
    )

    pandapipes.create_ext_grid(net, j0, p_bar=5, t_k=350, type="pt")
    pandapipes.create_sink(net, j2, mdot_kg_per_s=1)

    pandapipes.create_fluid_from_lib(net, "water", overwrite=True)

    max_iter_hyd = 3 if use_numba else 3
    max_iter_therm = 5 if use_numba else 5
    pandapipes.pipeflow(
        net,
        stop_condition="tol",
        max_iter_hyd=max_iter_hyd,
        max_iter_therm=max_iter_therm,
        friction_model="nikuradse",
        mode="sequential",
        transient=False,
        nonlinear_method="automatic",
        tol_p=1e-4,
        tol_m=1e-4,
        use_numba=use_numba,
    )

    data = pd.read_csv(
        os.path.join(data_path, "Temperature_masche_1load_an.csv"),
        sep=";",
        header=0,
        keep_default_na=False,
    )
    temp_an = data["T"]

    temp_pandapipes = net.res_junction["t_k"]
    temp_diff = np.abs(1 - temp_pandapipes / temp_an)

    assert np.all(temp_diff < 0.01)


@pytest.mark.parametrize("use_numba", [True, False])
def test_temperature_internal_nodes_masche_1load_changed_direction(use_numba):
    """

    :return:
    :rtype:
    """
    net = pandapipes.create_empty_network("net", add_stdtypes=False)
    d = 75e-3
    j0 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283)
    j2 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283)
    j3 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283)

    pandapipes.create_pipe_from_parameters(
        net, j0, j2, 2.5, d, k_mm=0.1, sections=5, u_w_per_m2k=5
    )
    pandapipes.create_pipe_from_parameters(
        net, j0, j3, 2.5, d, k_mm=0.1, sections=5, u_w_per_m2k=5
    )
    pandapipes.create_pipe_from_parameters(
        net, j3, j2, 2.5, d, k_mm=0.1, sections=5, u_w_per_m2k=5
    )

    pandapipes.create_fluid_from_lib(net, "water", overwrite=True)

    pandapipes.create_ext_grid(net, j0, p_bar=5, t_k=350, type="pt")
    pandapipes.create_sink(net, j3, mdot_kg_per_s=1)

    max_iter_hyd = 5 if use_numba else 5
    max_iter_therm = 4 if use_numba else 4
    pandapipes.pipeflow(
        net,
        stop_condition="tol",
        max_iter_hyd=max_iter_hyd,
        max_iter_therm=max_iter_therm,
        friction_model="nikuradse",
        mode="sequential",
        transient=False,
        nonlinear_method="automatic",
        tol_p=1e-4,
        tol_m=1e-4,
        use_numba=use_numba,
    )

    data = pd.read_csv(
        os.path.join(data_path, "Temperature_masche_1load_direction_an.csv"),
        sep=";",
        header=0,
        keep_default_na=False,
    )
    temp_an = data["T"]

    temp_pandapipes = net.res_junction["t_k"]
    temp_diff = np.abs(1 - temp_pandapipes / temp_an)

    assert np.all(temp_diff < 0.01)


def test_example_hot_water():
    mdot_kg_per_s = 10
    temps_k = [300, 350]
    d_pipe = 0.1
    l_pipe = 1

    net = pandapipes.create_empty_network("network", fluid="water")

    for t_fluid in temps_k:
        for t_j_k in temps_k:
            create_source_2pipes_sink(
                net,
                t_source_k=t_fluid,
                t_junctions_k=t_j_k,
                pipe_name_suffix=f"_tj_{t_j_k}_tf_{t_fluid}",
                mdot_kg_per_s=mdot_kg_per_s,
                diameter_m=d_pipe,
                length_km=l_pipe,
            )

    pandapipes.pipeflow(net, mode="bidirectional", friction_model="colebrook")

    dp = net.res_pipe.p_from_bar - net.res_pipe.p_to_bar

    # Density (for velocity calculation) is calculated at 273.15 K, which is not the same as the actual temperature of the fluid
    velocities_equal = net.res_pipe["v_mean_m_per_s"].nunique() == 1
    assert not velocities_equal
    rho_from_v_mean = mdot_kg_per_s / (net.res_pipe.loc[:, "v_mean_m_per_s"] * np.pi * d_pipe**2 / 4).to_numpy()
    assert np.allclose(rho_from_v_mean[[0, 4]], net.fluid.get_density(temps_k))

    # dp is really calculated from v_mean, the default density and the various lambdas
    dp_calc = (
        net.res_pipe.loc[:, "lambda"]
        * l_pipe
        * 1000
        / d_pipe
        * 0.5
        * rho_from_v_mean
        * net.res_pipe.loc[:, "v_mean_m_per_s"] ** 2
        / 10**5
    )
    assert np.allclose(dp, dp_calc)


if __name__ == "__main__":
    pytest.main(
        [r"pandapipes/test/pipflow_internals/test_pipeflow_analytic_comparison.py"]
    )
