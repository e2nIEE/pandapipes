import numpy as np
import pytest

import pandapipes


@pytest.mark.parametrize("use_numba", [True, False])
def test_flow_control_simple_heat(use_numba):
    net = pandapipes.create_empty_network("net", add_stdtypes=True, fluid="water")

    j = pandapipes.create_junctions(net, 8, pn_bar=5, tfluid_k=360)
    j1, j2, j3, j4, j5, j6, j7, j8 = j

    p12, p25, p48, p74 = pandapipes.create_pipes_from_parameters(
        net, [j1, j2, j4, j7], [j2, j5, j8, j4], 0.2, 100, k_mm=0.1, u_w_per_m2k=20.,
        text_k=280)

    pandapipes.create_heat_exchanger(net, j3, j4, 0.1, 50000, 1)
    pandapipes.create_heat_exchanger(net, j6, j7, 0.1, 50000, 1)

    pandapipes.create_flow_control(net, j2, j3, 2)
    pandapipes.create_flow_control(net, j5, j6, 2, control_active=False)

    pandapipes.create_ext_grid(net, j1, p_bar=5, t_k=360, type="pt")

    pandapipes.create_sink(net, j8, 3)

    max_iter_hyd = 3 if use_numba else 3
    max_iter_therm = 7 if use_numba else 7
    pandapipes.pipeflow(net, max_iter_hyd=max_iter_hyd, max_iter_therm=max_iter_therm,
                        mode='sequential', use_numba=use_numba)

    assert np.allclose(net.res_pipe.loc[[p12, p48, p25, p74], "mdot_from_kg_per_s"], [3, 3, 1, 1])
    assert np.allclose(net.res_flow_control["mdot_from_kg_per_s"].values, [2, 1])

    j_new = pandapipes.create_junctions(net, 2, pn_bar=5, tfluid_k=360)
    pandapipes.create_flow_control(net, j_new[0], j_new[1], 2, 0.1)

    pandapipes.pipeflow(net, max_iter_hyd=max_iter_hyd, max_iter_therm=max_iter_therm,
                        mode='sequential', use_numba=use_numba, check_connectivity=True)

    assert np.allclose(net.res_pipe.loc[[p12, p48, p25, p74], "mdot_from_kg_per_s"], [3, 3, 1, 1])
    assert np.allclose(net.res_flow_control["mdot_from_kg_per_s"].values[:2], [2, 1])
    assert np.isnan(net.res_flow_control["mdot_from_kg_per_s"].values[2])


@pytest.mark.parametrize("use_numba", [True, False])
def test_flow_control_simple_gas(use_numba):
    net = pandapipes.create_empty_network("net", add_stdtypes=True, fluid="hgas")

    j = pandapipes.create_junctions(net, 8, pn_bar=1, tfluid_k=298)
    j1, j2, j3, j4, j5, j6, j7, j8 = j

    p12, p23, p34, p45, p56, p58, p67 = pandapipes.create_pipes_from_parameters(
        net, [j1, j2, j3, j4, j5, j5, j6], [j2, j3, j4, j5, j6, j8, j7], 0.2, 100, k_mm=0.1)

    pandapipes.create_flow_controls(net, [j2, j3, j4], [j6, j5, j8], [0.03, 0.02, 0.03], 0.1)

    pandapipes.create_ext_grid(net, j1, p_bar=1, t_k=298)

    pandapipes.create_sinks(net, [j3, j4, j5, j7, j8], [0.02, 0.04, 0.03, 0.04, 0.02])

    max_iter_hyd = 4 if use_numba else 4
    pandapipes.pipeflow(net, max_iter_hyd=max_iter_hyd,
                        mode="hydraulics", use_numba=use_numba)

    assert np.allclose(net.res_pipe.loc[[p12, p23, p34, p45, p56, p58, p67], "mdot_from_kg_per_s"],
                       [0.15, 0.12, 0.08, 0.01, 0.01, -0.01, 0.04])
    assert np.allclose(net.res_flow_control["mdot_from_kg_per_s"].values, [0.03, 0.02, 0.03])


@pytest.mark.parametrize("use_numba", [True, False])
def test_flow_control_simple_gas_two_eg(use_numba):
    net = pandapipes.create_empty_network("net", add_stdtypes=True, fluid="hgas")

    j = pandapipes.create_junctions(net, 4, pn_bar=1, tfluid_k=298)
    j1, j2, j3, j4 = j

    p12, p34 = pandapipes.create_pipes_from_parameters(net, [j1, j3], [j2, j4], 0.2, 100, k_mm=0.1)

    pandapipes.create_flow_control(net, j2, j3, 0.03, 0.1)

    pandapipes.create_ext_grid(net, j1, p_bar=1, t_k=298)
    pandapipes.create_ext_grid(net, j4, p_bar=0.95, t_k=298)

    pandapipes.create_sinks(net, [j2, j3], [0.02, 0.04])

    max_iter_hyd = 3 if use_numba else 3
    pandapipes.pipeflow(net, max_iter_hyd=max_iter_hyd,
                        mode="hydraulics", use_numba=use_numba)

    assert np.allclose(net.res_pipe.loc[[p12, p34], "mdot_from_kg_per_s"], [0.05, -0.01])
    assert np.allclose(net.res_flow_control["mdot_from_kg_per_s"].values, [0.03])
    assert np.allclose(net.res_ext_grid["mdot_kg_per_s"].values, [-0.05, -0.01])


@pytest.mark.parametrize("use_numba", [True, False])
def test_flow_control_after_index_gap(use_numba):
    """
    FlowControlComponent.register_hydraulic_equations used to look up control_active /
    controlled_mdot_kg_per_s via net[table_name].values[tbl_idx], where tbl_idx came from
    IdxBranch.ELEMENT_IDX - the pandas *index label* of the flow control, not its position
    in net.flow_control. .values is positional, so as soon as the table's index isn't
    0..n-1 anymore (e.g. after dropping a flow control and adding a replacement, since
    pandas keeps counting new row labels upward instead of reusing the freed one), the
    lookup goes out of bounds or silently picks up a different flow control's set point.
    """
    net = pandapipes.create_empty_network("net", add_stdtypes=True, fluid="hgas")

    j = pandapipes.create_junctions(net, 8, pn_bar=1, tfluid_k=298)
    j1, j2, j3, j4, j5, j6, j7, j8 = j

    p12, p23, p34, p45, p56, p58, p67 = pandapipes.create_pipes_from_parameters(
        net, [j1, j2, j3, j4, j5, j5, j6], [j2, j3, j4, j5, j6, j8, j7], 0.2, 100, k_mm=0.1)

    fcs = pandapipes.create_flow_controls(net, [j2, j3, j4], [j6, j5, j8], [0.03, 0.02, 0.03], 0.1)

    pandapipes.create_ext_grid(net, j1, p_bar=1, t_k=298)
    pandapipes.create_sinks(net, [j3, j4, j5, j7, j8], [0.02, 0.04, 0.03, 0.04, 0.02])

    # same 3 flow controls/topology/set points as above, just re-labeled: drop the first
    # flow control and add an equivalent replacement -> index becomes [1, 2, 3] instead of
    # [0, 1, 2]
    net.flow_control.drop(index=[fcs[0]], inplace=True)
    pandapipes.create_flow_control(net, j2, j6, controlled_mdot_kg_per_s=0.03, diameter_m=0.1)
    assert net.flow_control.index.tolist() == [1, 2, 3]

    max_iter_hyd = 4 if use_numba else 4
    pandapipes.pipeflow(net, max_iter_hyd=max_iter_hyd, mode="hydraulics", use_numba=use_numba)

    assert np.allclose(net.res_pipe.loc[[p12, p23, p34, p45, p56, p58, p67], "mdot_from_kg_per_s"],
                       [0.15, 0.12, 0.08, 0.01, 0.01, -0.01, 0.04])
    assert np.allclose(net.res_flow_control["mdot_from_kg_per_s"].values, [0.02, 0.03, 0.03])
