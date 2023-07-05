# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import copy

import numpy as np
import pytest
from matplotlib.collections import PatchCollection, LineCollection

import pandapipes
import pandapipes.plotting as plot
from pandapipes.converter.stanet.valve_pipe_component.create_valve_pipe import create_valve_pipe_from_parameters
from pandapipes.converter.stanet.valve_pipe_component.valve_pipe_plotting import create_valve_pipe_collection
from pandapipes.test.test_toolbox import base_net_is_with_pumps


def test_collection_lengths():
    net = pandapipes.create_empty_network(add_stdtypes=False)
    d = 40e-3

    j1 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=293.15, geodata=(0, 0))
    j2 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=293.15, geodata=(2, 0))
    j3 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=293.15, geodata=(4, 2))
    j4 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=293.15, geodata=(4, 0))
    j5 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=293.15, geodata=(4, -2))
    j6 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=293.15, geodata=(6, 0))
    j7 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=293.15, geodata=(8, 0))
    j8 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=293.15, geodata=(10, 0))
    j9 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=293.15, geodata=(12, 0))
    j10 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=293.15, geodata=(14, 0))
    j11 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=293.15, geodata=(14, -2))

    pandapipes.create_ext_grid(net, j1, p_bar=5, t_k=293.15)
    pandapipes.create_sink(net, j5, mdot_kg_per_s=0.5)
    pandapipes.create_sink(net, j6, mdot_kg_per_s=0.5)
    pandapipes.create_sink(net, j8, mdot_kg_per_s=0.5)

    pandapipes.create_pipe_from_parameters(net, j1, j2, 0.1, diameter_m=d, k_mm=0.1,
                                           geodata=[(0, 0), (2, 0)])
    pandapipes.create_valve(net, j2, j3, diameter_m=d, opened=True, loss_coefficient=5e3)
    pandapipes.create_valve(net, j2, j4, diameter_m=d, opened=False, loss_coefficient=0.01)
    pandapipes.create_valve(net, j2, j5, diameter_m=d, opened=False, loss_coefficient=0.01)
    pandapipes.create_pipe_from_parameters(net, j3, j6, 0.1, diameter_m=d, k_mm=0.1,
                                           geodata=[(4, 2), (6, 2), (6, 0)])
    pandapipes.create_pipe_from_parameters(net, j4, j6, 0.2, diameter_m=d, k_mm=0.1,
                                           geodata=[(4, 0), (6, 0)])
    pandapipes.create_pipe_from_parameters(net, j5, j6, 0.2, diameter_m=d, k_mm=0.1,
                                           geodata=[(4, -2), (6, -2), (6, 0)])
    pandapipes.create_heat_exchanger(net, j6, j7, d, qext_w=20000)
    pandapipes.create_pump_from_parameters(net, j7, j8, 'P1')
    pandapipes.create_pressure_control(net, j8, j9, j9, 10.)
    pandapipes.create_flow_control(net, j9, j10, 0.5, 0.1)
    pandapipes.create_flow_control(net, j9, j11, 0.5, 0.1, control_active=False)

    pipe_coll_direct = plot.create_pipe_collection(net, use_junction_geodata=True)
    pipe_coll_real = plot.create_pipe_collection(net)
    assert np.all(pipe_coll_direct.indices.astype(np.int64)
                  == net.pipe.index.values.astype(np.int64))
    assert len(pipe_coll_direct.get_paths()) == len(net.pipe)
    assert all([len(p) == 2 for p in pipe_coll_direct.get_paths()])
    assert all([len(p) == len(net.pipe_geodata.coords.iloc[i])
                for i, p in enumerate(pipe_coll_real.get_paths())])

    junction_coll = plot.create_junction_collection(net)
    assert np.all(junction_coll.node_indices.astype(np.int64)
                  == net.junction.index.values.astype(np.int64))
    assert len(junction_coll.get_paths()) == len(net.junction)

    valve_coll_patches, valve_coll_lines = plot.create_valve_collection(net)
    assert len(valve_coll_patches.get_paths()) == 2 * len(net.valve)
    assert len(valve_coll_lines.get_paths()) == 2 * len(net.valve)

    pump_coll_patches, pump_coll_lines = plot.create_pump_collection(net)
    assert len(pump_coll_patches.get_paths()) == len(net.pump)
    assert len(pump_coll_lines.get_paths()) == 4 * len(net.pump)

    hex_coll_patches, hex_coll_lines = plot.create_heat_exchanger_collection(net)
    assert len(hex_coll_patches.get_paths()) == 2 * len(net.heat_exchanger)
    assert len(hex_coll_lines.get_paths()) == 2 * len(net.heat_exchanger)

    pc_coll_patches, pc_coll_lines = plot.create_pressure_control_collection(net)
    assert len(pc_coll_patches.get_paths()) == len(net.press_control)
    assert len(pc_coll_lines.get_paths()) == 4 * len(net.press_control)

    fc_coll_patches, fc_coll_lines = plot.create_flow_control_collection(net)
    assert len(fc_coll_patches.get_paths()) == 3 * len(net.flow_control)
    assert len(fc_coll_lines.get_paths()) == 2 * len(net.flow_control)


def test_collections2(base_net_is_with_pumps):
    net = copy.deepcopy(base_net_is_with_pumps)

    pipe_coll_direct = plot.create_pipe_collection(net, use_junction_geodata=True)
    pipe_coll_real = plot.create_pipe_collection(net)
    assert np.all(pipe_coll_direct.indices.astype(np.int64)
                  == net.pipe.index.values.astype(np.int64))
    assert len(pipe_coll_direct.get_paths()) == len(net.pipe)
    assert all([len(p) == 2 for p in pipe_coll_direct.get_paths()])
    assert all([len(p) == len(net.pipe_geodata.coords.iloc[i])
                for i, p in enumerate(pipe_coll_real.get_paths())])
    pipe_coll2 = plot.create_pipe_collection(net, pipes=[2, 4])
    assert len(pipe_coll2.get_paths()) == 2
    assert np.all(pipe_coll2.indices.astype(np.int64) == np.array([2, 4]))

    junction_coll = plot.create_junction_collection(net)
    assert np.all(junction_coll.node_indices.astype(np.int64)
                  == net.junction.index.values.astype(np.int64))
    assert len(junction_coll.get_paths()) == len(net.junction)

    valve_coll_patches, valve_coll_lines = plot.create_valve_collection(net)
    assert len(valve_coll_patches.get_paths()) == 2 * len(net.valve)
    assert len(valve_coll_lines.get_paths()) == 2 * len(net.valve)

    hex_coll_patches, hex_coll_lines = plot.create_heat_exchanger_collection(net)
    assert len(hex_coll_patches.get_paths()) == 2 * len(net.heat_exchanger)
    assert len(hex_coll_lines.get_paths()) == 2 * len(net.heat_exchanger)

    pump_coll_patches, pump_coll_lines = plot.create_pump_collection(net)
    assert len(pump_coll_patches.get_paths()) == len(net.pump)
    assert len(pump_coll_lines.get_paths()) == 4 * len(net.pump)

    pc_coll_patches, pc_coll_lines = plot.create_pressure_control_collection(net)
    assert len(pc_coll_patches.get_paths()) == len(net.press_control)
    assert len(pc_coll_lines.get_paths()) == 4 * len(net.press_control)

    sink_colls = plot.create_sink_collection(net)
    assert len(sink_colls) == 2
    sink_patch_coll, sink_line_coll = sink_colls
    assert isinstance(sink_patch_coll, PatchCollection)
    assert isinstance(sink_line_coll, LineCollection)
    assert len(sink_patch_coll.get_paths()) == len(net.sink)
    assert len(sink_line_coll.get_paths()) == len(net.sink)

    source_colls = plot.create_source_collection(net)
    assert len(source_colls) == 2
    source_path_coll, source_line_coll = source_colls
    assert isinstance(source_path_coll, PatchCollection)
    assert isinstance(source_line_coll, LineCollection)
    assert len(source_path_coll.get_paths()) == len(net.source)
    assert len(source_line_coll.get_paths()) == 3 * len(net.source)

    eg_collections = plot.create_ext_grid_collection(net)
    assert len(eg_collections) == 2
    eg_path_coll, eg_line_coll = eg_collections
    assert isinstance(eg_path_coll, PatchCollection)
    assert isinstance(eg_line_coll, LineCollection)
    assert len(eg_path_coll.get_paths()) == len(net.source)
    assert len(eg_line_coll.get_paths()) == len(net.source)

    jc2 = plot.create_junction_collection(net, [])
    assert jc2 is None
    pc2 = plot.create_pipe_collection(net, [])
    assert pc2 is None
    skc2 = plot.create_sink_collection(net, [])
    assert skc2 is None
    scc2 = plot.create_source_collection(net, [])
    assert scc2 is None
    vc2 = plot.create_valve_collection(net, [])
    assert vc2 is None
    hx2 = plot.create_heat_exchanger_collection(net, [])
    assert hx2 is None
    pu2 = plot.create_pump_collection(net, [])
    assert pu2 is None
    pc2 = plot.create_pressure_control_collection(net, [])
    assert pc2 is None


def test_collection_valve_pipe():
    net = pandapipes.create_empty_network(add_stdtypes=False)
    d = 40e-3

    j1 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=293.15, geodata=(0, 0))
    j2 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=293.15, geodata=(2, 0))

    create_valve_pipe_from_parameters(net, j1, j2, 0.1, diameter_m=d, opened=True, loss_coefficient=5e3)

    vp_coll_patches, vp_coll_lines = create_valve_pipe_collection(net)

    assert len(vp_coll_patches.get_paths()) == 2 * len(net.valve_pipe)
    assert len(vp_coll_lines.get_paths()) == 2 * len(net.valve_pipe)

    vp2 = create_valve_pipe_collection(net, [])
    assert vp2 is None


if __name__ == '__main__':
    pytest.main(["test_collections.py"])
