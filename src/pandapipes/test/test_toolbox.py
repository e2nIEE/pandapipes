# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import copy
import os

import numpy as np
import pandapower
import pandas as pd
import pytest
from packaging import version

import pandapipes
from pandapipes import networks as nw, BranchComponent
from pandapipes.component_models import NodeComponent
from pandapipes.idx_node import node_cols
from pandapipes.idx_branch import branch_cols
from pandapipes.test.api.test_convert_format import found_versions, folder, minimal_version_two_nets


def create_base_net(oos):
    net = pandapipes.create_empty_network(fluid="lgas")

    # create network elements, such as junctions, external grid, pipes, valves, sinks and sources
    junction1 = pandapipes.create_junction(net, pn_bar=1.05, tfluid_k=293.15, in_service=not oos,
                                           name="Connection to External Grid", geodata=(0, 0))
    junction2 = pandapipes.create_junction(net, pn_bar=1.05, tfluid_k=293.15, name="Junction 2",
                                           geodata=(2, 0), in_service=not oos)
    junction3 = pandapipes.create_junction(net, pn_bar=1.05, tfluid_k=293.15, name="Junction 3",
                                           geodata=(7, 4), in_service=not oos)
    junction4 = pandapipes.create_junction(net, pn_bar=1.05, tfluid_k=293.15, name="Junction 4",
                                           geodata=(7, -4), in_service=not oos)
    junction5 = pandapipes.create_junction(net, pn_bar=1.05, tfluid_k=293.15, name="Junction 5",
                                           geodata=(5, 3), in_service=not oos)
    junction6 = pandapipes.create_junction(net, pn_bar=1.05, tfluid_k=293.15, name="Junction 6",
                                           geodata=(5, -3))
    junction7 = pandapipes.create_junction(net, pn_bar=1.05, tfluid_k=293.15, name="Junction 7",
                                           geodata=(9, -4))
    junction8 = pandapipes.create_junction(net, pn_bar=1.05, tfluid_k=293.15, name="Junction 8",
                                           geodata=(9, 4))
    junction9 = pandapipes.create_junction(net, pn_bar=1.05, tfluid_k=293.15, name="Junction 9",
                                           geodata=(9, 0))
    junction10 = pandapipes.create_junction(net, pn_bar=1.05, tfluid_k=293.15, name="Junction 10",
                                            geodata=(12, 0))

    pandapipes.create_ext_grid(net, junction=junction1, p_bar=1.1, t_k=293.15,
                               name="Grid Connection")
    pandapipes.create_pipe_from_parameters(net, from_junction=junction1, to_junction=junction2,
                                           length_km=10, diameter_m=0.3, name="Pipe 1",
                                           geodata=[(0, 0), (2, 0)], in_service=not oos)
    pandapipes.create_pipe_from_parameters(net, from_junction=junction2, to_junction=junction3,
                                           length_km=2, diameter_m=0.3, name="Pipe 2",
                                           geodata=[(2, 0), (2, 4), (7, 4)], in_service=not oos)
    pandapipes.create_pipe_from_parameters(net, from_junction=junction2, to_junction=junction4,
                                           length_km=2.5, diameter_m=0.3, name="Pipe 3",
                                           geodata=[(2, 0), (2, -4), (7, -4)], in_service=not oos)
    pandapipes.create_pipe_from_parameters(net, from_junction=junction3, to_junction=junction5,
                                           length_km=1, diameter_m=0.3, name="Pipe 4",
                                           geodata=[(7, 4), (7, 3), (5, 3)])
    pandapipes.create_pipe_from_parameters(net, from_junction=junction4, to_junction=junction6,
                                           length_km=1, diameter_m=0.3, name="Pipe 5",
                                           geodata=[(7, -4), (7, -3), (5, -3)])
    pandapipes.create_pipe_from_parameters(net, from_junction=junction7, to_junction=junction8,
                                           length_km=1, diameter_m=0.3, name="Pipe 6",
                                           geodata=[(9, -4), (9, 0)])
    pandapipes.create_pipe_from_parameters(net, from_junction=junction7, to_junction=junction8,
                                           length_km=1, diameter_m=0.3, name="Pipe 7",
                                           geodata=[(9, 0), (9, 4)])

    pandapipes.create_valve(net, from_junction=junction5, to_junction=junction6, diameter_m=0.05,
                            opened=True)
    pandapipes.create_heat_exchanger(net, junction3, junction8, diameter_m=0.3, qext_w=20000)
    pandapipes.create_sink(net, junction=junction4, mdot_kg_per_s=0.545, name="Sink 1")
    pandapipes.create_source(net, junction=junction3, mdot_kg_per_s=0.234)
    pandapipes.create_pump_from_parameters(net, junction4, junction7, 'P1')
    pandapipes.create_pressure_control(net, junction9, junction10, junction10, 5.)

    if oos:
        pandapipes.create_ext_grid(net, junction=junction1, p_bar=1.1, t_k=293.15,
                                   name="Grid Connection", in_service=False)
        pandapipes.create_heat_exchanger(net, junction3, junction8, diameter_m=0.3, qext_w=20000,
                                         in_service=False)
        pandapipes.create_sink(net, junction=junction4, mdot_kg_per_s=0.545, name="Sink 2",
                               in_service=False)
        pandapipes.create_pump_from_parameters(net, junction4, junction7, 'P1', in_service=False)
        pandapipes.create_source(net, junction=junction3, mdot_kg_per_s=0.234, in_service=False)
        pandapipes.create_circ_pump_const_mass_flow(net, junction9, junction5, 1.05, 1,
                                                    in_service=False)
        pandapipes.create_circ_pump_const_pressure(net, junction9, junction6, 1.05, 0.5,
                                                   in_service=False)

    return net


def add_pumps(net):
    junction5 = net.junction.index[net.junction.name == "Junction 5"][0]
    junction6 = net.junction.index[net.junction.name == "Junction 6"][0]
    junction9 = net.junction.index[net.junction.name == "Junction 9"][0]

    pandapipes.create_circ_pump_const_mass_flow(net, junction9, junction5, 1.05, 1)
    pandapipes.create_circ_pump_const_pressure(net, junction9, junction6, 1.05, 0.5)
    return net


@pytest.fixture
def base_net_is_wo_pumps():
    return create_base_net(False)


@pytest.fixture
def base_net_is_with_pumps():
    net = create_base_net(False)
    add_pumps(net)
    return net


@pytest.fixture
def base_net_oos_wo_pumps():
    return create_base_net(True)


@pytest.fixture
def base_net_oos_with_pumps():
    net = create_base_net(True)
    add_pumps(net)
    return net


@pytest.fixture
def create_net_changed_indices(base_net_is_wo_pumps):
    net = copy.deepcopy(base_net_is_wo_pumps)

    new_junction_indices = [55, 38, 84, 65, 83, 82, 28, 49, 99, 105]
    new_pipe_indices = [30, 88, 72, 99,  0, 98, 70]
    new_valve_indices = [19]
    new_pump_indices = [93]
    new_hxc_indices = [67]
    new_sink_indices = [32]
    new_source_indices = [9]
    new_eg_indices = [20]
    new_pc_indices = [15]
    junction_lookup = dict(zip(net["junction"].index.values, new_junction_indices))

    net.junction.index = new_junction_indices
    net.junction_geodata.index = new_junction_indices
    for n_el in ["sink", "source", "ext_grid"]:
        net[n_el].junction = np.array([junction_lookup[k] for k in net[n_el].junction], dtype="int")
    for br_el in ["pipe", "valve", "heat_exchanger", "pump", "press_control"]:
        for junc_typ in ["from_junction", "to_junction"]:
            net[br_el][junc_typ] = np.array([junction_lookup[k] for k in net[br_el][junc_typ]],
                                            dtype="int")
    net.press_control.controlled_junction = np.array(
        [junction_lookup[k] for k in net.press_control.controlled_junction], dtype="int")

    net.pipe.index = new_pipe_indices
    net.pipe_geodata.index = new_pipe_indices
    net.valve.index = new_valve_indices
    net.pump.index = new_pump_indices
    net.heat_exchanger.index = new_hxc_indices
    net.sink.index = new_sink_indices
    net.source.index = new_source_indices
    net.ext_grid.index = new_eg_indices
    net.press_control.index = new_pc_indices

    return net


def get_junction_indices(net, branch_comp=("pipe", "valve", "heat_exchanger", "pump"),
                         node_comp=("ext_grid", "source", "sink")):
    junction_index = copy.deepcopy(net.junction.index.values)
    previous_junctions = {k: dict() for k in branch_comp + node_comp}
    for bc in branch_comp:
        previous_junctions[bc]["from_junction"] = copy.deepcopy(net[bc]["from_junction"])
        previous_junctions[bc]["to_junction"] = copy.deepcopy(net[bc]["to_junction"])
    for nc in node_comp:
        previous_junctions[nc]["junction"] = copy.deepcopy(net[nc]["junction"])
    return junction_index, previous_junctions


def test_reindex_junctions():
    net_orig = nw.simple_gas_networks.gas_tcross1()
    net = nw.simple_gas_networks.gas_tcross1()

    to_add = 5
    new_junction_idxs = np.array(list(net.junction.index)) + to_add
    junction_lookup = dict(zip(net["junction"].index.values, new_junction_idxs))
    # a more complexe junction_lookup of course should also work, but this one is easy to check
    _ = pandapipes.reindex_junctions(net, junction_lookup)

    for elm in net.keys():
        if isinstance(net[elm], pd.DataFrame) and net[elm].shape[0]:
            cols = pd.Series(net[elm].columns)
            junction_cols = cols.loc[cols.str.contains("junction")]
            if len(junction_cols):
                for junction_col in junction_cols:
                    assert all(net[elm][junction_col] == net_orig[elm][junction_col] + to_add)
            if elm == "junction":
                assert all(np.array(list(net[elm].index)) == np.array(list(
                    net_orig[elm].index)) + to_add)


def test_fuse_junctions(create_net_changed_indices):
    net = copy.deepcopy(create_net_changed_indices)
    junction_index, previous_junctions = get_junction_indices(net)

    j1, j2 = 84, 83
    pandapipes.fuse_junctions(net, j1, j2)
    new_junction_index = np.array([j for j in junction_index if j != j2])
    assert np.all(net.junction.index.values == new_junction_index)
    assert np.all(net.junction_geodata.index.values == new_junction_index)
    for comp, junc_dict in previous_junctions.items():
        for col, junctions in junc_dict.items():
            assert np.all(net[comp][col] == junctions.replace({j2: j1}))


def test_create_continuous_index(create_net_changed_indices):
    net = copy.deepcopy(create_net_changed_indices)
    junction_index, previous_junctions = get_junction_indices(net)

    junction_lookup = pandapipes.create_continuous_junction_index(net)
    new_junction_index = np.array([junction_lookup[j] for j in sorted(junction_index)])
    assert np.all(net.junction.index == new_junction_index)
    for comp, junc_dict in previous_junctions.items():
        for col, junctions in junc_dict.items():
            assert np.all(
                net[comp][col] == pd.Series(np.array([junction_lookup[k] for k in junctions.values],
                                                     dtype="int"), index=junctions.index))

    pandapipes.create_continuous_elements_index(net)
    for comp in previous_junctions.keys():
        assert np.all(net[comp].index == np.arange(len(net[comp])))


def test_select_subnet(base_net_is_wo_pumps):
    net = copy.deepcopy(base_net_is_wo_pumps)

    # Do nothing
    same_net = pandapipes.select_subnet(net, net.junction.index)
    assert len(same_net.component_list) == len(net.component_list)
    assert set(same_net.component_list) == set(net.component_list)
    for comp in net.component_list:
        assert pandapower.dataframes_equal(net[comp.table_name()], same_net[comp.table_name()])

    same_net2 = pandapipes.select_subnet(net, net.junction.index, include_results=True,
                                         keep_everything_else=True)
    assert pandapipes.nets_equal(net, same_net2)

    # Remove everything
    empty = pandapipes.select_subnet(net, set())
    for comp in net.component_list:
        assert len(empty[comp.table_name()]) == 0

    empty2 = pandapipes.select_subnet(net, set(), remove_unused_components=True)
    for comp in net.component_list:
        assert comp.table_name() not in empty2
        assert comp not in empty2.component_list

    # check length of results
    net = nw.gas_tcross2()
    pandapipes.pipeflow(net)
    net2 = pandapipes.select_subnet(net, net.junction.index[:-3], include_results=True)
    for comp in net.component_list:
        assert len(net2["res_" + comp.table_name()]) == len(net2[comp.table_name()])
    assert len(net.junction) == len(net2.junction) + 3


def test_pit_extraction():
    max_ver = max(found_versions)
    if version.parse(max_ver) >= version.parse(minimal_version_two_nets):
        names = ["_gas", "_water"]
    else:
        names = [""]

    for name in names:
        filename = os.path.join(folder, "example_%s%s.json" % (max_ver, name))
        net = pandapipes.from_json(filename)
        pandapipes.pipeflow(net)

        node_table, branch_table = pandapipes.get_internal_tables_pandas(net)

        assert node_table.shape[1] == node_cols
        assert branch_table.shape[1] == branch_cols

        for comp in net.component_list:
            tbl = comp.table_name()
            if len(net[tbl]) > 0:
                if issubclass(comp, NodeComponent):
                    assert tbl in node_table["TABLE_IDX"].values
                    assert np.all(np.isin(
                        net[tbl].index.values,
                        node_table.loc[node_table.TABLE_IDX == tbl].ELEMENT_IDX.values
                    ))
                elif issubclass(comp, BranchComponent):
                    assert tbl in branch_table["TABLE_IDX"].values
                    assert np.all(np.isin(
                        net[tbl].index.values,
                        branch_table.loc[branch_table.TABLE_IDX == tbl].ELEMENT_IDX.values
                    ))


def runpp_with_mark(net, **kwargs):
    pandapower.runpp(net, **kwargs)
    net['mark'] = "runpp"


def pipeflow_with_mark(net, **kwargs):
    pandapipes.pipeflow(net, **kwargs)
    net['mark'] = "pipeflow"


if __name__ == '__main__':
    n = pytest.main(["test_toolbox.py"])
