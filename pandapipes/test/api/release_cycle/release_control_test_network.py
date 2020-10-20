# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import os

import pandapower.control as control
import pandas as pd
from pandapower.timeseries import DFData

import pandapipes as pp
from pandapipes import pp_dir

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)
path = os.path.join(pp_dir, "test", "api", "old_versions")


def release_control_test_network():
    # empty net
    net = pp.create_empty_network("net", add_stdtypes=False)

    # fluid
    pp.create_fluid_from_lib(net, "water", overwrite=True)

    # junctions
    pp.create_junction(net, pn_bar=3, tfluid_k=293, height_m=0, name="Junction 0", index=None, in_service=True,
                       type="junction", geodata=None)
    pp.create_junction(net, pn_bar=3, tfluid_k=293, height_m=0, name="Junction 1")
    pp.create_junction(net, pn_bar=3, tfluid_k=293, height_m=0, name="Junction 2")
    pp.create_junction(net, pn_bar=3, tfluid_k=293, height_m=0, name="Junction 3")
    pp.create_junction(net, pn_bar=3, tfluid_k=293, height_m=0, name="Junction 4")
    pp.create_junction(net, pn_bar=3, tfluid_k=293, height_m=0, name="Junction 5")
    pp.create_junction(net, pn_bar=3, tfluid_k=293, height_m=0, name="Junction 6")
    pp.create_junction(net, pn_bar=3, tfluid_k=293, height_m=0, name="Junction 7")
    pp.create_junction(net, pn_bar=3, tfluid_k=293, height_m=0, name="Junction 8")
    pp.create_junction(net, pn_bar=3, tfluid_k=293, height_m=0, name="Junction 9")
    pp.create_junction(net, pn_bar=3, tfluid_k=293, height_m=0, name="Junction 10")
    pp.create_junction(net, pn_bar=3, tfluid_k=293, height_m=0, name="Junction 11")
    pp.create_junction(net, pn_bar=3, tfluid_k=293, height_m=0, name="Junction 12")

    # pipes
    pp.create_pipe_from_parameters(net, from_junction=0, to_junction=8, length_km=3, diameter_m=0.01, k_mm=1,
                                   loss_coefficient=0, sections=10, alpha_w_per_m2k=10, text_k=293,
                                   qext_w=0., name="Pipe 0", index=None, geodata=None, in_service=True, type="pipe")
    pp.create_pipe_from_parameters(net, 9, 2, length_km=6, diameter_m=0.075, k_mm=.1, sections=10,
                                   alpha_w_per_m2k=3,
                                   name="Pipe 1")
    pp.create_pipe_from_parameters(net, 2, 12, length_km=5, diameter_m=0.06, k_mm=.1, sections=10,
                                   alpha_w_per_m2k=20,
                                   name="Pipe 2")
    pp.create_pipe_from_parameters(net, 4, 12, length_km=0.1, diameter_m=0.07, k_mm=.1, sections=10,
                                   alpha_w_per_m2k=2,
                                   name="Pipe 3")
    pp.create_pipe_from_parameters(net, 5, 3, length_km=1, diameter_m=0.09, k_mm=.1, sections=10, alpha_w_per_m2k=3,
                                   name="Pipe 4")
    pp.create_pipe_from_parameters(net, 4, 11, length_km=2.5, diameter_m=0.08, k_mm=.1, sections=10,
                                   alpha_w_per_m2k=15,
                                   name="Pipe 5")
    pp.create_pipe_from_parameters(net, 7, 6, length_km=4.5, diameter_m=0.085, k_mm=.1, sections=10,
                                   alpha_w_per_m2k=2.5, name="Pipe 6")
    pp.create_pipe_from_parameters(net, 1, 7, length_km=4, diameter_m=0.03, k_mm=.1, sections=10, alpha_w_per_m2k=1,
                                   name="Pipe 7")

    # external grids
    pp.create_ext_grid(net, junction=0, p_bar=3, t_k=300, name="External Grid 0", in_service=True, index=None,
                       type="pt")
    pp.create_ext_grid(net, 1, p_bar=5, t_k=350, name="External Grid 1", type="pt")

    # sinks
    pp.create_sink(net, junction=2, mdot_kg_per_s=0.2, scaling=1., name="Sink 0", index=None, in_service=True,
                   type="sink")
    pp.create_sink(net, 3, mdot_kg_per_s=0.1, name="Sink 1")
    pp.create_sink(net, 4, mdot_kg_per_s=0.5, name="Sink 2")
    pp.create_sink(net, 5, mdot_kg_per_s=0.07, name="Sink 3")
    pp.create_sink(net, 6, mdot_kg_per_s=0.09, name="Sink 4")
    pp.create_sink(net, 7, mdot_kg_per_s=0.1, name="Sink 5")

    # sources
    pp.create_source(net, junction=8, mdot_kg_per_s=0.1, scaling=1., name="Source 0", index=None, in_service=True,
                     type="source")
    pp.create_source(net, junction=9, mdot_kg_per_s=0.03, name="Source 1")
    pp.create_source(net, junction=10, mdot_kg_per_s=0.04, name="Source 2")
    pp.create_source(net, junction=11, mdot_kg_per_s=0.09, name="Source 3")

    # valves
    pp.create_valve(net, from_junction=8, to_junction=9, diameter_m=0.1, opened=True, loss_coefficient=0,
                    name="Valve 0", index=None, type="valve")
    pp.create_valve(net, 9, 4, diameter_m=0.05, opened=True, name="Valve 1")

    # pump
    pp.create_pump_from_parameters(net, from_junction=8, to_junction=3, new_std_type_name="Pump",
                                   pressure_list=[6.1, 5.8, 4],
                                   flowrate_list=[0, 19, 83], reg_polynomial_degree=2,
                                   poly_coefficents=None, name=None, index=None, in_service=True,
                                   type="pump")

    # circulation pump mass
    pp.create_circ_pump_const_mass_flow(net, from_junction=3, to_junction=4, p_bar=6, mdot_kg_per_s=1,
                                        t_k=290, name="Circ. Pump Mass", index=None, in_service=True,
                                        type="pt")

    # circulation pump pressure
    pp.create_circ_pump_const_pressure(net, from_junction=11, to_junction=5, p_bar=5, plift_bar=2,
                                       t_k=290, name="Circ. Pump Pressure", index=None, in_service=True, type="pt")

    # heat exchanger
    pp.create_heat_exchanger(net, from_junction=10, to_junction=6, diameter_m=0.08, qext_w=50, loss_coefficient=0,
                             name="Heat Exchanger 0", index=None, in_service=True, type="heat_exchanger")
    pp.create_heat_exchanger(net, from_junction=4, to_junction=10, diameter_m=0.08, qext_w=28000,
                             loss_coefficient=0,
                             name="Heat Exchanger 1", index=None, in_service=True, type="heat_exchanger")
    # time series
    profiles_sink = pd.read_csv(os.path.join(pp_dir, 'test', 'api', 'release_cycle',
                                             'release_control_test_sink_profiles.csv'), index_col=0)
    profiles_source = pd.read_csv(os.path.join(pp_dir, 'test', 'api', 'release_cycle',
                                               'release_control_test_source_profiles.csv'), index_col=0)
    ds_sink = DFData(profiles_sink)
    ds_source = DFData(profiles_source)

    const_sink = control.ConstControl(net, element='sink', variable='mdot_kg_per_s',
                                      element_index=net.sink.index.values, data_source=ds_sink,
                                      profile_name=net.sink.index.values.astype(str))
    const_source = control.ConstControl(net, element='source', variable='mdot_kg_per_s',
                                        element_index=net.source.index.values,
                                        data_source=ds_source,
                                        profile_name=net.source.index.values.astype(str))

    const_sink.initial_run = False
    const_source.initial_run = False

    pp.pipeflow(net)

    pp.to_json(net, os.path.join(path, 'example_%s.json' % pp.__version__))

    return net


if __name__ == '__main__':
    net = release_control_test_network()
