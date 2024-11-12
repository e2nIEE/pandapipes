# Copyright (c) 2020-2024 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

from pandapipes.component_models.component_registry import ComponentRegistry
from pandapipes.component_models import Junction, Pipe, ExtGrid, Sink, Source, Valve, Pump, PressureControlComponent, \
    MassStorage, CirculationPumpMass, CirculationPumpPressure, Compressor, FlowControlComponent, HeatConsumer, \
    HeatExchanger, ValvePipe

"""
every high level Component from pandapipes should be initialized here
"""
COMPONENT_REGISTRY = ComponentRegistry(
    [Junction, Pipe, ExtGrid, Sink, Source, Valve, Pump, PressureControlComponent, MassStorage,
     CirculationPumpMass, CirculationPumpPressure, Compressor, FlowControlComponent, HeatConsumer,
     HeatExchanger, ValvePipe]).registry
