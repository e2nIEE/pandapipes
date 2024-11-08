# Copyright (c) 2020-2024 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
"""
jede Komponente die nicht abstrakt und im PF genutzt werden soll, muss im componentregistry registriert werden.
Interne Komponenten werden automatisch hinzugefügt
"""

import os
from pandapipes.component_models import Junction, Pipe, ExtGrid, Sink, Source, Valve, Pump, PressureControlComponent, \
    MassStorage, CirculationPumpMass, CirculationPumpPressure, Compressor, FlowControlComponent, HeatConsumer, \
    HeatExchanger


class ComponentRegistry(object):

    registry = dict()

    def __init__(self):
        for comp in [Junction, Pipe, ExtGrid, Sink, Source, Valve, Pump, PressureControlComponent, MassStorage,
                     CirculationPumpMass, CirculationPumpPressure, Compressor, FlowControlComponent, HeatConsumer,
                     HeatExchanger]:
            comp.__iscustom__ = False
            comp_instance = comp()
            self.registry[comp_instance.table_name] = comp_instance
            self.registry[comp] = comp_instance

    @classmethod
    def register(cls):
        def wrapper(wrapped_class):
            wrapped_class.__iscustom__ = True
            wrapped_class.__path_from_home__ = os.path.relpath(__file__, os.path.expanduser("~"))
            wrapped_instance = wrapped_class()
            cls.registry[wrapped_instance.table_name] = wrapped_instance
            cls.registry[wrapped_class] = wrapped_instance
            return wrapped_class
        return wrapper

def register_component(cls):
    return ComponentRegistry.register()(cls)


COMPONENT_REGISTRY = ComponentRegistry().registry
