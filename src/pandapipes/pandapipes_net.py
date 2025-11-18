# Copyright (c) 2020-2025 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import copy

import numpy as np
import pandas as pd
from numpy import dtype

from pandapipes import __version__, __format_version__
from pandapipes.component_models.flow_control_component import FlowControlComponent
from pandapipes.component_models.heat_consumer_component import HeatConsumer
from pandapipes.component_models.pressure_control_component import PressureControlComponent
from pandapipes.component_models.junction_component import Junction
from pandapipes.component_models.pipe_component import Pipe
from pandapipes.component_models.ext_grid_component import ExtGrid
from pandapipes.component_models.component_toolbox import add_new_component
from pandapipes.component_models.valve_component import Valve
from pandapipes.component_models.pump_component import Pump
from pandapipes.component_models.sink_component import Sink
from pandapipes.component_models.source_component import Source
from pandapipes.component_models.mass_storage_component import MassStorage
from pandapipes.component_models.compressor_component import Compressor
from pandapipes.component_models.heat_exchanger_component import HeatExchanger
from pandapipes.component_models.circulation_pump_pressure_component import CirculationPumpPressure
from pandapipes.component_models.circulation_pump_mass_component import CirculationPumpMass
from pandapower.auxiliary import ADict
from pandas import Index

try:
    from enum import StrEnum
except ImportError:
    from enum import Enum
    class StrEnum(str, Enum):
        pass

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

class Sector(StrEnum):
    ALL = 'all'
    HEAT = 'heat'
    GAS = 'gas'
    WATER = 'water'
    NONE = 'None'

SectorMap = {
    Sector.ALL : [Sector.ALL, Sector.HEAT, Sector.GAS, Sector.WATER],
    Sector.HEAT : [Sector.ALL, Sector.HEAT],
    Sector.GAS : [Sector.ALL, Sector.GAS],
    Sector.WATER: [Sector.ALL, Sector.WATER],
    Sector.NONE : [Sector.ALL, Sector.HEAT, Sector.GAS, Sector.WATER]
}

logger = logging.getLogger(__name__)


class pandapipesNet(ADict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(args[0], self.__class__):
            net = args[0]
            self.clear()
            self.update(**net.deepcopy())

    def deepcopy(self):
        return copy.deepcopy(self)

    def __repr__(self):  # pragma: no cover
        r = "This pandapipes network includes the following parameter tables:"
        par = []
        res = []
        for tb in list(self.keys()):
            if isinstance(self[tb], pd.DataFrame) and len(self[tb]) > 0:
                if 'res_' in tb:
                    res.append(tb)
                else:
                    par.append(tb)
            elif tb == 'std_type':
                par.append(tb)
        for tb in par:
            r += "\n   - %s (%s elements)" % (tb, len(self[tb]))
        if res:
            r += "\nand the following results tables:"
            for tb in res:
                r += "\n   - %s (%s elements)" % (tb, len(self[tb]))
        r += "."
        if "fluid" in self and self["fluid"] is not None:
            r += "\nIt contains the following fluid: \n%s" % self["fluid"]
        else:
            r += "\nIt does not contain any defined fluid"
        if "component_list" in self:
            r += "\nand uses the following component models:"
            for component in self.component_list:
                r += "\n   - %s" % component.__name__
        return r


def get_basic_net_entries():
    return {
        "fluid": None,
        "converged": False,
        "name": "",
        "version": __version__,
        "format_version": __format_version__,
        "component_list": [],
        "sector": ""}

def get_basic_gas_components():
    return Junction, Pipe, Valve, Pump, Sink, Source, Compressor, FlowControlComponent, PressureControlComponent, MassStorage, ExtGrid

def get_basic_heat_components():
    return Junction, Pipe, Valve, Pump, HeatConsumer, HeatExchanger, CirculationPumpMass, CirculationPumpPressure

def get_basic_all_components():
    return (Junction, Pipe, Valve, Pump, Sink, Source, Compressor, FlowControlComponent, PressureControlComponent, MassStorage,
            HeatConsumer, HeatExchanger, CirculationPumpMass, CirculationPumpPressure, ExtGrid)


def add_default_components(net, overwrite=False):
    if net.sector == Sector.HEAT:
        components = get_basic_heat_components()
    elif net.sector == Sector.GAS:
        components = get_basic_gas_components()
    elif net.sector == Sector.ALL:
        components = get_basic_all_components()
    elif net.sector == Sector.NONE:
        components = []
    else:
        raise UserWarning(f'sector {net.sector} is not defined')

    for comp in components:
        add_new_component(net, comp, overwrite)
    if "controller" not in net or overwrite:
        ctrl_dtypes = [('object', dtype(object)),
                       ('in_service', "bool"),
                       ('order', "float64"),
                       ('level', dtype(object)),
                       ('initial_run', "bool"),
                       ("recycle", "bool")]
        net['controller'] = pd.DataFrame(np.zeros(0, dtype=ctrl_dtypes),
                                         index=Index([], dtype=np.int64))
