# Copyright (c) 2020-2026 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
import numpy as np
from packaging import version

from pandapipes import __format_version__, __version__
from pandapipes.pandapipes_net import add_default_components, Sector
from pandapipes.component_models.circulation_pump_mass_component import CirculationPumpMass
from pandapipes.component_models.circulation_pump_pressure_component import CirculationPumpPressure
from pandapipes.component_models.valve_component import Valve
from pandapipes.component_models.pipe_component import Pipe
from pandapipes.component_models.heat_exchanger_component import HeatExchanger

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def convert_format(net):
    """
    Converts old nets to new format to ensure consistency. The converted net is returned.
    """
    _add_sector(net)
    add_default_components(net, overwrite=False)
    format_version = version.parse(__format_version__)
    # For possible problems with this line of code, please check out
    # https://github.com/e2nIEE/pandapipes/issues/320
    if not hasattr(net, 'format_version'):
        net.format_version = net.version
    if version.parse(net.format_version) >= format_version:
        return net
    _rename_columns(net)
    _add_missing_columns(net)
    _rename_attributes(net)
    net.version = __version__
    net.format_version = __format_version__
    return net


def _rename_columns(net):
    if "controller" in net:
        _rename_controller_columns(net)
    if "pipe" in net:
        _rename_pipe_columns(net)
    if Valve.table_name() in net:
        _rename_valve_columns(net)
    if "heat_exchanger" in net:
        _rename_heat_exchanger_columns(net)
    for comp in [CirculationPumpMass, CirculationPumpPressure]:
        cp_tbl = comp.table_name()
        if cp_tbl in net:
            old_cols = ["from_junction", "to_junction", "mdot_kg_per_s", "p_bar", "t_k"]
            new_cols = list(comp.from_to_node_cols()) + ["mdot_flow_kg_per_s", "p_flow_bar",
                                                         "t_flow_k"]
            for old_col, new_col in list(zip(old_cols, new_cols)):
                if old_col in net[cp_tbl].columns and new_col not in net[cp_tbl].columns:
                    net[cp_tbl].rename(columns={old_col: new_col}, inplace=True)


def _add_missing_columns(net):
    if "controller" in net and "initial_run" not in net.controller:
        net.controller.insert(4, 'initial_run', False)
        for _, ctrl in net.controller.iterrows():
            if hasattr(ctrl['object'], 'initial_run'):
                net.controller.at[ctrl.name, 'initial_run'] = ctrl['object'].initial_run
            else:
                net.controller.at[ctrl.name, 'initial_run'] = ctrl['object'].initial_pipeflow
    for comp in [Pipe, Valve, HeatExchanger]:
        if comp.table_name() in net:
            if "outer_diameter_mm" not in net[comp.table_name()].columns:
                net[comp.table_name()]["outer_diameter_mm"] = np.nan


def _rename_attributes(net):
    if "std_type" in net and "std_types" not in net:
        net["std_types"] = net["std_type"]
        del net["std_type"]


def _add_sector(net):
    if "sector" not in net:
        net["sector"] = Sector.ALL


def _rename_controller_columns(net):
    if ("controller" in net.controller) and ("object" in net.controller):
        if net['controller'].at[0, 'object'] is None:
            net['controller'].drop('object', inplace=True, axis=1)
        else:
            net['controller'].drop('controller', inplace=True, axis=1)
    net["controller"].rename(columns={"controller": "object"}, inplace=True)


def _rename_pipe_columns(net):
    if "u_w_per_m2k" not in net["pipe"].columns:
        net["pipe"].rename(columns={"alpha_w_per_m2k": "u_w_per_m2k"}, inplace=True)
    if "inner_diameter_mm" not in net["pipe"].columns:
        net["pipe"].diameter_m = net["pipe"].diameter_m * 1000.
        net["pipe"].rename(columns={"diameter_m": "inner_diameter_mm"}, inplace=True)


def _rename_valve_columns(net):
    if "inner_diameter_mm" not in net[Valve.table_name()].columns:
        net[Valve.table_name()].diameter_m = net[Valve.table_name()].diameter_m * 1000.
        net[Valve.table_name()].rename(columns={"diameter_m": "inner_diameter_mm"}, inplace=True)

    old_cols = ["from_junction", "to_junction"]
    new_cols = list(Valve.from_to_node_cols())
    old_net = False
    for o, n in zip(old_cols, new_cols):
        if o in net[Valve.table_name()]:
            net[Valve.table_name()].rename(columns={o: n}, inplace=True)
            old_net = True
    if old_net:
        if 'et' in net[Valve.table_name()]:
            logger.warning(r"'et' is a new required variable for valves. Therefore, 'et' will be overwritten.")
        net[Valve.table_name()]['et'] = 'ju'


def _rename_heat_exchanger_columns(net):
    if "inner_diameter_mm" not in net["heat_exchanger"].columns:
        if "diameter_m" in net["heat_exchanger"].columns:
            net["heat_exchanger"].diameter_m = net["heat_exchanger"].diameter_m * 1000.
            net["heat_exchanger"].rename(columns={"diameter_m": "inner_diameter_mm"}, inplace=True)
        else:
            logger.warning(r"Neither the required 'inner_diameter_mm' nor the deprecated "
                           r"'diameter_m'  are defined for heat exchangers. Please define "
                           r"'inner_diameter_mm' for heat exchangers to ensure correct results.")
            net["heat_exchanger"]["inner_diameter_mm"] = np.nan
