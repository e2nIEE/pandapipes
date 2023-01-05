# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

from packaging import version

from pandapipes import __format_version__, __version__
from pandapipes.pandapipes_net import add_default_components
from pandapipes.component_models.circulation_pump_mass_component import CirculationPumpMass
from pandapipes.component_models.circulation_pump_pressure_component import CirculationPumpPressure

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def convert_format(net):
    """
    Converts old nets to new format to ensure consistency. The converted net is returned.
    """
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
        if ("controller" in net.controller) and ("object" in net.controller):
            if net['controller'].at[0, 'object'] is None:
                net['controller'].drop('object', inplace=True, axis=1)
            else:
                net['controller'].drop('controller', inplace=True, axis=1)
        net["controller"].rename(columns={"controller": "object"}, inplace=True)
    for comp in [CirculationPumpMass, CirculationPumpPressure]:
        cp_tbl = comp.table_name()
        if cp_tbl in net:
            old_cols = ["to_junction", "from_junction", "mdot_kg_per_s", "p_bar", "t_k"]
            new_cols = list(comp.from_to_node_cols()) + ["mdot_flow_kg_per_s", "p_flow_bar",
                                                         "t_flow_k"]
            for old_col, new_col in list(zip(old_cols, new_cols)):
                if old_col in net[cp_tbl].columns and new_col not in net[cp_tbl].columns:
                    net[cp_tbl].rename(columns={old_col: new_col}, inplace=True)


def _add_missing_columns(net):
    if "initial_run" not in net.controller:
        net.controller.insert(4, 'initial_run', False)
        for _, ctrl in net.controller.iterrows():
            if hasattr(ctrl['object'], 'initial_run'):
                net.controller.at[ctrl.name, 'initial_run'] = ctrl['object'].initial_run
            else:
                net.controller.at[ctrl.name, 'initial_run'] = ctrl['object'].initial_pipeflow


def _rename_attributes(net):
    if "std_type" in net and "std_types" not in net:
        net["std_types"] = net["std_type"]
        del net["std_type"]
