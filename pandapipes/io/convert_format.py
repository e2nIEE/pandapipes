# Copyright (c) 2020-2022 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

from packaging import version

from pandapipes import __version__
from pandapipes.pandapipes_net import add_default_components
from pandapipes.component_models.abstract_models.branch_models import BranchComponent
from pandapipes.component_models.abstract_models.node_models import NodeComponent
from pandapipes.component_models.abstract_models.node_element_models import NodeElementComponent

try:
    from pandaplan.core import ppglog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def convert_format(net):
    """
    Converts old nets to new format to ensure consistency. The converted net is returned.
    """
    add_default_components(net, overwrite=False)
    current_version = version.parse(__version__)
    # For possible problems with this line of code, please check out
    # https://github.com/e2nIEE/pandapipes/issues/320
    if isinstance(net.version, str) and version.parse(net.version) >= current_version:
        return net
    if version.parse(net.version) <= version.parse('0.5.0'):
        _fluid_dictonary(net)
    if 'component_list' in net:
        _change_component_list(net)
    _rename_columns(net)
    _add_missing_columns(net)
    _rename_attributes(net)
    net.version = __version__
    return net

def _change_component_list(net):
    for component in net['component_list']:
        if issubclass(component, BranchComponent):
            net['branch_list'] += [component]
        if issubclass(component, NodeElementComponent):
            net['node_element_list'] += [component]
        if issubclass(component, NodeComponent):
            net['node_list'] += [component]



def _fluid_dictonary(net):
    fluid = net.fluid
    net.fluid = {}
    net.fluid[fluid.name] = fluid
    net.ext_grid.insert(4, 'fluid', fluid.name)
    if 'source' in net:
        net.source.insert(3, 'fluid', fluid.name)
    if 'heat_exchanger' in net:
        net.heat_exchanger.insert(5, 'fluid', fluid.name)

def _rename_columns(net):
    if "controller" in net:
        if ("controller" in net.controller) and ("object" in net.controller):
            if net['controller'].at[0, 'object'] is None:
                net['controller'].drop('object', inplace=True, axis=1)
            else:
                net['controller'].drop('controller', inplace=True, axis=1)
        net["controller"].rename(columns={"controller": "object"}, inplace=True)


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
