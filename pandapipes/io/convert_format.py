# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from packaging import version

from pandapipes import __version__

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def convert_format(net):
    """
    Converts old nets to new format to ensure consistency. The converted net is returned.
    """
    if isinstance(net.version, str) and version.parse(net.version) >= version.parse(__version__):
        return net
    _rename_columns(net)
    _update_initial_run(net)
    net.version = __version__
    return net


def _rename_columns(net):
    if "controller" in net:
        if ("controller" in net.controller) and ("object" in net.controller):
            if net['controller'].at[0, 'object'] is None:
                net['controller'].drop('object', inplace=True, axis=1)
            else:
                net['controller'].drop('controller', inplace=True, axis=1)
        net["controller"].rename(columns={"controller": "object"}, inplace=True)


def _update_initial_run(net):
    if "controller" in net:
        for ctrl in net.controller.object.values:
            if hasattr(ctrl, 'initial_pipeflow'):
                logger.warning(
                    "initial_pipeflow is deprecated, but it is still an attribute in your controllers. "
                    "It will be removed in the future. Please use initial_run instead!")
                ctrl.initial_run = ctrl.initial_pipeflow
