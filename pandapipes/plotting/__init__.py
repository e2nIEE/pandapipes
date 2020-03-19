# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

from pandapipes.plotting.collections import *
from pandapipes.plotting.generic_geodata import *
from pandapipes.plotting.simple_plot import *
from pandapipes.plotting.geo import *
from pandapower.plotting.collections import add_collections_to_axes, add_cmap_to_collection, \
    add_single_collection

import types
from matplotlib.backend_bases import GraphicsContextBase, RendererBase


class GC(GraphicsContextBase):
    """

    """

    def __init__(self):
        super().__init__()
        self._capstyle = 'round'


def custom_new_gc(self):
    """

    :param self:
    :type self:
    :return:
    :rtype:
    """
    return GC()


RendererBase.new_gc = types.MethodType(custom_new_gc, RendererBase)
