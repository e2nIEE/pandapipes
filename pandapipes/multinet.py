# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import copy

import pandas as pd
from numpy import dtype
from pandapipes import __version__
from pandapower.auxiliary import ADict
from pandapower import pandapowerNet
from pandapipes import pandapipesNet

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


class MultiNet(ADict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(args[0], self.__class__):
            net = args[0]
            self.clear()
            self.update(**net.deepcopy())

    def deepcopy(self):
        return copy.deepcopy(self)

    def __repr__(self):  # pragma: no cover
        r = "This multi net includes following nets:"
        for cat in self.nets:
            if isinstance(self['nets'][cat], pandapowerNet):
                r += "\n   - %s (%s pandapowerNet)" % (cat, 1)
            elif isinstance(self['nets'][cat], pandapipesNet):
                r += "\n   - %s (%s pandapipesNet)" % (cat, 1)
            else:
                r += "\n   - %s (%s nets)" % (cat, len(self['nets'][cat]))

        par = []
        for tb in list(self.keys()):
            if isinstance(self[tb], pd.DataFrame) and len(self[tb]) > 0:
                par.append(tb)
            elif tb == 'std_type':
                par.append(tb)
        if par:
            r += "\nand the following parameter tables:"
            for tb in par:
                r += "\n   - %s (%s elements)" % (tb, len(self[tb]))
        return r


def get_default_multinet_structure():
    """

    :return:
    :rtype:
    """
    default_multinet_structure = {
        # structure data
        # f8, u4 etc. are probably referencing numba or numpy data types
        "name": "",
        "nets": dict(),
        "version": __version__,
        "controller": [('object', dtype(object)),
                       ('in_service', "bool"),
                       ('order', "float64"),
                       ('level', dtype(object)),
                       ("recycle", "bool")]}
    return default_multinet_structure
