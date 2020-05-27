# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import json
import copy
import pandas as pd
from numpy import dtype
from pandapipes import __version__
from pandapower.auxiliary import ADict, _preserve_dtypes

try:
    import pplog as logging
except ImportError:
    import logging

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
                r += "\n   - %s" %component.__name__
        return r


def get_default_pandapipes_structure():
    """

    :return:
    :rtype:
    """
    default_pandapipes_structure = {
        # structure data
        # f8, u4 etc. are probably referencing numba or numpy data types
        "fluid": None,
        "converged": False,
        "name": "",
        "version": __version__,
        "controller": [('object', dtype(object)),
                       ('in_service', "bool"),
                       ('order', "float64"),
                       ('level', dtype(object)),
                       ("recycle", "bool")],
        "component_list": []}
    return default_pandapipes_structure
