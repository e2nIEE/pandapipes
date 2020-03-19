# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import json
import copy
import pandas as pd
from numpy import dtype
from pandapipes import __version__
from pandapower.auxiliary import ADict

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


class pandapipesNet(ADict):
    def __init__(self, *args, **kwargs):
        super(pandapipesNet, self).__init__(*args, **kwargs)

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
            r += "\n and the following results tables:"
            for tb in res:
                r += "\n   - %s (%s elements)" % (tb, len(self[tb]))
        return r

    def __str__(self):
        return self.__class__.__name__

    def __deepcopy__(self, memo):
        """
        overloads the deepcopy function of pandapipes if at least one DataFrame with column "object" is in net

        reason: some of these objects contain a reference to net which breaks the default deepcopy function.
        This fix was introduced in analogy to pandapower 2.2.1 and is rather a quick and dirty solution to the problem

        """
        save_to_json = False
        for el in self:
            if isinstance(self[el], pd.DataFrame) and "object" in self[el] and len(self[el]):
                save_to_json = True
                break
        if save_to_json:
            # deepcopy by dumping to json and loading from it
            from pandapipes.io.file_io import PPJSONEncoder, from_json_string
            from pandapipes.io.io_utils import with_signature
            json_string = json.dumps(with_signature(self, dict(self)), cls=PPJSONEncoder)
            return from_json_string(json_string)

        # deepcopy of every element in self (== the pandapower net)
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result


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
        "controller": [('controller', dtype(object)),
                       ('in_service', "bool"),
                       ('order', "float64"),
                       ('level', dtype(object)),
                       ("recycle", "bool")],
        "component_list": []}
    return default_pandapipes_structure
