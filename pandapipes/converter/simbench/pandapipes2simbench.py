# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import pandapipes as pp
import pandas as pd
import os
from glob import glob

def pandapipes_to_simbench(pandapipes_path, name="net"):
    net = pp.from_json(pandapipes_path)
    pandapipes_path = pandapipes_path[:-4]
    os.mkdir(pandapipes_path)
    for key in net.keys():
        if not isinstance(net[key], pd.DataFrame):
            continue
        if key == 'controller':
            continue
        net[key].rename(columns = {'name': 'id'}, inplace = True)
        net[key].to_csv(os.path.join(pandapipes_path, key + '.csv'), sep = ';', index_label = False, index = False)

if __name__ == '__main__':
    paths = glob(os.path.join(pp.pp_dir, 'networks', 'simple_test_networks', 'simbench_test_networks','*.json'))
    for path in paths:
        pandapipes_to_simbench(path)