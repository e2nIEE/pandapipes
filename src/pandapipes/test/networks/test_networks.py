# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import pandapipes as pp
import pytest
from pandapipes import networks


def test_schutterwald():
    net = networks.schutterwald(True, None)
    pp.pipeflow(net)
    assert net.converged

    net2 = networks.schutterwald(False, None)
    assert net2.sink.empty
    assert len(net2.pipe.loc[net2.pipe.type == "house_connection"]) == 0
    pp.pipeflow(net2)
    assert net2.converged

    net3 = networks.schutterwald(True, 30)
    assert len(net3.sink) == 1506
    assert net3.pipe.loc[net3.pipe.in_service & (net3.pipe.type == "house_connection"),
                         "length_km"].max() <= 0.03
    pp.pipeflow(net3)
    assert net3.converged


if __name__ == '__main__':
    n = pytest.main(["test_networks.py"])