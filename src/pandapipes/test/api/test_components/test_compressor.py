# Copyright (c) 2020-2026 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
import pytest

import pandapipes
from pandapipes.component_models.component_toolbox import p_correction_height_air


@pytest.mark.parametrize("use_numba", [True, False])
def test_compressor_pressure_ratio(use_numba):

    net = pandapipes.create_empty_network("net", add_stdtypes=True, fluid="hgas")

    j = pandapipes.create_junctions(net, 6, pn_bar=5, tfluid_k=283.15)
    j1, j2, j3, j4, j5, j6 = j

    pandapipes.create_pipe_from_parameters(net, j1, j2, length_km=0.43380, inner_diameter_mm=102.2)
    pandapipes.create_pipe_from_parameters(net, j3, j4, length_km=0.26370, inner_diameter_mm=102.2)
    pandapipes.create_ext_grid(net, j1, 5, 283.15, type="p")
    pandapipes.create_sink(net, j6, 0.02333)

    br1 = 1.5
    br3 = 1.1
    c1 = pandapipes.create_compressor(net, j2, j3, pressure_ratio=br1)
    c2 = pandapipes.create_compressor(net, j5, j4, pressure_ratio=br1)  # reverse flow -> bypass
    c3 = pandapipes.create_compressor(net, j5, j6, pressure_ratio=br3)

    max_iter_hyd = 5 if use_numba else 5
    pandapipes.pipeflow(net, max_iter_hyd=max_iter_hyd, use_numba=use_numba)
    net.res_junction["abs_p_bar"] = net.res_junction.p_bar + \
                                    p_correction_height_air(net.junction.height_m)

    assert np.isclose(net.res_junction.at[j3, "abs_p_bar"],
                      br1 * net.res_junction.at[j2, "abs_p_bar"])
    assert np.isclose(net.res_junction.at[j6, "abs_p_bar"],
                      br3 * net.res_junction.at[j5, "abs_p_bar"])
    assert np.isclose(net.res_junction.at[j5, "p_bar"], net.res_junction.at[j4, "p_bar"]), \
        "pressure lift on rev. flow should be 0"
    assert np.isclose(net.res_junction.at[j5, "abs_p_bar"], net.res_junction.at[j4, "abs_p_bar"]), \
        "pressure lift on rev. flow should be 0"
    assert np.isclose(net.res_compressor.at[c1, "deltap_bar"],
                      net.res_junction.at[j2, "abs_p_bar"] * (br1 - 1))
    assert np.isclose(net.res_compressor.at[c3, "deltap_bar"],
                      net.res_junction.at[j5, "abs_p_bar"] * (br3 - 1))
    assert np.isclose(net.res_compressor.at[c2, "deltap_bar"], 0.0), \
        "pressure lift on rev. flow should be 0"

    br_new = 1.3
    net.compressor.loc[c1, "pressure_ratio"] = br_new

    pandapipes.pipeflow(net, max_iter_hyd=max_iter_hyd)
    net.res_junction["abs_p_bar"] = net.res_junction.p_bar + \
                                    p_correction_height_air(net.junction.height_m)

    assert np.isclose(net.res_junction.at[j3, "abs_p_bar"],
                      br_new * net.res_junction.at[j2, "abs_p_bar"])
    assert np.isclose(net.res_compressor.at[c1, "deltap_bar"],
                      net.res_junction.at[j2, "abs_p_bar"] * (br_new - 1))
    assert np.isclose(net.res_junction.at[j5, "abs_p_bar"], net.res_junction.at[j4, "abs_p_bar"])
    assert np.isclose(net.res_compressor.at[c2, "deltap_bar"], 0.0), \
        "pressure lift on rev. flow should be 0"


@pytest.mark.parametrize("use_numba", [True, False])
def test_compressor_pressure_ratio_after_index_gap(use_numba):
    """
    Compressor._compute_pl (inherited call site in Pump.register_hydraulic_equations)
    looks up get_component_array(net, "compressor")[tbl_idx, PRESSURE_RATIO], where
    tbl_idx comes from IdxBranch.ELEMENT_IDX. That array is built positionally (row i =
    i-th row of net.compressor), but ELEMENT_IDX stores the pandas *index label* of the
    compressor, not its position - so once net.compressor's index isn't 0..n-1 anymore,
    the lookup goes out of bounds (or, if still in bounds, silently applies a different
    compressor's pressure_ratio).

    A non-contiguous index arises from perfectly ordinary usage: dropping a compressor
    and adding a replacement, since pandas keeps counting new row labels upward instead
    of reusing the freed one.
    """
    net = pandapipes.create_empty_network("net", fluid="hgas")

    j1, j2, j3, j4, j5, j6 = pandapipes.create_junctions(net, 6, pn_bar=5, tfluid_k=283.15)
    pandapipes.create_pipe_from_parameters(net, j1, j2, length_km=0.1, inner_diameter_mm=102.2)
    pandapipes.create_pipe_from_parameters(net, j5, j6, length_km=0.1, inner_diameter_mm=102.2)
    pandapipes.create_ext_grid(net, j1, 5, 283.15, type="p")
    pandapipes.create_sink(net, j6, 0.02333)

    r0, r1, r2 = 1.2, 1.5, 1.1
    c0 = pandapipes.create_compressor(net, j2, j3, pressure_ratio=r0)
    c1 = pandapipes.create_compressor(net, j3, j4, pressure_ratio=r1)
    c2 = pandapipes.create_compressor(net, j4, j5, pressure_ratio=r2)

    # same 3 compressors/topology/ratios as above, just re-labeled: drop the middle
    # compressor and add an equivalent replacement -> index becomes [0, 2, 3] instead of
    # [0, 1, 2]
    net.compressor.drop(index=[c1], inplace=True)
    c1 = pandapipes.create_compressor(net, j3, j4, pressure_ratio=r1)
    assert net.compressor.index.tolist() == [c0, c2, c1]

    max_iter_hyd = 5 if use_numba else 5
    pandapipes.pipeflow(net, max_iter_hyd=max_iter_hyd, use_numba=use_numba)
    net.res_junction["abs_p_bar"] = net.res_junction.p_bar + \
                                    p_correction_height_air(net.junction.height_m)

    assert np.isclose(net.res_compressor.at[c0, "deltap_bar"],
                      net.res_junction.at[j2, "abs_p_bar"] * (r0 - 1))
    assert np.isclose(net.res_compressor.at[c1, "deltap_bar"],
                      net.res_junction.at[j3, "abs_p_bar"] * (r1 - 1))
    assert np.isclose(net.res_compressor.at[c2, "deltap_bar"],
                      net.res_junction.at[j4, "abs_p_bar"] * (r2 - 1))


if __name__ == '__main__':
    n = pytest.main(["test_compressor.py"])