# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.


import pytest

import pandapipes


# "std_type";"nominal_width_mm";"outer_diameter_mm";"inner_diameter_mm";"standard_dimension_ratio";"material"
# 80_GGG;80;98.0;86.0;16.33;GGG


def test_create_and_load_std_type_pipe():
    net = pandapipes.create_empty_network()
    w = 82
    do = 99.5
    di = 87.5
    rat = 16.33
    mat = "GGG"
    name = "test_pipe"

    typdata = {}
    with pytest.raises(UserWarning):
        pandapipes.create_std_type(net, component="pipe", std_type_name=name, typedata=typdata,
                                   overwrite=True)

    typdata = {"standard_dimension_ratio": rat}
    with pytest.raises(UserWarning):
        pandapipes.create_std_type(net, component="pipe", std_type_name=name, typedata=typdata,
                                   overwrite=True)

    typdata = {"standard_dimension_ratio": rat, "material": mat}
    with pytest.raises(UserWarning):
        pandapipes.create_std_type(net, component="pipe", std_type_name=name, typedata=typdata,
                                   overwrite=True)

    typdata = {"standard_dimension_ratio": rat, "material": mat, "nominal_width_mm": w}
    with pytest.raises(UserWarning):
        pandapipes.create_std_type(net, component="pipe", std_type_name=name, typedata=typdata,
                                   overwrite=True)

    typdata = {"standard_dimension_ratio": rat, "material": mat, "inner_diameter_mm": di,
               "nominal_width_mm": w}
    pandapipes.create_std_type(net, component="pipe", std_type_name=name, typedata=typdata,
                               overwrite=True)

    typdata = {"standard_dimension_ratio": rat, "material": mat, "inner_diameter_mm": di,
               "outer_diameter_mm": do, "nominal_width_mm": w}
    pandapipes.create_std_type(net, component="pipe", std_type_name=name, typedata=typdata,
                               overwrite=True)
    assert net.std_types["pipe"][name] == typdata

    loaded_type = pandapipes.load_std_type(net, name, component="pipe")
    assert loaded_type == typdata

    with pytest.raises(ValueError):
        pandapipes.create_std_type(net, component="test_comp", std_type_name=name, typedata=typdata)
    with pytest.raises(UserWarning):
        pandapipes.load_std_type(net, "test_name", component="pipe")
    with pytest.raises(UserWarning):
        pandapipes.load_std_type(net, name, component="test_comp")


def test_create_std_types_pipe():
    net = pandapipes.create_empty_network()
    w = 82
    do = 99.5
    di = 87.5
    rat = 16.33
    mat = "GGG"

    typdata = {"standard_dimension_ratio": rat, "material": mat, "inner_diameter_mm": di,
               "outer_diameter_mm": do, "nominal_width_mm": w}

    typdatas = {"typ1": typdata, "typ2": typdata}
    pandapipes.create_std_types(net, component="pipe", type_dict=typdatas)
    assert net.std_types["pipe"]["typ1"] == typdata
    assert net.std_types["pipe"]["typ1"] == typdata


def test_copy_std_types_from_net_pipe():
    net1 = pandapipes.create_empty_network()
    net2 = pandapipes.create_empty_network()

    w = 82
    do = 99.5
    di = 87.5
    rat = 16.33
    mat = "GGG"

    typdata = {"standard_dimension_ratio": rat, "material": mat, "inner_diameter_mm": di,
               "outer_diameter_mm": do, "nominal_width_mm": w}

    pandapipes.create_std_type(net1, "pipe", "test_copy", typdata)
    pandapipes.copy_std_types(net2, net1, component="pipe")
    assert pandapipes.std_type_exists(net2, "test_copy", "pipe")
    assert net1.std_types["pipe"]["test_copy"] == net2.std_types["pipe"]["test_copy"]


# def test_create_and_load_std_type_pump():
#     net = pandapipes.create_empty_network()
#     sn_mva = 40
#     vn_hv_kv = 110
#     vn_lv_kv =  20
#     vk_percent = 5.
#     vkr_percent = 2.
#     pfe_kw=50
#     i0_percent = 0.1
#     shift_degree = 30
#     name = "test_trafo"
#
#     typdata = {"sn_mva": sn_mva}
#     with pytest.raises(UserWarning):
#         pandapipes.create_std_type(net, name=name, data=typdata, element="trafo")
#
#     typdata = {"sn_mva": sn_mva, "vn_hv_kv": vn_hv_kv}
#     with pytest.raises(UserWarning):
#         pandapipes.create_std_type(net, name=name, data=typdata, element="trafo")
#
#     typdata = {"sn_mva": sn_mva, "vn_hv_kv": vn_hv_kv, "vn_lv_kv": vn_lv_kv}
#     with pytest.raises(UserWarning):
#         pandapipes.create_std_type(net, name=name, data=typdata, element="trafo")
#
#     typdata = {"sn_mva": sn_mva, "vn_hv_kv": vn_hv_kv, "vn_lv_kv": vn_lv_kv, "vk_percent": vk_percent}
#     with pytest.raises(UserWarning):
#         pandapipes.create_std_type(net, name=name, data=typdata, element="trafo")
#
#     typdata = {"sn_mva": sn_mva, "vn_hv_kv": vn_hv_kv, "vn_lv_kv": vn_lv_kv, "vk_percent": vk_percent,
#                "vkr_percent": vkr_percent}
#     with pytest.raises(UserWarning):
#         pandapipes.create_std_type(net, name=name, data=typdata, element="trafo")
#
#     typdata = {"sn_mva": sn_mva, "vn_hv_kv": vn_hv_kv, "vn_lv_kv": vn_lv_kv, "vk_percent": vk_percent,
#                "vkr_percent": vkr_percent, "pfe_kw": pfe_kw}
#     with pytest.raises(UserWarning):
#         pandapipes.create_std_type(net, name=name, data=typdata, element="trafo")
#
#     typdata = {"sn_mva": sn_mva, "vn_hv_kv": vn_hv_kv, "vn_lv_kv": vn_lv_kv, "vk_percent": vk_percent,
#                "vkr_percent": vkr_percent, "pfe_kw": pfe_kw, "i0_percent": i0_percent}
#     with pytest.raises(UserWarning):
#         pandapipes.create_std_type(net, name=name, data=typdata, element="trafo")
#     typdata = {"sn_mva": sn_mva, "vn_hv_kv": vn_hv_kv, "vn_lv_kv": vn_lv_kv, "vk_percent": vk_percent,
#                "vkr_percent": vkr_percent, "pfe_kw": pfe_kw, "i0_percent": i0_percent,
#                "shift_degree": shift_degree}
#     pandapipes.create_std_type(net, name=name, data=typdata, element="trafo")
#     assert net.std_types["trafo"][name] == typdata
#
#     loaded_type = pandapipes.load_std_type(net, name, element="trafo")
#     assert loaded_type == typdata


# def test_create_std_types_pump():
#     net = pandapipes.create_empty_network()
#     sn_mva = 40
#     vn_hv_kv = 110
#     vn_lv_kv =  20
#     vk_percent = 5.
#     vkr_percent = 2.
#     pfe_kw=50
#     i0_percent = 0.1
#     shift_degree = 30
#
#     typdata = {"sn_mva": sn_mva, "vn_hv_kv": vn_hv_kv, "vn_lv_kv": vn_lv_kv, "vk_percent": vk_percent,
#                "vkr_percent": vkr_percent, "pfe_kw": pfe_kw, "i0_percent": i0_percent,
#                "shift_degree": shift_degree}
#     typdatas = {"typ1": typdata, "typ2": typdata}
#     pandapipes.create_std_types(net, data=typdatas, element="trafo")
#     assert net.std_types["trafo"]["typ1"] == typdata
#     assert net.std_types["trafo"]["typ2"] == typdata


# def test_find_pipe_type():
#     net = pandapipes.create_empty_network()
#     w = 82
#     do = 99.5
#     di = 87.5
#     rat = 16.33
#     mat = "GGG"
#     name = "test_pipe1"
#
#     typdata = {"standard_dimension_ratio": rat, "material": mat, "inner_diameter_mm": di,
#                "outer_diameter_mm": do, "nominal_width_mm": w}
#     pandapipes.create_std_type(net, typedata=typdata, std_type_name=name, component="pipe")
#     fitting_type = pandapipes.find_std_type_by_parameter(net, typdata)
#     assert len(fitting_type) == 1
#     assert fitting_type[0] == name
#
#     fitting_type = pandapipes.find_std_type_by_parameter(net, {"inner_diameter_mm": di+0.05},
#                                                          epsilon=.06)
#     assert len(fitting_type) == 1
#     assert fitting_type[0] == name
#
#     fitting_type = pandapipes.find_std_type_by_parameter(net, {"inner_diameter_mm": di+0.07},
#                                                          epsilon=.06)
#     assert len(fitting_type) == 0


@pytest.mark.xfail(reason="The standard type library has not yet been well integrated into "
                          "the pandapipes table structure, as e.g. the diameter in the std_types "
                          "is given as inner_diameter_mm, while it is given in the table as "
                          "diameter_m.")
def test_change_type_pipe():
    net = pandapipes.create_empty_network()
    w1 = 82
    do1 = 99.5
    di1 = 87.5
    rat1 = 16.33
    mat = "GGG"
    name1 = "test_pipe1"

    typdata1 = {"standard_dimension_ratio": rat1, "material": mat, "inner_diameter_mm": di1,
                "outer_diameter_mm": do1, "nominal_width_mm": w1}
    pandapipes.create_std_type(net, typedata=typdata1, std_type_name=name1, component="pipe")

    w2 = 125
    do2 = 144.0
    di2 = 131.6
    rat2 = 23.23
    mat = "GGG"
    name2 = "test_pipe2"

    typdata2 = {"standard_dimension_ratio": rat2, "material": mat, "inner_diameter_mm": di2,
                "outer_diameter_mm": do2, "nominal_width_mm": w2}
    pandapipes.create_std_type(net, typedata=typdata2, std_type_name=name2, component="pipe")

    j1 = pandapipes.create_junction(net, pn_bar=1, tfluid_k=293)
    j2 = pandapipes.create_junction(net, pn_bar=1, tfluid_k=293)
    pid = pandapipes.create_pipe(net, j1, j2, name1, 1.)

    assert net.pipe.diameter_m.at[pid] == di1 / 1000
    assert net.pipe.std_type.at[pid] == name1

    pandapipes.change_std_type(net, pid, name2, component="pipe")

    assert net.pipe.diameter_m.at[pid] == di2 / 1000
    assert net.pipe.std_type.at[pid] == name2


# def test_parameter_from_std_type_pipe():
#     net = pandapipes.create_empty_network()
#     w1 = 82
#     do1 = 99.5
#     di1 = 87.5
#     rat1 = 16.33
#     mat = "GGG"
#     name1 = "test_pipe1"
#
#     typdata1 = {"standard_dimension_ratio": rat1, "material": mat, "inner_diameter_mm": di1,
#                 "outer_diameter_mm": do1, "nominal_width_mm": w1}
#     pandapipes.create_std_type(net, data=typ1, name=name1, element="pipe")
#
#     r2 = 0.02
#     x2 = 0.04
#     c2 = 20
#     i2 = 0.4
#     endtemp2 = 40
#
#     endtemp_fill = 20
#
#     name2 = "test_line2"
#     typ2 = {"c_nf_per_km": c2, "r_ohm_per_km": r2, "x_ohm_per_km": x2, "max_i_ka": i2,
#             "endtemp_degree": endtemp2}
#     pandapipes.create_std_type(net, data=typ2, name=name2, element="pipe")
#
#     b1 = pandapipes.create_bus(net, vn_kv=0.4)
#     b2 = pandapipes.create_bus(net, vn_kv=0.4)
#     lid1 = pandapipes.create_line(net, b1, b2, 1., std_type=name1)
#     lid2 = pandapipes.create_line(net, b1, b2, 1., std_type=name2)
#     lid3 = pandapipes.create_line_from_parameters(net, b1, b2, 1., r_ohm_per_km=0.03, x_ohm_per_km=0.04,
#                                           c_nf_per_km=20, max_i_ka=0.3)
#
#     pandapipes.parameter_from_std_type(net, "endtemp_degree", fill=endtemp_fill)
#     assert net.line.endtemp_degree.at[lid1] == endtemp_fill #type1 one has not specified an endtemp
#     assert net.line.endtemp_degree.at[lid2] == endtemp2 #type2 has specified endtemp
#     assert net.line.endtemp_degree.at[lid3] == endtemp_fill #line3 has no standard type
#
#     net.line.endtemp_degree.at[lid3] = 10
#     pandapipes.parameter_from_std_type(net, "endtemp_degree", fill=endtemp_fill)
#     assert net.line.endtemp_degree.at[lid3] == 10 #check that existing values arent overwritten


def test_delete_std_type():
    net = pandapipes.create_empty_network()
    w = 82
    do = 99.5
    di = 87.5
    rat = 16.33
    mat = "GGG"

    typdata = {"standard_dimension_ratio": rat, "material": mat, "inner_diameter_mm": di,
               "outer_diameter_mm": do, "nominal_width_mm": w}

    typdatas = {"typ1": typdata, "typ2": typdata}
    pandapipes.create_std_types(net, component="pipe", type_dict=typdatas)
    pandapipes.delete_std_type(net, "typ1", "pipe")
    with pytest.raises(UserWarning):
        pandapipes.delete_std_type(net, "typ1", "pipe")


def test_available_std_types():
    net = pandapipes.create_empty_network()
    w = 82
    do = 99.5
    di = 87.5
    rat = 16.33
    mat = "GGG"

    typdata = {"standard_dimension_ratio": rat, "material": mat, "inner_diameter_mm": di,
               "outer_diameter_mm": do, "nominal_width_mm": w}

    typdatas = {"typ1": typdata, "typ2": typdata}
    pandapipes.create_std_types(net, component="pipe", type_dict=typdatas)
    av = pandapipes.available_std_types(net, component="pipe")
    assert av.to_dict(orient="index") == net.std_types["pipe"]


if __name__ == "__main__":
    pytest.main([__file__])
