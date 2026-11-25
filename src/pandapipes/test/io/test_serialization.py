import dataclasses
import json
import warnings
from collections.abc import Mapping

import pytest

import pandapipes as pp
from pandapipes.io import serialization
from pandapipes.pf import friction_factor_model as fm


@pytest.fixture(autouse=True)
def _restore_registry():
    snap = dict(serialization._REGISTRY)
    yield
    serialization._REGISTRY.clear()
    serialization._REGISTRY.update(snap)


@pytest.fixture
def empty_registry(_restore_registry):
    serialization._REGISTRY.clear()
    yield


def test_serializable_registers_under_class_name(empty_registry):
    @pp.serializable
    class A: ...

    assert serialization._REGISTRY["A"] is A


def test_all_built_ins_are_registered():
    expected = {"Colebrook", "Nikuradse", "SwameeJain"}
    actual = {
        k for k, v in serialization._REGISTRY.items() if v.__module__ == fm.__name__
    }
    assert actual == expected


def test_reregistering_same_name_with_different_class_warns():
    @pp.serializable
    class IAmOverriden: ...

    with pytest.warns(UserWarning, match="overrides"):

        @pp.serializable
        class IAmOverriden: ...


def test_reregistering_same_class_is_silent():
    @pp.serializable
    class A: ...

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # must not warn
        pp.serializable(A)


def test_friction_model_encodes_by_class_name(model_class):
    net = pp.create_empty_network("", "lgas", add_stdtypes=False)
    pp.set_user_pf_options(net, friction_model=model_class())
    pp.init_options(net)

    net_dict = json.loads(pp.to_json(net, None))
    friction_model = net_dict["_object"]["user_pf_options"]["friction_model"]
    assert isinstance(friction_model, Mapping)
    # pins pandapower's key names on purpose: a rename there should fail here first
    assert friction_model["_class"] == model_class.__name__
    assert friction_model["_module"] == fm.__name__


def test_friction_model_roundtrips(model_class):
    net = pp.create_empty_network("", "lgas", add_stdtypes=False)
    pp.set_user_pf_options(net, friction_model=model_class())
    pp.init_options(net)

    net = pp.from_json_string(pp.to_json(net, None))
    friction_model = net["user_pf_options"]["friction_model"]
    assert isinstance(friction_model, model_class)


def test_user_subclass_roundtrips_as_own_name(empty_registry):
    @pp.serializable
    @dataclasses.dataclass(slots=True)
    class MyNikuradse(fm.Nikuradse): ...

    net = pp.create_empty_network("", "lgas", add_stdtypes=False)
    pp.set_user_pf_options(net, friction_model=MyNikuradse())
    pp.init_options(net)

    net = pp.from_json_string(pp.to_json(net, None))
    assert isinstance(net["user_pf_options"]["friction_model"], MyNikuradse)


@pytest.mark.filterwarnings(
    "ignore:.*overrides.*:UserWarning"
)  # we assert the decode warning, not the registration one
def test_decode_warns_when_name_resolves_to_class_from_another_module():
    net = pp.create_empty_network("", "lgas", add_stdtypes=False)
    pp.set_user_pf_options(net, friction_model=pp.Nikuradse())
    pp.init_options(net)

    json_str = pp.to_json(net, None)

    @pp.serializable
    class Nikuradse: ...

    with pytest.warns(UserWarning, match="resolved to"):
        pp.from_json_string(json_str)
