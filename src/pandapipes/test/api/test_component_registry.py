# Copyright (c) 2020-2024 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

from pandapipes.component_models.component_registry import register_component, ComponentRegistry
from pandapipes.component_models.pipe_component import Pipe
from pandapipes.component_init import COMPONENT_REGISTRY, COMPONENT_LIST


def test_register():
    @register_component
    class TestPipe(Pipe):
        @property
        def table_name(self):
            return "test_pipe"

        @property
        def mymethod(self):  # Lazy-loading of the Junction component
            return ComponentRegistry.get(self.connected_node_type).table_name

    assert TestPipe in COMPONENT_REGISTRY
    assert "test_pipe" in COMPONENT_REGISTRY
    assert TestPipe.__iscustom__
    assert COMPONENT_REGISTRY["test_pipe"].mymethod == "junction"
    assert COMPONENT_REGISTRY[TestPipe].mymethod == "junction"
    assert len(COMPONENT_REGISTRY) == len(COMPONENT_LIST) * 2 + 2
