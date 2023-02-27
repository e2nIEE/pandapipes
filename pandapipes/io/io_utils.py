# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import importlib
import json
from copy import deepcopy
from functools import partial
from inspect import isclass

from pandapower.io_utils import pp_hook
from pandapower.io_utils import with_signature, to_serializable, JSONSerializableClass, \
    isinstance_partial as ppow_isinstance, FromSerializableRegistry, PPJSONDecoder

from pandapipes.component_models.abstract_models.branch_models import Component
from pandapipes.multinet.create_multinet import MultiNet, create_empty_multinet
from pandapipes.pandapipes_net import pandapipesNet, get_basic_net_entries

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


MODULE_CHANGES = {"PumpStdType": "pandapipes.std_types.std_type_class",
                  "StdType": "pandapipes.std_types.std_type_class"}


def isinstance_partial(obj, cls):
    if isinstance(obj, pandapipesNet):
        return False
    elif isinstance(obj, MultiNet):
        return False
    return ppow_isinstance(obj, cls)


class FromSerializableRegistryPpipe(FromSerializableRegistry):
    from_serializable = deepcopy(FromSerializableRegistry.from_serializable)
    class_name = ''
    module_name = ''

    def __init__(self, obj, d, ppipes_hook):
        """

        :param obj: object the data is written to
        :type obj: object
        :param d: data to be re-serialized
        :type d: any kind
        :param ppipes_hook: a way how to handle non-default data
        :type ppipes_hook: funct
        """
        super().__init__(obj, d, ppipes_hook)

    @from_serializable.register(class_name="method")
    def method(self):
        module = importlib.import_module(self.module_name)
        logger.warning('Deserializing of method not tested. This might fail...')
        func = getattr(module, self.obj)
        # class_ = getattr(module, obj) # doesn't work
        return func

    @from_serializable.register(class_name='pandapipesNet', module_name='pandapipes.pandapipes_net')
    def pandapipesNet(self):
        if isinstance(self.obj, str):  # backwards compatibility
            from pandapipes.io.file_io import from_json_string
            return from_json_string(self.obj)
        else:
            entries = get_basic_net_entries()
            entries =  {k: entries[k] for k in entries if k in self.obj}
            net = pandapipesNet(entries)
            net.update(self.obj)
            return net

    @from_serializable.register()
    def rest(self):
        try:
            module = importlib.import_module(self.module_name)
        except ModuleNotFoundError as e:
            if self.class_name in MODULE_CHANGES:
                module = importlib.import_module(MODULE_CHANGES[self.class_name])
            else:
                raise e
        class_ = getattr(module, self.class_name)
        if isclass(class_) and issubclass(class_, JSONSerializableClass):
            if isinstance(self.obj, str):
                self.obj = json.loads(self.obj, cls=PPJSONDecoder,
                                      object_hook=partial(pp_hook,
                                                          registry_class=FromSerializableRegistryPpipe))
                # backwards compatibility
            if "net" in self.obj:
                del self.obj["net"]
            return class_.from_dict(self.obj)
        if isclass(class_) and issubclass(class_, Component):
            return class_
        else:
            # for non-pp objects, e.g. tuple
            return class_(self.obj, **self.d)

    @from_serializable.register(class_name='MultiNet')
    def MultiNet(self):
        if isinstance(self.obj, str):  # backwards compatibility
            from pandapipes.io.file_io import from_json_string
            return from_json_string(self.obj)
        else:
            net = create_empty_multinet()
            net.update(self.obj)
            return net


@to_serializable.register(pandapipesNet)
def json_net(obj):
    net_dict = {k: item for k, item in obj.items() if not k.startswith("_")}
    d = with_signature(obj, net_dict)
    return d


@to_serializable.register(type)
def json_component(class_):
    if issubclass(class_, Component):
        d = with_signature(class_(), str(class_().__dict__))
        return d
    else:
        raise (UserWarning('with_signature needs to be defined for '
                           'class %s in @to_serializable.register(type)!' % class_))


@to_serializable.register(MultiNet)
def json_net(obj):
    net_dict = {k: item for k, item in obj.items() if not k.startswith("_")}
    d = with_signature(obj, net_dict)
    return d
