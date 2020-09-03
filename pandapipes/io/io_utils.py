# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import importlib
import json
from functools import partial
from inspect import isclass

import pandapower as pp
from pandapipes.component_models.abstract_models import Component
from pandapipes.create import create_empty_network as create_fluid_network
from pandapipes.pandapipes_net import pandapipesNet
from pandapower.io_utils import pp_hook
from pandapower.io_utils import with_signature, to_serializable, JSONSerializableClass, \
    isinstance_partial as ppow_isinstance, FromSerializableRegistry, PPJSONDecoder

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)

try:
    import fiona
    import geopandas

    GEOPANDAS_INSTALLED = True
except ImportError:
    GEOPANDAS_INSTALLED = False

try:
    import shapely.geometry

    SHAPELY_INSTALLED = True
except (ImportError, OSError):
    SHAPELY_INSTALLED = False


def isinstance_partial(obj, cls):
    if isinstance(obj, pandapipesNet):
        return False
    return ppow_isinstance(obj, cls)


class FromSerializableRegistryPpipe(FromSerializableRegistry):
    from_serializable = FromSerializableRegistry.from_serializable
    class_name = ''
    module_name = ''

    def __init__(self, obj, d, net, ppipes_hook):
        super().__init__(obj, d, net, ppipes_hook)

    @from_serializable.register(class_name='pandapowerNet', module_name='pandapower.auxiliary')
    def pandapowerNet(self):
        if isinstance(self.obj, str):  # backwards compatibility
            from pandapower import from_json_string
            return from_json_string(self.obj)
        else:
            net = pp.create_empty_network()
            net.update(self.obj)
            return net

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
            from pandapipes import from_json_string
            return from_json_string(self.obj)
        else:
            self.net.update(self.obj)
            return self.net

    @from_serializable.register()
    def rest(self):
        module = importlib.import_module(self.module_name)
        class_ = getattr(module, self.class_name)
        if isclass(class_) and issubclass(class_, JSONSerializableClass):
            if isinstance(self.obj, str):
                self.obj = json.loads(self.obj, cls=PPJSONDecoder,
                                      object_hook=partial(pp_hook, net=self.net,
                                                          registry_class=FromSerializableRegistryPpipe))
                                                          # backwards compatibility
            return class_.from_dict(self.obj, self.net)
        if isclass(class_) and issubclass(class_, Component):
            return class_
        else:
            # for non-pp objects, e.g. tuple
            return class_(self.obj, **self.d)


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


if __name__ == '__main__':
    ntw = create_fluid_network()
    import pandapipes as pp

    pp.to_json(ntw, 'test.json')
    ntw = pp.from_json('test.json')
