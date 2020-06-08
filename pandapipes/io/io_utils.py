# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import importlib
import json
from functools import partial
from inspect import isclass

import pandapower as pp
from pandapower.io_utils import with_signature, to_serializable, JSONSerializableClass, \
    isinstance_partial as ppow_isinstance, from_serializable_registry, from_serializable

from pandapipes.component_models.abstract_models import Component
from pandapipes.create import create_empty_network as create_fluid_network
from pandapipes.pandapipes_net import pandapipesNet

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


class PPJSONDecoder(json.JSONDecoder):
    def __init__(self, **kwargs):
        net = create_fluid_network(add_stdtypes=False)
        super_kwargs = {"object_hook": partial(ppipes_hook, net=net)}
        super_kwargs.update(kwargs)
        super().__init__(**super_kwargs)

def ppipes_hook(d, net=None):
    if "_object" in d:
        obj = d.pop('_object')
    elif "_state" in d:
        obj = d['_state']
        if d['has_net']:
            obj['net'] = 'net'
        if '_init' in obj:
            del obj['_init']
        return obj  # backwards compatibility
    else:
        # obj = {"_init": d, "_state": dict()}  # backwards compatibility
        obj = {key: val for key, val in d.items() if key not in ['_module', '_class']}
    return ppipes_hook_serialization(obj, d, net=net)


def ppipes_hook_serialization(obj, d, net):
    if '_module' in d and '_class' in d:
        class_name = d.pop('_class')
        module_name = d.pop('_module')
        fs = from_serializable_registry_ppipe(obj, d, net)
        fs.class_name = class_name
        fs.module_name = module_name
        return fs.from_serializable()
    else:
        return d


class from_serializable_registry_ppipe(from_serializable_registry):
    from_serializable = from_serializable()
    class_name = ''
    module_name = ''

    def __init__(self, obj, d, net):
        super().__init__(obj, d, net)

    @from_serializable.register(class_name='Series', module_name='pandas.core.series')
    def Series(self):
        return super().Series()

    @from_serializable.register(class_name='DataFrame', module_name='pandas.core.frame')
    def DataFrame(self):
        return super().DataFrame()

    @from_serializable.register(module_name="networkx")
    def networkx(self):
        return super().networkx()

    @from_serializable.register(class_name='function')
    def function(self):
        return super().function()

    if GEOPANDAS_INSTALLED:
        @from_serializable.register(class_name='GeoDataFrame')
        def GeoDataFrame(self):
            return super().GeoDataFrame()

    if SHAPELY_INSTALLED:
        @from_serializable.register(module_name='shapely')
        def shapely(self):
            return super().shapely()

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

    @from_serializable.register(class_name='pandapipesNet', module_name='pandapower.pandapipes_net')
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
                obj = json.loads(self.obj, cls=PPJSONDecoder)  # backwards compatibility
            return class_.from_dict(obj, self.net)
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
