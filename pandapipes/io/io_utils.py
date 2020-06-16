# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import importlib
import json
from functools import partial
from inspect import isclass

import numpy
import pandas as pd
from networkx.readwrite import json_graph
from pandapipes.component_models.abstract_models import Component
from pandapipes.pandapipes_net import pandapipesNet
from pandapower.io_utils import with_signature, to_serializable, JSONSerializableClass, \
    isinstance_partial as ppow_isinstance
from pandapipes.create import create_empty_network as create_fluid_network

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
    if '_module' in d and '_class' in d:
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
            obj = {key: val for key, val in d.items() if key not in ['_module', '_class']}
        class_name = d.pop('_class')
        module_name = d.pop('_module')

        if class_name == 'Series':
            return pd.read_json(obj, precise_float=True, **d)
        elif class_name == "DataFrame":
            df = pd.read_json(obj, precise_float=True, **d)
            try:
                df.set_index(df.index.astype(numpy.int64), inplace=True)
            except (ValueError, TypeError, AttributeError):
                logger.debug("failed setting int64 index")
            # recreate jsoned objects
            for col in ('object', 'controller'):  # "controller" for backwards compatibility
                if col in df.columns:
                    df[col] = df[col].apply(ppipes_hook, args=(net,))
            return df
        elif GEOPANDAS_INSTALLED and class_name == 'GeoDataFrame':
            df = geopandas.GeoDataFrame.from_features(fiona.Collection(obj), crs=d['crs'])
            if "id" in df:
                df.set_index(df['id'].values.astype(numpy.int64), inplace=True)
            # coords column is not handled properly when using from_features
            if 'coords' in df:
                # df['coords'] = df.coords.apply(json.loads)
                valid_coords = ~pd.isnull(df.coords)
                df.loc[valid_coords, 'coords'] = df.loc[valid_coords, "coords"].apply(json.loads)
            df = df.reindex(columns=d['columns'])
            return df
        elif SHAPELY_INSTALLED and module_name == "shapely":
            return shapely.geometry.shape(obj)
        elif class_name == "pandapipesNet":
            net = create_fluid_network(add_stdtypes=False)
            net.update(obj)
            return net
        elif class_name == "pandapowerNet":
            if isinstance(obj, str):  # backwards compatibility
                from pandapower import from_json_string
                return from_json_string(obj)
            else:
                # net = create_empty_network()
                net.update(obj)
                return net
        elif module_name == "networkx":
            return json_graph.adjacency_graph(obj, attrs={'id': 'json_id', 'key': 'json_key'})
        else:
            module = importlib.import_module(module_name)
            if class_name == "method":
                logger.warning('Deserializing of method not tested. This might fail...')
                func = getattr(module, obj)  # doesn't always work
                return func
            elif class_name == "function":
                class_ = getattr(module, obj)  # works
                return class_
            class_ = getattr(module, class_name)
            if isclass(class_) and issubclass(class_, JSONSerializableClass):
                if isinstance(obj, str):
                    obj = json.loads(obj, cls=PPJSONDecoder)  # backwards compatibility
                return class_.from_dict(obj, net)
            if isclass(class_) and issubclass(class_, Component):
                return class_
            else:
                # for non-pp objects, e.g. tuple
                return class_(obj, **d)
    else:
        return d


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
