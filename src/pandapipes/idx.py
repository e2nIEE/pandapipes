# Copyright (c) 2020-2026 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.


class IndexMeta(type):
    """Metaclass that collects a class's ``int``-valued attributes into an iterable dict.

    The resulting ``_indices`` dict is merged with any base classes' ``_indices`` (so
    subclasses inherit and can extend their parent's index set).
    """

    def __new__(cls, name, bases, classdict):
        """Create the class and collect its int-valued attributes into a merged ``_indices`` dict."""
        clsobj = super().__new__(cls, name, bases, classdict)

        clsobj._indices = {
            k: v for k, v in classdict.items()
            if not k.startswith("__") and isinstance(v, int)
        }

        for base in bases:
            if hasattr(base, "_indices"):
                clsobj._indices = {**base._indices, **clsobj._indices}

        return clsobj

    def __iter__(cls):
        """Iterate over ``(name, value)`` pairs of the class's collected index attributes."""
        return iter(cls._indices.items())

    def keys(cls):
        return cls._indices.keys()

    def values(cls):
        return cls._indices.values()

    def items(cls):
        return cls._indices.items()
