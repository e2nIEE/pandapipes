import warnings

from pandapower.io_utils import to_serializable, with_signature

_REGISTRY = {}


def to_json(obj):
    # No state: the class name is the entire payload.
    return with_signature(obj, {})


def _resolve(class_name, module_name):
    """Return the class registered under ``class_name``, or ``None``.

    Warns when the registered class comes from a module other than
    ``module_name`` -- i.e. the name resolved to a different class than the
    one recorded in the file.
    """
    cls = _REGISTRY.get(class_name)
    if cls is None:
        return None
    if cls.__module__ != module_name:
        msg = (
            f"{class_name!r} from {module_name} resolved to "
            f"{cls.__module__}.{class_name}"
        )
        warnings.warn(msg)
    return cls


def serializable(cls):
    """Register ``cls`` for name-only JSON serialization.

    Instances are serialized to the class name only (via pandapower's
    ``with_signature``); on load, ``cls()`` is called, so the class must be
    constructible with no arguments. Re-registering a name with a different
    class overwrites it and warns.
    """
    name = cls.__name__
    existing = _REGISTRY.get(name)
    if existing is not None and existing is not cls:
        msg = f"{name!r} overrides {existing!r}"
        warnings.warn(msg, stacklevel=2)

    _REGISTRY[name] = cls
    to_serializable.register(cls)(to_json)
    return cls
