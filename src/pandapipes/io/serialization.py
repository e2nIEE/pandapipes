import warnings

from pandapower.io_utils import to_serializable, with_signature

_REGISTRY = {}
_MARKER = "__pandapipes_serializable__"


class SerializableNotRegistered(LookupError):
    """A net references a class that is not in the registry here."""


def to_json(obj):
    # String sentinel: marks this payload as registry-produced without
    # carrying state (the class name is the entire payload).
    return with_signature(obj, _MARKER)


def _find_registered_class(class_name, module_name):
    """Return the class registered under ``class_name``, or ``None``.

    Warns when the resolved class's module differs from ``module_name``.
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


def _restore_registered(class_name, module_name, payload):
    """Instantiate the class registered under ``class_name``.

    Returns ``None`` if ``payload`` does not carry our marker (i.e. this
    entry was not produced by a ``@serializable`` class). Raises
    ``SerializableNotRegistered`` if the marker is present but the class is
    no longer registered.
    """
    if payload != _MARKER:
        return None
    cls = _find_registered_class(class_name, module_name)
    if cls is None:
        msg = (
            f"{class_name!r} was saved as a registered class but is not in "
            f"the registry. Register it with @pp.serializable before "
            f"loading this net."
        )
        raise SerializableNotRegistered(msg)
    return cls()


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
