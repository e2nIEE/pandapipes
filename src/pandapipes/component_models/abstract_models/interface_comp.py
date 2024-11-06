from pandapipes.component_models.component_toolbox import init_results_element

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)

from abc import ABCMeta, abstractmethod


class NoRequirements(RuntimeError):
    def __init__(self, message):
        RuntimeError.__init__(self, message)


class ForceAttrMeta(ABCMeta):
    """
    Metaclass that forces child classes to define specific attributes listed in the required_attributes
    of the interface class
    """
    def __new__(mcls, name, bases, attrs):
        def get_missing_requirements(c, b):
            """
            iterates over the required attributes from the parent class and
            checks their existence in the class in creation
            """
            missing = list()
            for p in b:
                for req in p.required_attributes:
                    if not hasattr(c, req):
                        missing.append(req)
            return missing

        # @abstractmethod
        # def prevent_instantiating():
        #     pass

        #attrs[prevent_instantiating.__name__] = prevent_instantiating
        cls = super().__new__(mcls, name, bases, attrs)
        # check if parent class has the attribute required_attributes if parent class is the metaclass
        basic_metaclass = True
        for parent in bases:
            if type(parent) == ForceAttrMeta:  # and not pass attrs
                basic_metaclass = False
                if not hasattr(parent, 'required_attributes'):
                    raise NotImplementedError(f"Class '{parent.__name__}' has no 'required_attributes' property")

        if basic_metaclass:
            return cls

        missingreqs = get_missing_requirements(cls, bases)
        if len(missingreqs) > 0:
            raise NotImplementedError(
                f"Class '{cls.__name__}' has not implemented the following attributes: {str(missingreqs)[1:-1]}")

        return cls
