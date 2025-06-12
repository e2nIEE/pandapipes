try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def deprecated_input(input_handler, multiple=False):
    def decorator(f):
        def wrap(*args, **kwargs):
            args, kwargs = input_handler(multiple, *args, **kwargs)
            return f(*args, **kwargs)
        return wrap
    return decorator

def input_handler_valve(multiple=False, *args, **kwargs):
    suffix = "s" if multiple else ""
    if ("from_junction" + suffix in kwargs) | ("to_junction" + suffix in kwargs):
        kwargs['et'] = 'j'
    if "from_junction" + suffix in kwargs:
        if "junction" + suffix in kwargs:
            raise UserWarning(r"If you define 'junction%s', "
                              r"do not use the deprecated variable 'from_junction%s' as well!" %(suffix, suffix))
        print(r"from_junction%s is deprecated and will be removed in the future. "
                       r"Use junction%s instead!" %(suffix, suffix))
        kwargs['junction' + suffix] = kwargs['from_junction' + suffix]
        kwargs.pop('from_junction' + suffix)
    if "to_junction" + suffix in kwargs:
        if "element" + suffix in kwargs:
            raise UserWarning(r"If you define 'element%s', "
                              r"do not use the deprecated variable 'to_junction%s' as well" %(suffix, suffix))
        print(r"to_junction%s is deprecated and will be removed in the future. "
                       r"Use element%s instead and set et equal to 'j'!" %(suffix, suffix))
        kwargs['element' + suffix] = kwargs['to_junction' + suffix]
        kwargs.pop('to_junction' + suffix)
    return args, kwargs