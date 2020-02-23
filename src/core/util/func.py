import inspect
from types import FunctionType
from util.strings import banner, title
import timeit
from datetime import timedelta
from functools import wraps


# ----------------------------------------------------------------------------------------------------------------------
#                                                       Useful Decorators
# ----------------------------------------------------------------------------------------------------------------------

def tutorial(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        banner(title(func.__name__))
        return func(*args, **kwargs)

    return wrapper


def time_me(func):
    @wraps(func)
    def timed(*args, **kw):
        ts = timeit.default_timer()
        result = func(*args, **kw)
        te = timeit.default_timer()
        # This snippet here allows extraction of the timing:
        # Snippet:
        # if 'log_time' in kw:
        #     name = kw.get('log_name', method.__name__.upper()) # Key defaults to method name
        #     kw['log_time'][name] = int((te - ts) * 1000)
        # Usage:
        # logtime_data = {}
        # ret_val = some_func_with_decorator(log_time=logtime_data)
        # else:
        print(f'{func.__name__} compute time :: {str(timedelta(seconds=te - ts))}')
        return result

    return timed


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------

def list_class_declared_methods(o):
    # dynasty - parent = class_declared
    # narrow_class - parent_methods = class_declared
    # Only the new methods - not related to the parent class
    parent_methods = list_parent_class_methods(o)
    only_in_class_methods = list_narrow_class_methods(o)
    # Now remove the intersection
    return only_in_class_methods - parent_methods


def list_narrow_class_methods(o):
    # Class Only Methods
    if not inspect.isclass(o):
        o = o.__class__
    return set(x for x, y in o.__dict__.items() if isinstance(y, (FunctionType, classmethod, staticmethod)))


def list_dynasty_class_methods(o):
    # Class + Parent Class Methods
    if not inspect.isclass(o):
        o = o.__class__
    return {func for func in dir(o) if callable(getattr(o, func))}
    # # https://docs.python.org/3/library/inspect.html#inspect.isclass
    # TODO - Many objects inside the class are callable as well - this is a problem. Depends on definition.


def list_parent_class_methods(o):
    if not inspect.isclass(o):
        o = o.__class__

    parent_methods = set()
    for c in o.__bases__:
        parent_methods |= list_dynasty_class_methods(c)
        # parent_methods |= list_parent_class_methods(c) # Recursive Tree DFS - Removed
    return parent_methods


def func_name():
    import traceback
    return traceback.extract_stack(None, 2)[0][2]


def all_variables_by_module_name(module_name):
    from importlib import import_module
    from types import ModuleType  # ,ClassType
    module = import_module(module_name)
    return {k: v for k, v in module.__dict__.items() if
            not (k.startswith('__') or k.startswith('_'))
            and not isinstance(v, ModuleType)}
    # and not isinstance(v,ClassType)}

# ----------------------------------------------------------------------------------------------------------------------
#                                                  Testing Suite
# ----------------------------------------------------------------------------------------------------------------------
#
# class Parent:
#     PARENT_STATIC = 1
#
#     def __init__(self):
#         self.father_inside = 5
#
#     def papa(self):
#         pass
#
#     def mama(self):
#         pass
#
#     @classmethod
#     def parent_class(cls):
#         pass
#
#     @staticmethod
#     def parent_static():
#         pass
#
#
# class Son(Parent):
#     SON_VAR = 1
#
#     def __init__(self):
#         super().__init__()
#         self.son_inside = 1
#
#     def papa(self):
#         pass
#
#     def child(self):
#         pass
#
#     @classmethod
#     def son_class(cls):
#         pass
#
#     @staticmethod
#     def son_static():
#         pass
