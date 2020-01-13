import sys
import warnings
import os
import timeit
from datetime import timedelta
import inspect
from types import FunctionType

# ----------------------------------------------------------------------------------------------------------------------
#                                                   Pretty Prints
# ----------------------------------------------------------------------------------------------------------------------
def banner(text=None, ch='=', length=88):
    if text is not None:
        spaced_text = ' %s ' % text
    else:
        spaced_text = ''
    print(spaced_text.center(length, ch))


def tutorial(func):
    def wrapper(*args, **kwargs):
        banner(title(func.__name__))
        return func(*args,**kwargs)
    return wrapper


def title(s):
    s = s.replace('_',' ')
    s = s.replace('-', ' ')
    s = s.title()
    return s

def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60.
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)

class BColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# Decorator to time functions
def time_me(method):
    def timed(*args, **kw):
        ts = timeit.default_timer()
        result = method(*args, **kw)
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
        print(f'{method.__name__} compute time :: {str(timedelta(seconds=te - ts))}')
        return result

    return timed


def print_warning(message, category, filename, lineno, file=None, line=None):
    # if line is None:
    #     try:
    #         import linecache
    #         line = linecache.getline(filename, lineno)
    #     except Exception:
    #         # When a warning is logged during Python shutdown, linecache
    #         # and the import machinery don't work anymore
    #         line = None
    #         linecache = None
    # else:
    #     line = line
    # if line:
    #     line = line.strip()

    filename = os.path.basename(filename)
    print(BColors.WARNING + f'{filename}:{lineno}:\nWARNING: {message}' + BColors.ENDC)


warnings.showwarning = print_warning


def warn(s, stacklevel=1):
    warnings.warn(s, stacklevel=stacklevel + 1)


# ----------------------------------------------------------------------------------------------------------------------
#                                             File System
# ----------------------------------------------------------------------------------------------------------------------
def assert_is_dir(d):
    assert os.path.isdir(d), f"Directory {d} is invalid"

def get_exp_version(cache_dir):
    last_version = -1
    try:
        for f in os.listdir(cache_dir):
            if 'version_' in f:
                file_parts = f.split('_')
                version = int(file_parts[-1])
                last_version = max(last_version, version)
    except:  # No such dir
        pass

    return last_version + 1

def convert_bytes(num):
    """
    this function will convert bytes to MB.... GB... etc
    """
    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if num < 1024.0:
            return "%3.1f %s" % (num, x)
        num /= 1024.0

# def make_directory(name):
#     result = Path("result")
#     result.mkdir(exist_ok=True)
#     if name is not None:
#         dir_name = name
#     else:
#         now = datetime.datetime.now()
#         dir_name = datetime.datetime.strftime(now, "%y_%m_%d_%H")
#     log_dir = result / dir_name
#     log_dir.mkdir(exist_ok=True)
#
#     return log_dir

# ----------------------------------------------------------------------------------------------------------------------
#                                               Arguments
# ----------------------------------------------------------------------------------------------------------------------

def none_or_str(value):
    if value == 'None':
        return None
    return value


def none_or_int(value):
    if value == 'None':
        return None
    return int(value)


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


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
    return set(x for x, y in o.__dict__.items() if isinstance(y, (FunctionType,classmethod,staticmethod)))

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


# ----------------------------------------------------------------------------------------------------------------------
#
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