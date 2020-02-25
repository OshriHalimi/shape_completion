import warnings
import logging
import os
import sys


class BColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# ----------------------------------------------------------------------------------------------------------------------
#                                       Hacky, changes global behaviour
# ----------------------------------------------------------------------------------------------------------------------

def print_warning(str):
    print(BColors.WARNING + str + BColors.ENDC)


def print_error(str):
    print(BColors.FAIL + str + BColors.ENDC)


def warn_overload(message, category, filename, lineno, file=None, line=None):
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
    print_warning(f'{filename}:{lineno}:\nWARNING: {message}')


warnings.showwarning = warn_overload


# warnings.simplefilter('always', DeprecationWarning)

def set_logging_to_stdout():
    # Set logging to both the STDOUT and the File
    root = logging.getLogger()
    hdlr = root.handlers[0]
    fmt = logging.Formatter('[%(asctime)s] %(message)s')  # ,'%x %X.%f'
    hdlr.setFormatter(fmt)
    hdlr.stream = sys.stdout


# ----------------------------------------------------------------------------------------------------------------------
#                                                   Pretty Prints
# ----------------------------------------------------------------------------------------------------------------------

def warn(s, stacklevel=1):
    warnings.warn(s, stacklevel=stacklevel + 1)


def banner(text=None, ch='=', length=100):
    if text is not None:
        spaced_text = ' %s ' % text
    else:
        spaced_text = ''
    print(spaced_text.center(length, ch))


def title(s):
    s = s.replace('_', ' ')
    s = s.replace('-', ' ')
    s = s.title()
    return s


def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60.
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)


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
