import sys
import warnings
import os


# ----------------------------------------------------------------------------------------------------------------------
#                                                   Pretty Prints
# ----------------------------------------------------------------------------------------------------------------------
def banner(text=None, ch='=', length=88):
    if text is not None:
        spaced_text = ' %s ' % text
    else:
        spaced_text = ''
    print(spaced_text.center(length, ch))


class BColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


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


def convert_bytes(num):
    """
    this function will convert bytes to MB.... GB... etc
    """
    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if num < 1024.0:
            return "%3.1f %s" % (num, x)
        num /= 1024.0


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
