import sys
from pathlib import Path
import warnings
import os

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
def assert_is_dir(dir):
    assert Path(dir).is_dir(),f"Directory {dir} is invalid"
# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
def banner(text=None, ch='=', length=88):
    if text is not None:
        spaced_text = ' %s ' % text
    else:
        spaced_text = ''
    print(spaced_text.center(length, ch))

def print_warning(message, category, filename, lineno, file=None, line=None):

    if line is None:
        try:
            import linecache
            line = linecache.getline(filename, lineno)
        except Exception:
            # When a warning is logged during Python shutdown, linecache
            # and the import machinery don't work anymore
            line = None
            linecache = None
    else:
        line = line
    if line:
        line = line.strip()

    filename = os.path.basename(filename)
    print(bcolors.WARNING + f'At {filename}:{lineno} : {line}\nWARNING: {message}'+ bcolors.ENDC)

warnings.showwarning = print_warning
def warn(str,stacklevel=1):
    warnings.warn(str,stacklevel=stacklevel+1)

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

def convert_bytes(num):
    """
    this function will convert bytes to MB.... GB... etc
    """
    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if num < 1024.0:
            return "%3.1f %s" % (num, x)
        num /= 1024.0
