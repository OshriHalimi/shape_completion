import os


# ----------------------------------------------------------------------------------------------------------------------
#                                             File System
# ----------------------------------------------------------------------------------------------------------------------

def align_file_extension(fp, tgt):
    full_tgt = tgt if tgt[0] == '.' else f'.{tgt}'
    fp = str(fp)  # Support for pathlib.Path
    if fp.endswith(full_tgt):
        return fp
    else:
        return fp + full_tgt


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
    except:  # No such dir # TODO - detect the explict error
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
