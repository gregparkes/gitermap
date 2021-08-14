"""Python file for handling utility functions """

import warnings
import os


def check_file_path(path: str, create_folder: bool, raise_error=True, verbose: int = 0):
    """Check the file path."""
    if raise_error and create_folder:
        warnings.warn("`raise_error` and `create_folder` cannot both be true, defaulting to `raise_error`")

    def _remove_garb(x: str):
        return x != "" and x != ".." and x.find(".") == -1

    # get a list of folders in order
    folders = list(filter(_remove_garb, path.split("/")))

    if verbose > 1:
        print("folders: {}".format(folders))

    # iterate through each folder subtype and check
    for f in folders:
        constructed_string = path[:path.find(f) + len(f)]
        if verbose > 1:
            print("folder step: " + constructed_string)
        # does this subpart exist
        if not os.path.isdir(constructed_string):
            absp = os.path.abspath(constructed_string)
            # now create, raise or warn
            if raise_error:
                # raise an error
                raise FileNotFoundError("directory at '{}' does not exist".format(absp))
            elif create_folder:
                if verbose > 0:
                    print("creating folder at '{}'".format(absp))
                os.mkdir(constructed_string)
            else:
                warnings.warn("directory at '{}' does not exist".format(absp), UserWarning)
    return True


def is_tqdm_installed(raise_error: bool = False):
    """Determines whether tqdm is installed."""
    try:
        from tqdm import tqdm  # noqa
        is_installed = True
    except ModuleNotFoundError:  # pragma: no cover
        is_installed = False
    # Raise error (if needed) :
    if raise_error and not is_installed:  # pragma: no cover
        raise ModuleNotFoundError("tqdm not installed. Use `pip " "install tqdm`.")
    return is_installed


def is_numpy_installed(raise_error: bool = False):
    """Determines whether numpy is installed."""
    try:
        import numpy as np  # noqa
        is_installed = True
    except ModuleNotFoundError:  # pragma: no cover
        is_installed = False
    # Raise error (if needed) :
    if raise_error and not is_installed:  # pragma: no cover
        raise ModuleNotFoundError("numpy not installed. Use `pip " "install numpy`.")
    return is_installed


def is_pandas_installed(raise_error: bool = False):
    """Determines whether pandas is installed."""
    try:
        import pandas  # noqa
        is_installed = True
    except ModuleNotFoundError:  # pragma: no cover
        is_installed = False
    # Raise error (if needed) :
    if raise_error and not is_installed:  # pragma: no cover
        raise ModuleNotFoundError("pandas not installed. Use `pip " "install pandas`.")
    return is_installed


def is_simpleaudio_installed(raise_error: bool = False):
    """Determines whether pandas is installed."""
    try:
        import simpleaudio  # noqa
        is_installed = True
    except ModuleNotFoundError:  # pragma: no cover
        is_installed = False
    # Raise error (if needed) :
    if raise_error and not is_installed:  # pragma: no cover
        raise ModuleNotFoundError("simpleaudio not installed. Use `pip " "install simpleaudio`.")
    return is_installed


def directory_info(fn: str):
    """Returns the relative file and absolute cache directory information"""
    # create a cache directory in the directory below where to plant the file
    fnspl = fn.rsplit("/", 1)
    if len(fnspl) == 2:
        parent_rel, just_file = fnspl
        parent_abs = os.path.abspath(parent_rel)
    else:
        just_file = fnspl[0]
        parent_rel = "."
        parent_abs = os.getcwd()

    abscachedir = os.path.join(parent_abs, "_tmp_umapcc_")
    relfile = os.path.join(os.path.join(parent_rel, "_tmp_umapcc_"), just_file)
    return relfile, abscachedir


def create_cache_directory(fn: str):
    """Create a cache directory"""
    relfile, abscachedir = directory_info(fn)
    # if it doesn't already exist, create it.
    if not os.path.isdir(abscachedir):
        os.mkdir(abscachedir)

    return relfile, abscachedir


def add_suffix(ins: int, filename: str) -> str:
    """Inserts a string to the end of a file name.

    Parameters
    ----------
    ins : int
        The insertable number character
    filename : str
        The filename full string

    Returns
    -------
    new_filename : str
        The new filename full string
    """
    lhs, rhs = filename.rsplit(".", 1)
    return lhs + str(ins) + "." + rhs
