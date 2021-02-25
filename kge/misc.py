from typing import List, Union

from torch import nn as nn
import os
from path import Path
import inspect
import subprocess
import types

import importlib


def init_from(
    class_name: str, modules: List[Union[str, types.ModuleType]], *args, **kwargs
):
    """Initializes class from its name and list of module names it might be part of.

    Args:
        class_name: the name of the class that is to be initialized.
        modules: the list of modules or module names that are to be searched for
                 the class.
        *args: the non-keyword arguments for the constructor of the given class.
        **kwargs: the keyword arguments for the constructor of the given class.

    Returns:
        An instantiation of the class that first matched the given name during the
        search through the given modules.

    Raises:
        ValueError: If the given class cannot be found in any of the given modules.
    """
    modules = [
        m if isinstance(m, types.ModuleType) else importlib.import_module(m)
        for m in modules
    ]
    for module in modules:
        if hasattr(module, class_name):
            return getattr(module, class_name)(*args, **kwargs)
    else:
        raise ValueError(
            f"Can't find class {class_name} in modules {[m.__name__ for m in modules]}"
        )


def is_number(s, number_type):
    """ Returns True is string is a number. """
    try:
        number_type(s)
        return True
    except ValueError:
        return False


# from https://stackoverflow.com/questions/14989858/get-the-current-git-hash-in-a-python-script
def get_git_revision_hash():
    try:
        if which("git") is not None:
            with Path(kge_base_dir()):
                return (
                    subprocess.check_output(["git", "rev-parse", "HEAD"])
                    .strip()
                    .decode()
                )
        else:
            return "No git binary found"
    except:
        return "No working git repository found."


# from https://stackoverflow.com/questions/14989858/get-the-current-git-hash-in-a-python-script
def get_git_revision_short_hash():
    try:
        if which("git") is not None:
            with Path(kge_base_dir()):
                return (
                    subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
                    .strip()
                    .decode()
                )
        else:
            return "No git binary found"
    except:
        return "No working git repository found."


# from https://stackoverflow.com/questions/377017/test-if-executable-exists-in-python
def which(program):
    import os

    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None


def module_base_dir(module_name):
    module = importlib.import_module(module_name)
    return os.path.abspath(filename_in_module(module, ".."))


def kge_base_dir():
    return module_base_dir("kge")


def filename_in_module(module_or_module_list, filename):
    if not isinstance(module_or_module_list, list):
        module_or_module_list = [module_or_module_list]
    for module in module_or_module_list:
        f = os.path.dirname(inspect.getfile(module)) + "/" + filename
        if os.path.exists(f):
            return f
    raise FileNotFoundError(
        "{} not found in one of modules {}".format(filename, module_or_module_list)
    )


def get_activation_function(s: str):
    if s == "tanh":
        return nn.Tanh()
    elif s == "relu":
        return nn.ReLU()
    else:
        raise ValueError("activation function {} unknown".format(s))


def round_to_points(round_points_to: List[int], to_be_rounded: int):
    """
    Rounds to_be_rounded to the points in round_points_to. Assumes
    that the first element in round_points_to is the lower bound and that
    the last is the upper bound.
    :param round_points_to: List[int]
    :param to_be_rounded: int
    :return: int
    """
    if len(round_points_to) > 0:
        assert (
            round_points_to[0] <= round_points_to[-1]
        ), "First element in round_points_to should be the lower bound and the last the upper bound"
        last = -1
        for i, round_point in enumerate(round_points_to):
            if to_be_rounded < (round_point - last) / 2 + last:
                # Assumes that the first element in round_points_to is
                # the lower bound.
                if i == 0:
                    return round_point
                else:
                    return last
            last = round_point
        return last
    else:
        raise Exception(
            "{} was called with an empty list to be rounded to.".format(
                round_to_points.__name__
            )
        )
