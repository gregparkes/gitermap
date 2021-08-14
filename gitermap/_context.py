"""Handles the MapContext object."""

import warnings
import os
from functools import partial, reduce
import operator
from joblib import load, dump, cpu_count, Parallel, delayed
import itertools as it
from typing import Callable

from ._utils import is_tqdm_installed, check_file_path, add_suffix, \
    create_cache_directory, directory_info, is_simpleaudio_installed
from ._tqdm_parallel import TqdmParallel
from ._audio import play_arpeggio


class MapContext:
    """A context manager for list comprehensions."""

    def __init__(self,
                 filename: str = None,
                 verbose: int = 0,
                 n_jobs: int = None,
                 chunks: bool = False,
                 return_type: str = "list",
                 progressbar: bool = True,
                 savemode: str = "initial",
                 end_audio: bool = False):
        """Creates a context to wrap list comprehensions in.

        Parameters
        ----------
        filename: str, optional
            A directory to cache files in.
        verbose: int, optional
            Display outputs
        n_jobs : int, optional
            Number of threads to create for multi-threaded operations
        chunks : bool, default=False
            Determines whether caching by chunks occurs, if filename is set (to indicate caching)
        return_type : str, {'list', 'generator'}, default="list"
            Determines the return type from calls to 'compute'
        savemode : str, {'initial', 'override', 'add'}, default="initial"
            Determines how and when to write cache files.
            if savemode=='initial': writes once then reads after
            if savemode=='override': writes every run
            if savemode=='add': writes additional runs every run
        end_audio : bool, default=False
            Whether to play music to signify the ending of the run
        """
        ret_types = {'list', 'generator'}
        save_types = {'initial', 'override', 'add'}
        assert return_type in ret_types, "{} must be in {}".format(return_type, ret_types)
        assert savemode in save_types, "{} must be in {}".format(savemode, save_types)

        self._fn = filename
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.has_chunks = chunks
        self.return_type = return_type
        self.savemode = savemode
        self.progress = progressbar
        self.end_audio = end_audio if is_simpleaudio_installed() else False
        self._Nargs = -1
        self._estN = -1
        self._result = []

    @classmethod
    def _is_generator(cls, obj):
        return hasattr(obj, "__iter__") and not hasattr(obj, "__len__")

    @classmethod
    def _delete_temps(cls, directory):
        if os.path.isdir(directory):
            import shutil
            try:
                shutil.rmtree(directory)
            except OSError as e:
                print("Error: %s - %s." % (e.filename, e.strerror))

    @classmethod
    def _play_success(cls):
        """Plays a positive arpeggio"""
        play_arpeggio("C", "major")

    @classmethod
    def _play_failure(cls):
        """Plays a negative arpeggio"""
        play_arpeggio("C", "minor")

    def _get_parallel_object(self):
        # determine parallel object
        if is_tqdm_installed(False) and self.progress:
            # load custom parallel tqdm object if we use it, else joblib normal.
            if self._estN != -1:
                return TqdmParallel(use_tqdm=True, total=self._estN)
            else:
                return TqdmParallel(use_tqdm=True)
        else:
            return Parallel

    def __enter__(self):
        # check whether any data exists and load it if so.
        return self

    def __exit__(self, *exc):
        # close file pointer
        pass

    @property
    def ncpu(self):
        """create an ncpu property based on n_jobs """
        if self.n_jobs is None:
            return 1
        elif self.n_jobs == -1:
            # default it
            if self.is_fitted():
                # from joblib
                C = cpu_count()
                if self._estN < 0:
                    return C - 1
                else:
                    return self._estN if self._estN < C else (C - 1)
            else:
                return -1
        else:
            return self.n_jobs

    @property
    def filename(self):
        """Fetches the file directory for this context."""
        return self._fn if self._fn is not None else "temp.pkl"

    @property
    def cachef(self):
        """Retrieve the appropriate cache function"""
        if self.savemode == 'initial':
            return self._cache_initial
        elif self.savemode == 'override':
            return self._cache_override
        elif self.savemode == 'add':
            return self._cache_add

    def _load_file(self, fn=None):
        _file = self.filename if fn is None else fn
        if self.verbose > 0:
            print("loading from file '%s'" % _file)
        return load(_file, 'r')

    def _write_file(self, data, fn=None):
        _file = self.filename if fn is None else fn
        if self.verbose > 0:
            print("writing to file '%s'" % _file)
        dump(data, _file)

    def _estimate_n(self, *args):
        # try and produce an `_estN` parameter. -1 means unobtainable
        _len_args = list(map(lambda arg: hasattr(arg, "__len__"), args))
        # if there are any true, compress and take the first one
        if any(_len_args):
            self._estN = len(list(it.compress(args, _len_args))[0])

    def _generator_args(self, *args):
        return args if self._Nargs == 0 else zip(*args)

    def _wrap_tqdm(self, _gen_args):
        if is_tqdm_installed(False) and not (self.return_type == 'generator') and self.progress:
            from tqdm import tqdm
            return tqdm(_gen_args, position=0, total=self._estN)
        else:
            return _gen_args

    def _generate(self, result):
        """Receives a 'generator' result and determines whether to run or not."""
        if self.return_type == 'generator':
            return result
        else:
            if self.ncpu == 1:
                return list(result)
            else:
                ParallelObj = self._get_parallel_object()
                return ParallelObj(self.ncpu)(result)

    def _cache_initial(self, fn, f, *args):
        if os.path.isfile(self._fn):
            return self._load_file(fn)
        else:
            result = f(*args)
            self._write_file(result, fn)
            return result

    def _cache_override(self, fn, f, *args):
        result = f(*args)
        self._write_file(result, fn)
        return result

    def _cache_add(self, fn, f, *args):
        init = self._load_file(fn) if os.path.isfile(self._fn) else []
        # add on to estN the size of init
        result = init + f(*args)
        # write joining together
        self._write_file(result, fn)
        return result

    def _map_comp(self, f, *args):
        _argset = self._generator_args(*args)
        if self.verbose > 1:
            print("Generating chunk (n-args={}, n={})".format(self._Nargs, self._estN))
        if self.ncpu == 1:
            result = (f(*arg) for arg in self._wrap_tqdm(_argset))
        else:
            result = (delayed(f)(*arg) for arg in _argset)
        return self._generate(result)

    def _compute_chunks(self, f, *args):
        # check the valid file path exists.
        check_file_path(self._fn, False, True, 0)
        # make a directory
        relfile, _ = create_cache_directory(self._fn)
        # combine generator args with a count iterator to add to the string suffix in _cache_chunk
        _suffx = lambda i: add_suffix(i[0], relfile)
        # args of (fn, f, *args)
        _initial_args = (list(map(_suffx, zip(it.count(), self._generator_args(*args)))), it.repeat(f), *args)
        # comprehension over cache initials.
        return self._map_comp(self._cache_initial, *_initial_args)

    def _full_compute(self, f, *args):
        if self.is_filepath_set():
            # definitely caching, determine whether partial or not
            if self.has_chunks:
                result = self.cachef(self._fn, self._compute_chunks, f, *args)
                # delete temps
                _, cachedir = directory_info(self._fn)
                MapContext._delete_temps(cachedir)
                return result
            else:
                return self.cachef(self._fn, self._map_comp, f, *args)
        else:
            return self._map_comp(f, *args)

    def _compute_wrap_audio(self, f, *args):
        if self.end_audio:
            try:
                result = self._full_compute(f, *args)
                # play music if possible
                MapContext._play_success()
                return result
            except Exception:
                MapContext._play_failure()
        else:
            return self._full_compute(f, *args)

    """ !!!!!!!!!!!!!! ACTUAL FUNCTIONS BEGIN HERE !!!!!!!!!!!!!! """

    def is_fitted(self):
        """Whether 'compute' has been called. """
        return self._Nargs >= 0

    def is_filepath_set(self):
        """Determines whether a file(path) is set for this context."""
        return self._fn is not None

    def is_filepath_exists(self):
        """Determines whether the file already exists."""
        return self.is_filepath_set() and os.path.isfile(self._fn)

    def compute(self, f: Callable, *args, **kwargs):
        """Computes the pipeline.

        Parameters
        ----------
        f : function
            The function to call
        *args : list-like
            Arguments to pass as f(*args)
        **kwargs : dict
            Keyword-arguments passed to every iteration of f(*arg)

        Returns
        -------
        res : Any
            The results from f(*args) or from file
        """
        self._Nargs = len(args)
        # if we have no args, just call f using kws
        if self._Nargs == 0:
            return f(**kwargs)
        # compute the number of arguments, and potential list within each argument if multiple.
        # if we have an iterable, just panick.
        self._estimate_n(*args)
        # create a new function that wraps keywords into the call
        f_new = partial(f, **kwargs)
        # if we have end audio, play at end
        return self._compute_wrap_audio(f_new, *args)

    def compute_prod(self, f: Callable, *args, **kwargs):
        """Computes the pipeline, using the product of *args.

        Parameters
        ----------
        f : function
            The function to call
        *args : list-like
            Arguments to pass as f(it.product(*args), **kwargs)
        **kwargs : dict
            Keyword-arguments passed to every iteration of f(*arg)

        Returns
        -------
        res : Any
            The results from f(*args) or from file
        """
        self._Nargs = len(args)
        if self._Nargs == 0:
            return f(**kwargs)
        elif self._Nargs == 1:
            # run normal
            return self.compute(f, args[0], **kwargs)
        else:
            # create a new function that wraps keywords into the call
            f_new = partial(f, **kwargs)
            # estimate n from the length of each argument.
            _len_args = list(map(lambda arg: hasattr(arg, "__len__"), args))
            if any(_len_args):
                self._estN = reduce(operator.mul, map(len, it.compress(args, _len_args)))
            # product over variables, and retuple (map)
            new_args = (tuple(map(lambda t: t[i], it.product(*args))) for i in range(self._Nargs))
            # now run...?
            return self._compute_wrap_audio(f_new, *new_args)

    def clear(self):
        """Clears the cache."""
        if os.path.isfile(self._fn):
            os.remove(self._fn)
