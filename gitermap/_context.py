"""Handles the MapContext object."""

import warnings
import os
from functools import partial
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
        self.end_audio = end_audio if is_simpleaudio_installed() else False
        self._Nargs = -1
        self._estN = -1

    @classmethod
    def _get_parallel_object(cls, n):
        # determine parallel object
        if is_tqdm_installed(False):
            # load custom parallel tqdm object if we use it, else joblib normal.
            return TqdmParallel(use_tqdm=True, total=n)
        else:
            return Parallel

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

    def _generator_args(self, *args):
        return args if self._Nargs == 0 else it.zip_longest(*args)

    def _wrap_tqdm(self, _gen_args):
        if is_tqdm_installed(False) and not (self.return_type == 'generator'):
            from tqdm import tqdm
            return tqdm(_gen_args, position=0, total=self._estN)
        else:
            return _gen_args

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
        _gen = self._wrap_tqdm(self._generator_args(*args))
        if self.verbose > 1:
            print("Running chunk (n-args={}, n={})".format(self._Nargs, self._estN))
        if self.return_type == 'list':
            return [f(*arg) for arg in _gen]
        elif self.return_type == 'generator':
            return (f(*arg) for arg in _gen)

    def _parallel_map_comp(self, f, *args):
        ParallelObj = MapContext._get_parallel_object(self._estN)

        if self.verbose > 1:
            print("Running chunk (n-args={}, n={})".format(self._Nargs, self._estN))

        if self._Nargs == 1:
            itr_result = ParallelObj(self.ncpu)(delayed(f)(arg) for arg in args[0])
        else:
            itr_result = ParallelObj(self.ncpu)(delayed(f)(*arg) for arg in it.zip_longest(*args))
        return itr_result

    def _compute_chunks(self, f, *args):
        # check the valid file path exists.
        check_file_path(self._fn, False, True, 0)
        # make a directory
        relfile, _ = create_cache_directory(self._fn)
        # allocate chunk names.
        fn_chunks = map(partial(add_suffix, filename=relfile), range(self._estN))
        # now combine with generator args
        _gen = self._wrap_tqdm(zip(fn_chunks, self._generator_args(*args)))
        # if we are dealing with parallel, then do so
        if self.ncpu == 1:
            its_result = [self._cache_initial(name, f, *arg) for name, arg in _gen]
        else:
            ParallelObj = MapContext._get_parallel_object(self._estN)
            its_result = ParallelObj(self.ncpu)(
                delayed(self._cache_initial)(name, f, *arg) for name, arg in _gen
            )
        # return
        return its_result

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
                # normal caching, maybe with parallelism
                if self.ncpu == 1:
                    return self.cachef(self._fn, self._map_comp, f, *args)
                else:
                    return self.cachef(self._fn, self._parallel_map_comp, f, *args)
        else:
            if self.ncpu == 1:
                return self._map_comp(f, *args)
            else:
                return self._parallel_map_comp(f, *args)

    def is_fitted(self):
        """Whether 'compute' has been called. """
        return self._Nargs >= 0

    def is_filepath_set(self):
        """Determines whether a file(path) is set for this context."""
        return self._fn is not None

    def is_filepath_exists(self):
        """Determines whether the file already exists."""
        return self.is_filepath_set() and os.path.isfile(self._fn)

    def compute(self, f: Callable, *args):
        """Computes the pipeline.

        Parameters
        ----------
        f : function
            The function to call
        *args : list-like
            Arguments to pass as f(*args)

        Returns
        -------
        res : Any
            The results from f(*args) or from file
        """
        self._Nargs = len(args)

        if self._Nargs == 0:
            return f()
        # compute the number of arguments, and potential list within each argument if multiple.
        self._estN = len(args[0])
        # wrap in music?
        try:
            result = self._full_compute(f, *args)
            # play music if possible
            if self.end_audio:
                MapContext._play_success()
        except Exception as e:
            if self.end_audio:
                MapContext._play_failure()
        return result

    def compute_kw(self, f: Callable, *args, **kwargs):
        """Compute the pipeline with keyword arguments."""
        new_f = partial(f, **kwargs)
        return self.compute(new_f, *args)

    def clear(self):
        """Clears the cache."""
        if os.path.isfile(self._fn):
            os.remove(self._fn)
