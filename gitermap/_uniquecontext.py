import warnings
from functools import reduce
import operator

from ._context import MapContext


class UniqueMapContext(MapContext):
    """MapContext with unique labels for each iterable in a run."""

    def __init__(self,
                 filename: str = None,
                 filter_none: bool = False,
                 verbose: int = 0,
                 n_jobs: int = None,
                 chunks: bool = False,
                 return_type: str = "list",
                 progressbar: bool = True,
                 savemode: str = "initial",
                 end_audio: bool = False):
        """Creates a context to wrap list comprehensions in, using unique keys.

        Parameters
        ----------
        filename: str, optional
            A directory to cache files in.
        filter_none: bool, optional
            If True, removes None variables where duplicates exist
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
        super().__init__(filename, verbose, n_jobs, chunks,
                         return_type, progressbar, savemode, end_audio)
        # now prep hash cache
        self.filter_none = filter_none
        self._hash_cache = set()

    def _pre_f(self, f, *arg):
        # this function is called for each arg set.
        H = hash(reduce(operator.add, (map(str, arg))))
        # call f
        result = f(*arg) if H not in self._hash_cache else None
        # now add to hash_cache
        self._hash_cache.add(H)
        return result

    def _map_comp(self, f, *args):
        """Overriding from MapContext to enumerate over args"""
        _argset = zip(*args)
        if self.verbose > 1:
            print("Generating chunk (n-args={}, n={})".format(self._Nargs, self._estN))
        if self.ncpu == 1:
            result = (self._pre_f(f, *arg) for arg in self._wrap_tqdm(_argset))
        else:
            result = (delayed(self._pre_f)(f, *arg) for arg in _argset)
        # we also filter out None variables only returning useful values.
        if self.filter_none:
            return self._generate(filter(None.__ne__, result))
        else:
            return self._generate(result)
