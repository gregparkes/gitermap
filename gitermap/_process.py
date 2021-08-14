"""Handles the main processes of umaps."""
from typing import Callable
from functools import partial

# local functions
from ._context import MapContext

__all__ = ['ugen', 'umap', 'umapc', 'umapp', 'umappc', 'umapcc', 'umappcc']

""" ############# ACTUAL FUNCTIONS BEGIN HERE ########################### """


def ugen(f: Callable, *args, **kwargs):
    """Performs MAP list-comprehension.

    This is identicial to umap, except it returns an iterable for further use.

    Parameters
    ----------
    f : function
        The function to call
    *args : list-like
        Arguments to pass as f(*args), as a, ..., k
    **kwargs : dict
        Keyword arguments that are passed to every call of f(*args)

    Returns
    -------
    res : iterable
        The results from f(*args)
    """
    f_new = partial(f, **kwargs)
    return map(f_new, *args)


def umap(f: Callable, *args, **kwargs):
    """Performs MAP list-comprehension.

    Given function f(x) and arguments a, ..., k;
        map f(a_i, ..., k_i), ..., f(a_z, ..., k_z).

    Nearly equivalent to list(map(f, a, ..., k))

    ..note: tqdm is optional but highly recommended.

    Parameters
    ----------
    f : function
        The function to call
    *args : list-like
        Arguments to pass as f(*args), as a, ..., k
    **kwargs : dict
        Keyword arguments that are passed to every call of f(*args)

    Returns
    -------
    res : list
        The results from f(*args)

    Examples
    --------
    Provides a clean way to do a list comprehension:
    >>> from gitermap import umap
    >>> umap(lambda x: x**2, [2, 4, 6])
    >>> [4, 16, 36]
    Like the normal mapping, multiple lists map to
    multiple parameters passed to the function:
    >>> umap(lambda x, y: x + y, [2, 4, 6], [1, 2, 4])
    >>> [3, 6, 10]
    For functions that require additional keyword arguments 
    that are fixed across all iterations,
    `partial` can be used to the function as a precursor:
    >>> from functools import partial
    >>> def my_func(x, y, special):
    >>>		return (x + y) * special
    >>> updated_func = partial(my_func, special=1.5)
    >>> umap(updated_func, [1, 3, 5], [2, 4, 6])
    """
    return MapContext().compute(f, *args, **kwargs)


def umapc(fn: str, f: Callable, *args, **kwargs):
    """Performs MAP comprehension with End-Cache.

    That is to say that the first time this runs,
        function f(*args) is called, storing a cache file at address `fn`.
        The second time and onwards, the resulting cached
        file is read and no execution takes place.

    .. note: The saving only happens at the end of execution, so if the program
        stops, nothing is saved.

    Parameters
    ----------
    fn : str
        The path and filename.
    f : function
        The function to call
    *args : list-like
        Arguments to pass as f(*args)
    **kwargs : dict
        Keyword arguments that are passed to every call of f(*args)

    Returns
    -------
    res : Any
        The results from f(*args) or from file

    Examples
    --------
    See `gitermap.umap` for examples.
    """
    return MapContext(fn).compute(f, *args, **kwargs)


def umapp(f: Callable, *args, **kwargs):
    """Performs MAP comprehension with Parallelism.

    This assumes each iteration is independent
        from each other in the list comprehension.

    Parameters
    ----------
    f : function
        The function to call
    *args : list-like
        Arguments to pass as f(*args)
    **kwargs : dict
        Keyword arguments that are passed to every call of f(*args)

    Returns
    -------
    res : Any
        The results from f(*args) or from file

    Examples
    --------
    See `gitermap.umap` for examples.
    """
    return MapContext(n_jobs=-1).compute(f, *args, **kwargs)


def umappc(fn: str, f: Callable, *args, **kwargs):
    """Performs MAP comprehension with Parallelism and End-Caching.

    That is to say that the first time this runs, function f(*args) is called,
        storing a cache file. The second time and onwards, the resulting
        cached file is read and no execution on `f` takes place.

    This assumes each iteration is independent
        from each other in the list comprehension for parallelism.

    Parameters
    ----------
    fn : str
        The path and filename.
    f : function
        The function to call
    *args : list-like
        Arguments to pass as f(*args)
    **kwargs : dict
        Keyword arguments that are passed to every call of f(*args)

    Returns
    -------
    res : Any
        The results from f(*args) or from file

    Examples
    --------
    See `gitermap.umapp` for examples.
    """
    # add f to args
    return MapContext(fn, n_jobs=-1).compute(f, *args, **kwargs)


def umapcc(fn: str, f: Callable, *args, **kwargs):
    """Performs MAP comprehension with Caching by Chunks.

    That is to say that the first time this runs, function f(*args) is called,
        storing a cache file. The second time and onwards, the resulting
        cached file is read and no execution takes place.

    Further to this, 'by-chunks' means that each step is stored separately as a file
    and concatenated together at the end. The intermediate caches are removed
    at the end of the process automatically. If the program crashes part-way through
    this, re-running will resume from the last stored chunk.

    Parameters
    ----------
    fn : str
        The path and filename.
    f : function
        The function to call
    *args : list-like
        Arguments to pass as f(*args)
    **kwargs : dict
        Keyword arguments that are passed to every call of f(*args)

    Returns
    -------
    res : Any
        The results from f(*args) or from file

    Examples
    --------
    See `gitermap.umap` for examples.
    """
    return MapContext(fn, chunks=True).compute(f, *args, **kwargs)


def umappcc(fn: str, f: Callable, *args, **kwargs):
    """Performs MAP comprehension with Parallelism and Caching by Chunks.

    That is to say that the first time this runs, function f(*args) is called,
        storing a cache file. The second time and onwards, the resulting
        cached file is read and no execution takes place.

    Further to this, 'by-chunks' means that each step is stored separately as a file
        and concatenated together at the end. This means that if a program stops half way through
        execution, when re-run, it restarts from the last cached element, which is incredibly
        useful during debugging and prototype development.

    This assumes each iteration is independent from each other in the list comprehension.

    Parameters
    ----------
    fn : str
        The path and filename.
    f : function
        The function to call
    *args : list-like
        Arguments to pass as f(*args)
    **kwargs : dict
        Keyword arguments that are passed to every call of f(*args)

    Returns
    -------
    res : Any
        The results from f(*args) or from file

    Examples
    --------
    See `gitermap.umapcc` for examples.
    """
    return MapContext(fn, n_jobs=-1, chunks=True).compute(f, *args, **kwargs)
