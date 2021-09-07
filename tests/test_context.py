"""Testing code for context object."""
import os
import pytest

from gitermap import MapContext


def f1(x):
    return x ** 2


def f2(x, y):
    return x + y


def f_none(x):
    return None


def test_map_context1():
    ctx = MapContext()


def test_map_context2():
    ctx = MapContext()
    assert ctx.compute(f1, [1, 3]) == [1, 9]
    assert ctx.compute(f1, [5]) == [25]


@pytest.mark.parametrize("fn", ["temp.pkl", "temp2", "temp3.json"])
def test_map_context_filename(fn):
    """Mapcontext with cache file"""
    ctx = MapContext(fn)
    assert ctx.compute(f1, [1, 3]) == [1, 9]
    # call again
    assert ctx.compute(f1, [1, 3]) == [1, 9]
    # check file
    assert os.path.isfile(fn), "temp file not present"

    os.remove(fn)


@pytest.mark.parametrize("njobs", [1, 3, -1, 20])
def test_map_context_njobs(njobs):
    """Mapcontext with cache file"""
    ctx = MapContext(n_jobs=njobs)
    assert ctx.compute(f1, [1, 3]) == [1, 9]
    assert ctx.compute(f1, [5]) == [25]
