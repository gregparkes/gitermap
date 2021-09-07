"""Testing code for mapping information."""
# ignore errors
import pytest
import itertools as it
import numpy as np
import os
import operator
from gitermap import umap, umapc


def f1(x):
    return x ** 2


def f2(x, y):
    return x + y


def f_none(x):
    return None


def test_umap1():
    """Test code for mapping list comprehensions.
    1. empty, 2. one elem, 3. wrong types
    """
    with pytest.raises(TypeError):
        assert umap(None, None)
        assert umap([], 2.)
        assert umap(lambda x: x, None)

    assert umap(f_none, [None])
    assert umap(f_none, [1, 3]) == [None, None]
    # try iterable
    assert umap(f1, it.islice(it.count(), 1, 10)) == list(map(np.square, range(1, 10)))
    # with keyword args
    assert umap(f2, [0, 2], y=3) == [3, 5]


@pytest.mark.parametrize("x", [[1, 3], (1, 3), [5], (2, 3, 7),
                               pytest.param([None], marks=pytest.mark.xfail)])
def test_umap_1d(x):
    assert umap(f1, x) == list(np.square(list(x)))


def test_umap_iterable():
    assert umap(f1, it.islice(it.count(), 0, 10)) == [n ** 2 for n in it.islice(it.count(), 0, 10)]


@pytest.mark.parametrize("x1,x2", [([1, 3], [2, 4]), ((1, 3), [2, 5])])
def test_umap_2d(x1, x2):
    assert umap(f2, x1, x2) == list(map(operator.add, x1, x2))


def test_umapc():
    with pytest.raises(TypeError):
        assert umapc(None, None, None)
        assert umapc(None, [], 2.)
        assert umapc(None, lambda x: x, None)

    assert umapc("temp.pkl", f1, [1, 3]) == [1, 9]
    # try with no .pkl
    assert umapc("temp2", f1, [1, 3]) == [1, 9]

    # now call again?
    assert umapc("temp.pkl", f1, [1, 3]) == [1, 9]
    assert umapc("temp2", f1, [1, 3]) == [1, 9]
    # check that file exists
    assert os.path.isfile("temp.pkl"), "temp file not created"
    assert os.path.isfile("temp2"), "temp2 file not created"

    # now delete temps
    os.remove("temp.pkl")
    os.remove("temp2")


def test_differing_map_lengths():
    """Handling the case where arguments aren't same length. Now raises an error."""
    with pytest.raises(ValueError):
        assert umap(f2, [5, 3], [2]) == [7, 3]
