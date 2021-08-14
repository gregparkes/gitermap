# `gitermap`: Easy parallelizable and cacheable list comprehensions

List comprehensions and `map()` operations are great in Python, but sometimes it would be nice if they just *did more*. gitermap allows users to work through a map operation with seemlessly integrated parallelization and automatic end-caching or step-by-step caching within your workflow. Key functionalities include:

- Easy parallelization built on top of `joblib`
- Automatic caching of results at the end of iteration (end-caching)
- Automatic caching at each step of iteration (chunk-caching)
- Additive caching for randomized repeats
- Seemless integration with `itertools`, allowing for generators and functional computing

`gitermap` provides simple access with useful accessory functions, and detailed access through
exposure to underlying classes via `MapContext` which neatly wrap the complexity for you.

## Examples

See below for an example that is identical to `map()` function, except it returns a list instead of an iterable:

```python
>>> from gitermap import umap
>>> umap(lambda x: x**2, [1, 3, 5])
[1, 9, 25]
```

This example works exactly as `map()` would do, except that with the `tqdm` package installed, a progressbar will also display, which is incredibly handy if each iteration of `f(x)` takes a longish time. But we take this even further; for long runs saving the result at the end is particularly handy to prevent the temptation of re-runs. We follow the convention of adding appropriate characters to the end of function names depending on need:

```python
>>> # umap with end-caching
>>> from gitermap import umapc
>>> umapc("temp.pkl", lambda x, y: x**2 + y**2, [1, 3, 5], [2, 4, 6])
[5, 25, 61]
```

Under the hood, `umapc` uses joblib to dump the data to "temp.pkl" which is in the local directory of wherever the python code is running. The only requirement is that
the function f(x) must return something picklable for joblib to use. Assuming independence between samples, we can perform parallelism across the iterable list using `umapp`:

```python
>>> # umap with parallelism
>>> from gitermap import umapp
>>> # creates three threads
>>> umapp(lambda x: x**2, [1, 3, 5])
[1, 9, 25]
```

For particularly long runs, it may be necessary to store the result at each iteration rather than just at the end. This means that if a sneaky bug appears in one of your iterations, all of the computed data can be read in up to the point of the bug, meaning your compute pipeline doesn't need to be fully re-computed:

```python
>>> # umap with caching by chunks
>>> from gitermap import umapcc
>>> # no threading, saving each iteration in a subfile
>>> umapcc("temp.pkl", lambda x: x**2, range(50))
[1, 9, ...]
```

Note that at the end of `umapcc`, the temporary directory and files are deleted, leaving only "temp.pkl". See the below table summary of each function and what functionality it supports:

| Function Name | Parallelization | End-Caching | Chunk-Caching |
| --------- | -------------- | --------- | ----------- |
| `umap` | &#x2612; | &#x2612; | &#x2612; |
| `umapp` | &#x2611; | &#x2612; | &#x2612; |
| `umapc` | &#x2612; | &#x2611; | &#x2612; |
| `umappc` | &#x2611; | &#x2611; | &#x2612; |
| `umapcc` | &#x2612; | &#x2611; | &#x2611; |
| `umappcc` | &#x2611; | &#x2611; | &#x2611; |

For more control over how many threads are used for parallelization, end sounds, keyword arguments in `f(...)` and so on, we expose an object called `MapContext`, passing in the `n_jobs` argument as you would for `joblib` or in scikit-learn (default=`None`, or 1):

```python
>>> import itertools as it
>>> from gitermap import MapContext
>>> with MapContext(n_jobs=-1) as ctx:
>>>     result = ctx.compute(lambda x: x**2, it.islice(it.count(), 0, 100))
>>> result
[0, 1, 4, 9, ...]
```

Note that this is equivalent to `umap`, albeit in longer form - under the hood `umap` simply creates
a `MapContext` object and calls compute like so. Note that if you wish to have lazy evaluation by deferring execution,
we provide this through the `return_type` parameter:

```python
>>> from gitermap import MapContext
>>> with MapContext(return_type="generator") as ctx:
>>>     result = ctx.compute(lambda x: x**2, range(20))
>>> result
<generator object MapContext._map_comp.<locals>.<genxpr> at 0x00000000000>
```

Note that generators do not also perform parallelization when evoked. 
For further details, see the example notebooks within this project.

## Requirements

The following requirements are essential for `gitermap`:

- Python >= 3.8
- `joblib` >= 1.4

The following packages are highly recommended but not essential:

- `tqdm` >= 1.0: For progressbars

For ending sound in `MapContext`, you will also need:

- `numpy`: For processing sound waves
- `simpleaudio`: For playing sounds

The other packages such as `itertools`, `functools` and more are default and come with the standard Python distribution.
