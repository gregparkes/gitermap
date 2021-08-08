# `gitermap`: Easy parallelizable and cacheable list comprehensions

List comprehensions and `map()` operations are great in Python, but sometimes it would be nice if they just *did more*. gitermap allows users to work through a map operation with seemlessly integrated parallelization and automatic end-caching or step-by-step caching within your workflow. See below for a quick example:

```python
>>> from gitermap import umap
>>> umap(lambda x: x**2, [1, 3, 5])
[1, 9, 25]
```

This example works exactly as `map()` would do, except that with the `tqdm` package installed, a progressbar will also display, which is incredibly handy if each iteration of `f(x)` takes a longish time. But we take this even further; for long runs saving the result at the end is particularly handy to prevent the temptation of re-runs. We follow the convention of adding appropriate characters to the end of function names depending on need:

```python
>>> # umap with end-caching
>>> from gitermap import umapc
>>> umapc("temp.pkl", lambda x: x**2, [1, 3, 5])
[1, 9, 25]
```

Under the hood, `umapc` uses joblib to dump the data to "temp.pkl" which is in the local directory of wherever the python code is running. Assuming independence between samples, we can perform parallelism across the iterable list using `umapp`:

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

Note that at the end of `umapcc`, the temporary directory and files are deleted, leaving only "temp.pkl".

## Requirements

`gitermap` only technically requires a modern Python version (>=3.8) and `joblib` packages, which provides a lot of the grunt work for this project. We **highly** recommend also installing `tqdm` for beautiful progress bars on all list comprehension objects, but this is optional.

