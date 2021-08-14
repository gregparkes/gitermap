"""init method for package."""

from ._process import *
from ._context import MapContext

__version__ = '0.1.0'
__name__ = "gitermap"
__doc__ = """gitermap: Easy parallelizable and cacheable list comprehensions

List comprehensions and `map()` operations are great in Python, 
but sometimes it would be nice if they just *did more*. gitermap allows users to work through 
a map operation with seemlessly integrated parallelization and automatic end-caching or 
step-by-step caching within your workflow.
"""
