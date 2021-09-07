#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 5 11:31:39 2019

@author: gparkes
"""

from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="gitermap",
    version="0.3.0",
    description="Easy parallelizable and cacheable list comprehensions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gregparkes/gitermap",
    author="Gregory Parkes",
    author_email="gregorymparkes@gmail.com",
    license="GPL-3.0",
    packages=find_packages(),
    zip_safe=False,
    python_requires='>=3.8',
    install_requires=[
        "joblib>=0.14.1"
    ],
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Framework :: Jupyter",
    ],
)