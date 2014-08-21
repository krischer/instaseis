#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Python interface to AxiSEM's netCDF based database mode.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
    Martin van Driel (vandriel@tomo.ig.erdw.ethz.ch), 2014
    Simon Stähler (staehler@geophysik.uni-muenchen.de), 2014
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
from setuptools import setup, find_packages


setup_config = dict(
    name="axisem_db",
    version="0.0.1",
    description="Python Interface to AxiSEM's DB mode",
    author=u"Lion Krischer, Martin van Driel, and Simon Stähler",
    author_email="krischer@geophysik.uni-muenchen.de",
    url="",
    packages=find_packages(),
    license="GNU General Public License, version 3 (GPLv3)",
    platforms="OS Independent",
    install_requires=["netCDF4", "numpy", "obspy"],
)

if __name__ == "__main__":
    setup(**setup_config)
