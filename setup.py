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
from distutils.ccompiler import CCompiler
from distutils.errors import DistutilsExecError, CompileError
from distutils.unixccompiler import UnixCCompiler
from setuptools import find_packages, setup
from setuptools.extension import Extension

import inspect
import os
import sys


# Monkey patch the compilers to treat Fortran files like C files.
CCompiler.language_map['.f90'] = "c"
UnixCCompiler.src_extensions.append(".f90")


def get_package_data():
    """
    Returns a list of all files needed for the installation relativ to the
    "instaseis" subfolder.
    """
    filenames = []
    # The lasif root dir.
    root_dir = os.path.join(os.path.dirname(os.path.abspath(
        inspect.getfile(inspect.currentframe()))), "instaseis")
    # Recursively include all files in these folders:
    folders = [os.path.join(root_dir, "tests", "data")]
    for folder in folders:
        for directory, _, files in os.walk(folder):
            for filename in files:
                # Exclude hidden files.
                if filename.startswith("."):
                    continue
                filenames.append(os.path.relpath(
                    os.path.join(directory, filename),
                    root_dir))
    return filenames


def _compile(self, obj, src, ext, cc_args, extra_postargs, pp_opts):
    compiler_so = self.compiler_so
    if ext == ".f90":
        if sys.platform == 'darwin' or sys.platform == 'linux2':
            compiler_so = ["gfortran"]
            cc_args = ["-O", "-fPIC", "-c", "-ffree-form"]
    try:
        self.spawn(compiler_so + cc_args + [src, '-o', obj] +
                   extra_postargs)
    except DistutilsExecError, msg:
        raise CompileError(msg)
UnixCCompiler._compile = _compile


# Hack to prevent build_ext from trying to append "init" to the export symbols.
class finallist(list):
    def append(self, object):
        return


class MyExtension(Extension):
    def __init__(self, *args, **kwargs):
        Extension.__init__(self, *args, **kwargs)
        self.export_symbols = finallist(self.export_symbols)

src = os.path.join('instaseis', 'src')
lib = MyExtension('instaseis',
                  libraries=["gfortran"],
                  # Be careful with the order.
                  sources=[
                      os.path.join(src, "global_parameters.f90"),
                      os.path.join(src, "finite_elem_mapping.f90"),
                      os.path.join(src, "spectral_basis.f90"),
                      os.path.join(src, "sem_derivatives.f90"),
                      os.path.join(src, "lanczos.f90"),
                  ])

setup_config = dict(
    name="instaseis",
    version="0.0.1",
    description="Python Interface to AxiSEM's DB mode",
    author=u"Lion Krischer, Martin van Driel, and Simon Stähler",
    author_email="krischer@geophysik.uni-muenchen.de",
    url="",
    packages=find_packages(),
    package_data={
        "instaseis": [os.path.join("lib", "instaseis.so")] +
            get_package_data()},
    license="GNU General Public License, version 3 (GPLv3)",
    platforms="OS Independent",
    install_requires=["netCDF4", "numpy", "obspy"],
    extras_require={
        'tests': ['flake8>=2', 'pytest']},
    ext_package='instaseis.lib',
    ext_modules=[lib],
)

if __name__ == "__main__":
    setup(**setup_config)

    # Attempt to remove the mod files once again.
    for filename in ["finite_elem_mapping.mod", "global_parameters.mod",
                     "lanczos.mod", "sem_derivatives.mod",
                     "spectral_basis.mod"]:
        try:
            os.remove(filename)
        except:
            pass
