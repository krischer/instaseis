#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Python interface to AxiSEM's netCDF based database mode.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
    Martin van Driel (vandriel@tomo.ig.erdw.ethz.ch), 2014
    Simon Stähler (staehler@geophysik.uni-muenchen.de), 2014
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lgpl.html)
"""
from distutils.ccompiler import CCompiler
from distutils.errors import DistutilsExecError, CompileError
from distutils.unixccompiler import UnixCCompiler
from setuptools import find_packages, setup
from setuptools.extension import Extension

import inspect
import os
from subprocess import Popen, PIPE
import sys


# Import the version string.
path = os.path.join(os.path.abspath(os.path.dirname(inspect.getfile(
    inspect.currentframe()))), "instaseis")
sys.path.insert(0, path)
from version import get_git_version


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
    except DistutilsExecError as msg:
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


def get_libgfortran_dir():
    """
    Helper function returning the library directory of libgfortran. Useful
    on OSX where the C compiler oftentimes has no knowledge of the library
    directories of the Fortran compiler. I don't think it can do any harm on
    Linux.
    """
    for ending in [".3.dylib", ".dylib", ".3.so", ".so"]:
        try:
            p = Popen(['gfortran', "-print-file-name=libgfortran" + ending],
                      stdout=PIPE, stderr=PIPE)
            p.stderr.close()
            line = p.stdout.readline().decode().strip()
            p.stdout.close()
            if os.path.exists(line):
                return [os.path.dirname(line)]
        except:
            continue
        return []


src = os.path.join('instaseis', 'src')
lib = MyExtension('instaseis',
                  libraries=["gfortran"],
                  library_dirs=get_libgfortran_dir(),
                  # Be careful with the order.
                  sources=[
                      os.path.join(src, "global_parameters.f90"),
                      os.path.join(src, "finite_elem_mapping.f90"),
                      os.path.join(src, "spectral_basis.f90"),
                      os.path.join(src, "sem_derivatives.f90"),
                      os.path.join(src, "lanczos.f90"),
                  ])

INSTALL_REQUIRES = ["netCDF4 >= 1.1", "numpy", "obspy", "future", "requests",
                    "tornado", "flake8>=2", "pytest", "responses"]

# Add argparse and ordereddict for Python 2.6. Both are standard library
# packages for Python >= 2.7.
if sys.version_info[:2] == (2, 6):
    INSTALL_REQUIRES.extend(["argparse", "ordereddict"])
# Add mock for Python 2.x. Starting with Python 3 it is part of the standard
# library.
if sys.version_info[0] == 2:
    INSTALL_REQUIRES.append("mock")

setup_config = dict(
    name="instaseis",
    version=get_git_version(),
    description="Instant seismograms from an AxiSEM Green's functions' DB.",
    author=u"Lion Krischer, Martin van Driel, and Simon Stähler",
    author_email="krischer@geophysik.uni-muenchen.de",
    url="",
    packages=find_packages(),
    package_data={
        "instaseis":
            [os.path.join("lib", "instaseis.so")] +
            [os.path.join("gui", "qt_window.ui")] +
            get_package_data()},
    license="GNU Lesser General Public License, version 3 (LGPLv3)",
    platforms="OS Independent",
    install_requires=INSTALL_REQUIRES,
    ext_package='instaseis.lib',
    ext_modules=[lib],
    classifiers=[
        # complete classifier list:
        # http://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Operating System :: Unix",
        "Operating System :: POSIX",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2.6",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: Implementation :: CPython",
        ],
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
