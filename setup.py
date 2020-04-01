#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Instaseis: Instant Global Broadband Seismograms Based on a Waveform Database

Instaseis calculates broadband seismograms from Green’s function databases
generated with AxiSEM and allows for near instantaneous (on the order of
milliseconds) extraction of seismograms. Using the 2.5D axisymmetric
spectral element method, the generation of these databases, based on
reciprocity of the Green’s functions, is very efficient and is approximately
half as expensive as a single AxiSEM forward run. Thus this enables the
computation of full databases at half the cost of the computation of
seismograms for a single source in the previous scheme and hence allows to
compute databases at the highest frequencies globally observed. By storing
the basis coefficients of the numerical scheme (Lagrange polynomials),
the Green’s functions are 4th order accurate in space and the spatial
discretization respects discontinuities in the velocity model exactly. On
top, AxiSEM allows to include 2D structure in the source receiver plane and
readily includes other planets such as Mars.

For more information visit http://www.instaseis.net.

:copyright:
    The Instaseis Development Team (instaseis@googlegroups.com), 2014-2020
:license:
    GNU Lesser General Public License, Version 3 [non-commercial/academic use]
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
path = os.path.join(
    os.path.abspath(os.path.dirname(inspect.getfile(inspect.currentframe()))),
    "instaseis",
)
sys.path.insert(0, path)
from version import get_git_version  # noqa


# Monkey patch the compilers to treat Fortran files like C files.
CCompiler.language_map[".f90"] = "c"
UnixCCompiler.src_extensions.append(".f90")

DOCSTRING = __doc__.strip().split("\n")


def get_package_data():
    """
    Returns a list of all files needed for the installation relativ to the
    "instaseis" subfolder.
    """
    filenames = []
    # The lasif root dir.
    root_dir = os.path.join(
        os.path.dirname(
            os.path.abspath(inspect.getfile(inspect.currentframe()))
        ),
        "instaseis",
    )
    # Recursively include all files in these folders:
    folders = [
        os.path.join(root_dir, "tests", "data"),
        os.path.join(root_dir, "gui", "data"),
        os.path.join(root_dir, "server", "data"),
    ]
    for folder in folders:
        for directory, _, files in os.walk(folder):
            for filename in files:
                # Exclude hidden files.
                if filename.startswith("."):
                    continue
                filenames.append(
                    os.path.relpath(
                        os.path.join(directory, filename), root_dir
                    )
                )
    filenames.append("RELEASE-VERSION")
    return filenames


def _compile(self, obj, src, ext, cc_args, extra_postargs, pp_opts):
    compiler_so = self.compiler_so
    if ext == ".f90":
        if sys.platform == "darwin" or sys.platform == "linux2":
            compiler_so = ["gfortran"]
            cc_args = ["-O", "-fPIC", "-c", "-ffree-form"]
    try:
        self.spawn(compiler_so + cc_args + [src, "-o", obj] + extra_postargs)
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
            p = Popen(
                ["gfortran", "-print-file-name=libgfortran" + ending],
                stdout=PIPE,
                stderr=PIPE,
            )
            p.stderr.close()
            line = p.stdout.readline().decode().strip()
            p.stdout.close()
            if os.path.exists(line):
                return [os.path.dirname(line)]
        except:
            continue
        return []


src = os.path.join("instaseis", "src")
lib = MyExtension(
    "instaseis",
    libraries=["gfortran"],
    library_dirs=get_libgfortran_dir(),
    # Be careful with the order.
    sources=[
        os.path.join(src, "global_parameters.f90"),
        os.path.join(src, "finite_elem_mapping.f90"),
        os.path.join(src, "spectral_basis.f90"),
        os.path.join(src, "sem_derivatives.f90"),
    ],
)

INSTALL_REQUIRES = [
    "h5py",
    "numpy",
    "obspy >= 1.2.1",
    "tornado>=6.0.0",
    "requests",
    "geographiclib",
    "jsonschema >= 2.4.0",
]

EXTRAS_REQUIRE = {
    "tests": [
        "click",
        "netCDF4",
        "pytest-xdist",
        "flake8>=3",
        "pytest>=5.0",
        "responses",
    ]
}

# Add mock for Python 2.x. Starting with Python 3 it is part of the standard
# library.
if sys.version_info[0] == 2:
    INSTALL_REQUIRES.append("mock")

setup_config = dict(
    name="instaseis",
    version=get_git_version(),
    description=DOCSTRING[0],
    long_description="\n".join(DOCSTRING[2:]),
    author="Lion Krischer, Martin van Driel, and Simon Stähler",
    author_email="lion.krischer@gmail.com",
    url="http://instaseis.net",
    packages=find_packages(),
    package_data={
        "instaseis": [os.path.join("lib", "instaseis.so")]
        + [os.path.join("gui", "qt_window.ui")]
        + get_package_data()
    },
    license="GNU Lesser General Public License, version 3 (LGPLv3) for "
    "non-commercial/academic use",
    platforms="OS Independent",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    ext_package="instaseis.lib",
    ext_modules=[lib],
    # this is needed for "pip install instaseis==dev"
    download_url=(
        "https://github.com/krischer/instaseis/zipball/master"
        "#egg=instaseis=dev"
    ),
    python_requires=">=3.6",
    classifiers=[
        # complete classifier list:
        # http://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Operating System :: Unix",
        "Operating System :: POSIX",
        "Operating System :: MacOS",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)

if __name__ == "__main__":
    setup(**setup_config)

    # Attempt to remove the mod files once again.
    for filename in [
        "finite_elem_mapping.mod",
        "global_parameters.mod",
        "sem_derivatives.mod",
        "spectral_basis.mod",
    ]:
        try:
            os.remove(filename)
        except:
            pass
