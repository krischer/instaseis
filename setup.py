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
import os
from setuptools import find_packages, setup
from setuptools.extension import Extension
import sys


from setuptools import setup, find_packages

macros = []
extra_link_args = []
extra_compile_args = []

# Monkey patch CCompiler. We pretend that .f90 is a C extension and overwrite
# the corresponding compilation calls
CCompiler.language_map['.f90'] = "c"
# Monkey patch UnixCCompiler for Unix, Linux and darwin
UnixCCompiler.src_extensions.append(".f90")


def _compile(self, obj, src, ext, cc_args, extra_postargs, pp_opts):
    compiler_so = self.compiler_so
    #if sys.platform == 'darwin':
        #from distutils.unixccompiler import _darwin_compiler_fixup
        #compiler_so = _darwin_compiler_fixup(compiler_so, cc_args + \
                #extra_postargs)
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
# set library dir for mac and linux
libs = ['gfortran']

### Hack to prevent build_ext from trying to append "init" to the export
### symbols
class finallist(list):
    def append(self, object):
        return

class MyExtension(Extension):
    def __init__(self, *args, **kwargs):
        Extension.__init__(self, *args, **kwargs)
        self.export_symbols = finallist(self.export_symbols)

src = os.path.join('instaseis', 'src') + os.sep

lib = MyExtension('instaseis',
                  define_macros=macros,
                  libraries=libs,
                  extra_link_args=extra_link_args,
                  extra_compile_args=extra_compile_args,
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
    license="GNU General Public License, version 3 (GPLv3)",
    platforms="OS Independent",
    install_requires=["netCDF4", "numpy", "obspy"],
    ext_package='instaseis.lib',
    ext_modules=[lib],

)

if __name__ == "__main__":
    setup(**setup_config)
