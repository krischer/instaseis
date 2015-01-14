#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)


class InstaseisError(Exception):
    pass


class InstaseisWarning(UserWarning):
    pass


class InstaseisNotFoundError(InstaseisError):
    pass


class ReceiverParseError(InstaseisError):
    pass


class SourceParseError(InstaseisError):
    pass


def open_db(path, *args, **kwargs):
    """
    Central function to open a local or remote Instaseis database. Any
    keyword arguments are passed to the underlying
    :class:`~instaseis.instaseis_db.InstaseisDB` or
    :class:`~instaseis.remote_instaseis_db.RemoteInstaseisDB` classes.

    :type path: str
    :param path: Filepath or URL. Instaseis will determine if it is a local
        file path or a HTTP URL and delegate to the corresponding class.
    :returns: An initialized database object.
    :rtype: :class:`~instaseis.instaseis_db.InstaseisDB` or
        :class:`~instaseis.remote_instaseis_db.RemoteInstaseisDB`

    If a directory is passed, it will return a local Instaseis database:

    >>> import instaseis
    >>> db = instaseis.open_db("/path/to/DB")
    >>> print(db)
    InstaseisDB reciprocal Green's function Database (v7) generated ...
        components           : vertical and horizontal
        velocity model       : ak135f
        ...

    For an HTTP URL it will return a remote Instaseis database:

    >>> db = instaseis.open_db("http://webadress.com:8765")
    >>> print(db)
    RemoteInstaseisDB reciprocal Green's function Database (v7) generated ...
        components           : vertical and horizontal
        velocity model       : ak135f

    .. note::

        If opening a local database and the ``ordered_output.nc4`` files are
        located for example in ``/path/to/DB/PZ/Data`` and
        ``/path/to/DB/PX/Data``, please pass ``/path/to/DB`` to the
        :func:`~instaseis.open_db` function. Instaseis will recursively
        search the child directories for the  necessary files and open them.
    """
    if "://" in path:
        from . import remote_instaseis_db
        return remote_instaseis_db.RemoteInstaseisDB(path, *args, **kwargs)
    else:
        from . import instaseis_db
        return instaseis_db.InstaseisDB(path, *args, **kwargs)

import netCDF4
import re
import warnings

from .version import get_git_version
__version__ = get_git_version()

if __version__.startswith("0.0.0-tar/zipball"):
    warnings.warn("Please don't install from a tarball. Use the proper pypi "
                  "release or install from git.", UserWarning)


def __version_cmp(required_version, version):
    """
    based on http://stackoverflow.com/a/1714190/1657047
    """
    def normalize(v):
        return [int(x) for x in re.sub(r'(\.0+)*$', '', v).split(".")]

    if normalize(version) < normalize(required_version):
        return False
    return True


# There are some serious errors with older netCDF version when doing lots of
# I/O. Thus raise a warning and delegate responsibilities to the users.
if not __version_cmp("4.3", netCDF4.__netcdf4libversion__):
    msg = ("There are some issues when using a netCDF version smaller "
           "than 4.3 (you are running version %s). Please update your netCDF "
           "installation." % netCDF4.__netcdf4libversion__)
    warnings.warn(msg, InstaseisWarning)


from .source import Source, Receiver, ForceSource, FiniteSource  # NoQa
