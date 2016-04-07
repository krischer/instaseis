#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import re
import warnings

from .version import get_git_version


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

    The special syntax ``syngine://MODEL_NAME`` will connect to the IRIS
    syngine web service for the specified model.

    >>> db = instaseis.open_db("syngine://ak135f")
    >>> print(db)
    SyngineInstaseisDB reciprocal Green's function Database (v7) ...
    Syngine model name:      'ak135f'
    Syngine service version:  0.0.2
        components           : vertical and horizontal
        velocity model       : ak135f

    .. note::

        If opening a local database and the ``ordered_output.nc4`` files are
        located for example in ``/path/to/DB/PZ/Data`` and
        ``/path/to/DB/PX/Data``, please pass ``/path/to/DB`` to the
        :func:`~instaseis.open_db` function. Instaseis will recursively
        search the child directories for the  necessary files and open them.
    """
    if path.startswith("syngine://"):
        model = re.sub("syngine://", "", path).strip()
        from instaseis.database_interfaces import syngine_instaseis_db
        return syngine_instaseis_db.SyngineInstaseisDB(model=model, *args,
                                                       **kwargs)
    elif "://" in path:
        from .database_interfaces import remote_instaseis_db
        return remote_instaseis_db.RemoteInstaseisDB(path, *args, **kwargs)
    else:
        from .database_interfaces import find_and_open_files
        return find_and_open_files(path=path, *args, **kwargs)


__version__ = get_git_version()


if __version__.startswith("0.0.0-tar/zipball"):  # pragma: no cover
    warnings.warn("Please don't install from a tarball. Use the proper pypi "
                  "release or install from git.", UserWarning)


from .source import Source, Receiver, ForceSource, FiniteSource  # NoQa
