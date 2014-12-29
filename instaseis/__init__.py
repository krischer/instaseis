#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)


class InstaseisError(Exception):
    pass


class InstaseisNotFoundError(InstaseisError):
    pass


class ReceiverParseError(InstaseisError):
    pass


class SourceParseError(InstaseisError):
    pass


def open_db(path, *args, **kwargs):
    """
    Open a local or remote Instaseis database.

    :param path: Filepath or URL.
    """
    if "://" in path:
        from . import remote_instaseis_db
        return remote_instaseis_db.RemoteInstaseisDB(path, *args, **kwargs)
    else:
        from . import instaseis_db
        return instaseis_db.InstaseisDB(path, *args, **kwargs)


from .source import Source, Receiver, ForceSource, FiniteSource  # NoQa
