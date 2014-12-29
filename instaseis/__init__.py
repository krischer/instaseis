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
        raise NotImplementedError
    else:
        from .instaseis_db import InstaseisDB
        return InstaseisDB(path, *args, **kwargs)


from .source import Source, Receiver, ForceSource, FiniteSource  # NoQa
