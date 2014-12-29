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


def open(path, *args, **kwargs):
    if "://" in path:
        pass
    else:
        pass


from .instaseisdb import InstaseisDB  # NoQa
from .source import Source, Receiver, ForceSource, FiniteSource  # NoQa
