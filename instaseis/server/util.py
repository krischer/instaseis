#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2015
:license:
    GNU Lesser General Public License, Version 3 [non-commercial/academic use]
    (http://www.gnu.org/copyleft/lgpl.html)
"""
import re
import functools
import threading
import obspy

# Valid phase offset pattern including capture groups.
PHASE_OFFSET_PATTERN = re.compile(r"(^[A-Za-z0-9^]+)([\+-])([\deE\.\-\+]+$)")


def run_async(func):
    """
    Decorator executing a function in a thread.

    Adapted from http://stackoverflow.com/a/15952516/1657047
    """
    @functools.wraps(func)
    def async_func(*args, **kwargs):
        func_hl = threading.Thread(target=func, args=args, kwargs=kwargs)
        func_hl.start()
        return func_hl
    return async_func


class IOQueue(object):
    """
    Object passed to the zipfile constructor which acts as a file-like object.

    Iterating over the object yields the data pieces written to it since it
    has last been iterated over DELETING those pieces at the end of each
    loop. This enables the server to send unbounded zipfiles without running
    into memory issues.
    """
    def __init__(self):
        self.count = 0
        self.data = []

    def flush(self):
        pass

    def tell(self):
        return self.count

    def write(self, data):
        self.data.append(data)
        self.count += len(data)

    def __iter__(self):
        for _i in self.data:
            yield _i
        self.data = []
        raise StopIteration


def _validtimesetting(value):
    try:
        return obspy.UTCDateTime(value)
    except:
        pass

    try:
        return float(value)
    except:
        pass

    m = PHASE_OFFSET_PATTERN.match(value)
    if m is None:
        raise ValueError

    operator = m.group(2)
    if operator == "+":
        offset = float(m.group(3))
    else:
        offset = -float(m.group(3))

    return {
        "phase": m.group(1),
        "offset": offset
    }
