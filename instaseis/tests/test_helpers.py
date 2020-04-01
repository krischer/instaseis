#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for some helper routines.

:copyright:
    Lion Krischer (lion.krischer@gmail.com), 2020
:license:
    GNU Lesser General Public License, Version 3 [non-commercial/academic use]
    (http://www.gnu.org/copyleft/lgpl.html)
"""
from instaseis.helpers import io_chunker


def test_io_chunker():
    # Single continuous reads possible.
    assert io_chunker([0, 1, 2]) == [[0, 3]]
    assert io_chunker([0, 1, 2, 3, 4, 5, 6, 7]) == [[0, 8]]
    assert io_chunker([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) == [[0, 10]]
    assert io_chunker([1, 2, 3, 4, 5, 6, 7]) == [[1, 8]]

    # Does not work with unsorted arrays. Has to be explicitly sorted before
    # it can do anything.
    assert io_chunker([3, 2, 1, 0]) == [3, 2, 1, 0]
    assert io_chunker([1, 3, 2, 0]) == [1, 3, 2, 0]

    # A couple more complex cases.
    assert io_chunker([0, 1, 2, 4, 6, 7, 8]) == [[0, 3], 4, [6, 9]]
    assert io_chunker([0, 2, 4, 6, 7, 8, 10]) == [0, 2, 4, [6, 9], 10]
