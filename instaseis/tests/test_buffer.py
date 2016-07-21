#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for the array buffer.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2016
:license:
    GNU Lesser General Public License, Version 3 [non-commercial/academic use]
    (http://www.gnu.org/copyleft/lgpl.html)
"""
from __future__ import absolute_import, division

import numpy as np

from instaseis.database_interfaces.mesh import Buffer


def test_buffer():
    buf = Buffer(max_size_in_mb=1.0)
    assert buf.efficiency == 0.0
    # Exactly one bytes less then a megabyte.
    buf.add("a", np.empty(1024 ** 2 - 1, dtype=np.int8))

    assert "a" in buf
    assert buf._total_size == 1024 ** 2 - 1

    buf.add("b", np.empty(2, dtype=np.int8))

    assert "a" not in buf
    assert "b" in buf
    assert buf._total_size == 2

    buf.add("c", np.empty(2, dtype=np.int8))
    assert buf._total_size == 4

    assert buf.get_size_mb() == 4 / (1024 ** 2)

    # Twice in, once not.
    assert buf.efficiency == 2.0 / 3.0

    # Once more not in.
    assert "d" not in buf
    assert buf.efficiency == 2.0 / 4.0
