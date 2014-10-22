#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for source handling.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
from __future__ import absolute_import

import obspy
import os
import numpy as np

from instaseis import Source

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
EVENT_FILE = os.path.join(DATA, "GCMT_event_STRAIT_OF_GIBRALTAR.xml")


def test_parse_CMTSOLUTIONS_file(tmpdir):
    """
    Tests parsing from a CMTSOLUTIONS file.
    """
    filename = os.path.join(str(tmpdir), "CMTSOLUTIONS")
    lines = (
        "PDEW2011  8 23 17 51  4.60  37.9400  -77.9300   6.0 5.9 5.8 VIRGINIA",
        "event name:     201108231751A",
        "time shift:      1.1100",
        "half duration:   1.8000",
        "latitude:       37.9100",
        "longitude:     -77.9300",
        "depth:          12.0000",
        "Mrr:       4.710000e+24",
        "Mtt:       3.810000e+22",
        "Mpp:      -4.740000e+24",
        "Mrt:       3.990000e+23",
        "Mrp:      -8.050000e+23",
        "Mtp:      -1.230000e+24")
    with open(filename, "wt") as fh:
        fh.write("\n".join(lines))

    src = Source.parse(filename)
    src_params = np.array([src.latitude, src.longitude, src.depth_in_m,
                           src.m_rr, src.m_tt, src.m_pp, src.m_rt, src.m_rp,
                           src.m_tp], dtype="float64")
    np.testing.assert_allclose(src_params, np.array(
        (37.91, -77.93, 12000, 4.71E17, 3.81E15, -4.74E17, 3.99E16, -8.05E16,
         -1.23E17), dtype="float64"))


def _assert_src(src):
    """
    We constantly test the same event in various configurations.
    """
    assert (src.latitude, src.longitude, src.depth_in_m, src.m_rr, src.m_tt,
            src.m_pp, src.m_rt, src.m_rp, src.m_tp) == \
           (36.97, -3.54, 609800.0, -2.16E18, 5.36E17, 1.62E18, 1.3E16,
            3.23E18, 1.75E18)


def test_parse_QuakeML():
    """
    Tests parsing from a QuakeML file.
    """
    src = Source.parse(EVENT_FILE)
    _assert_src(src)


def test_parse_obspy_objects():
    """
    Tests parsing from ObsPy objects.
    """
    cat = obspy.readEvents(EVENT_FILE)
    ev = cat[0]

    _assert_src(Source.parse(cat))
    _assert_src(Source.parse(ev))
