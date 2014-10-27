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

from instaseis import Source, FiniteSource

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
EVENT_FILE = os.path.join(DATA, "GCMT_event_STRAIT_OF_GIBRALTAR.xml")
SRF_FILE = os.path.join(DATA, "strike_slip_eq_10pts.srf")


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


def test_parse_srf_file():
    """
    Tests parsing from a .srf file.
    """
    finitesource = FiniteSource.from_srf_file(SRF_FILE, True)
    assert finitesource.npointsources == 10
    longitudes = np.array([0.0, 0.99925, 1.99849, 2.99774, 3.99698, 4.99623,
                           5.99548, 6.99472, 7.99397, 8.99322])

    for isrc, src in enumerate(finitesource):
        src_params = np.array([src.latitude, src.longitude, src.depth_in_m,
                               src.m_rr, src.m_tt, src.m_pp, src.m_rt,
                               src.m_rp, src.m_tp], dtype="float64")

        src_params_ref = np.array([
            0.00000000e+00, longitudes[isrc], 5.00000000e+04, 0.00000000e+00,
            -3.91886976e+03, 3.91886976e+03, -1.19980783e-13, 1.95943488e+03,
            3.20000000e+19])
        np.testing.assert_allclose(src_params, src_params_ref)


def test_resample_stf():
    """
    Tests resampling sliprates
    """
    finitesource = FiniteSource.from_srf_file(SRF_FILE, True)
    finitesource.resample_sliprate(dt=1., nsamp=10)

    stf_ref = np.array([
        0.00000000e+00, 1.60000000e-05, 3.20000000e-05, 4.80000000e-05,
        6.40000000e-05, 8.00000000e-05, 1.84000000e-04, 2.88000000e-04,
        3.92000000e-04, 4.96000000e-04])

    for isrc, src in enumerate(finitesource):
        np.testing.assert_allclose(stf_ref, src.sliprate)


def test_hypocenter():
    """
    Tests finding the hypocenter
    """
    finitesource = FiniteSource.from_srf_file(SRF_FILE, True)
    finitesource.find_hypocenter()

    assert finitesource.hypocenter_longitude == 0.0
    assert finitesource.hypocenter_latitude == 0.0
    assert finitesource.hypocenter_depth_in_m == 50e3
    assert finitesource.epicenter_longitude == 0.0
    assert finitesource.epicenter_latitude == 0.0


def test_min_max_functions():
    """
    Tests the min/max convenience functions
    """
    finitesource = FiniteSource.from_srf_file(SRF_FILE, True)
    finitesource.find_hypocenter()

    assert finitesource.min_depth_in_m == 50e3
    assert finitesource.max_depth_in_m == 50e3

    assert finitesource.min_longitude == 0.0
    assert finitesource.max_longitude == 8.99322

    assert finitesource.min_latitude == 0.0
    assert finitesource.max_latitude == 0.0


def test_M0():
    """
    Tests computation of scalar Moment.
    """
    strike = 10.
    dip = 20.
    rake = 30.
    M0 = 1e16
    source = Source.from_strike_dip_rake(0., 0., 0., strike, dip, rake, M0)

    assert source.M0 == M0


def test_M0_finite_source():
    """
    Tests computation of scalar Moment.
    """
    finitesource = FiniteSource.from_srf_file(SRF_FILE, True)
    finitesource.find_hypocenter()

    np.testing.assert_allclose(np.array([finitesource.M0]), np.array([3.2e20]))
