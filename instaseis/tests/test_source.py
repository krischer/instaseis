#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for source handling.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
:license:
    GNU Lesser General Public License, Version 3 [non-commercial/academic use]
    (http://www.gnu.org/copyleft/lgpl.html)
"""
from __future__ import absolute_import

import io
import obspy
import os
import numpy as np
import pytest

from instaseis import Source, FiniteSource
from instaseis.helpers import elliptic_to_geocentric_latitude
from instaseis.source import moment2magnitude, magnitude2moment
from instaseis.source import (fault_vectors_lmn, strike_dip_rake_from_ln,
                              USGSParamFileParsingException)

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
EVENT_FILE = os.path.join(DATA, "GCMT_event_STRAIT_OF_GIBRALTAR.xml")
SRF_FILE = os.path.join(DATA, "strike_slip_eq_10pts.srf")
USGS_PARAM_FILE1 = os.path.join(DATA, "nepal.param")
USGS_PARAM_FILE2 = os.path.join(DATA, "chile.param")
USGS_PARAM_FILE_EMPTY = os.path.join(DATA, "empty.param")


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

    origin_time = obspy.UTCDateTime(2011, 8, 23, 17, 51, 4.6)

    src = Source.parse(filename)
    src_params = np.array([src.latitude, src.longitude, src.depth_in_m,
                           src.m_rr, src.m_tt, src.m_pp, src.m_rt, src.m_rp,
                           src.m_tp], dtype="float64")
    # Latitude will have assumed to be WGS84 and converted to geocentric
    # latitude.
    np.testing.assert_allclose(src_params, np.array(
        (elliptic_to_geocentric_latitude(37.91), -77.93, 12000, 4.71E17,
         3.81E15, -4.74E17, 3.99E16, -8.05E16,
         -1.23E17), dtype="float64"))
    assert src.origin_time == origin_time

    # Write again. Reset latitude beforehand.
    filename = os.path.join(str(tmpdir), "CMTSOLUTIONS2")
    src.latitude = 37.91
    src.write_CMTSOLUTION_file(filename)

    # This time there is no need to convert latitudes. Writing will convert
    # to WGS84 and reading will convert back.
    src = Source.parse(filename)
    src_params = np.array([src.latitude, src.longitude, src.depth_in_m,
                           src.m_rr, src.m_tt, src.m_pp, src.m_rt, src.m_rp,
                           src.m_tp], dtype="float64")
    np.testing.assert_allclose(src_params, np.array(
        (37.91, -77.93, 12000, 4.71E17, 3.81E15, -4.74E17, 3.99E16, -8.05E16,
         -1.23E17), dtype="float64"), rtol=1E-5)
    assert src.origin_time == origin_time


def _assert_src(src):
    """
    We constantly test the same event in various configurations.
    """
    # Latitude will have been assumed to be WGS84 and converted to geocentric!
    assert (src.latitude, src.longitude, src.depth_in_m, src.m_rr, src.m_tt,
            src.m_pp, src.m_rt, src.m_rp, src.m_tp) == \
           (elliptic_to_geocentric_latitude(36.97), -3.54, 609800.0, -2.16E18,
            5.36E17, 1.62E18, 1.3E16, 3.23E18, 1.75E18)

    # Also check the time!
    assert src.origin_time == obspy.UTCDateTime("2010-04-11T22:08:12.800000Z")


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
    cat = obspy.read_events(EVENT_FILE)
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


def test_parsing_empty_usgs_file():
    """
    Parsing an empty USGS file should fail.
    """
    with pytest.raises(USGSParamFileParsingException) as e:
        FiniteSource.from_usgs_param_file(USGS_PARAM_FILE_EMPTY)

    assert e.value.args[0] == 'No point sources found in the file.'


def test_parse_usgs_param_file():
    """
    Tests parsing from a .param file.
    """
    # single segment file
    finitesource = FiniteSource.from_usgs_param_file(USGS_PARAM_FILE1)
    np.testing.assert_almost_equal(finitesource.moment_magnitude,
                                   7.9374427577095901)
    assert finitesource.npointsources == 121

    # multi segment file
    finitesource = FiniteSource.from_usgs_param_file(USGS_PARAM_FILE2)
    np.testing.assert_almost_equal(finitesource.moment_magnitude,
                                   8.26413197488)
    assert finitesource.npointsources == 400


def test_parse_usgs_param_file_from_bytes_io_and_open_files():
    """
    Tests parsing a USGS file from a BytesIO stream and open files..
    """
    with io.open(USGS_PARAM_FILE1, "rb") as fh:
        finitesource = FiniteSource.from_usgs_param_file(fh)
    np.testing.assert_almost_equal(finitesource.moment_magnitude,
                                   7.9374427577095901)
    assert finitesource.npointsources == 121

    with io.open(USGS_PARAM_FILE1, "rb") as fh:
        with io.BytesIO(fh.read()) as buf:
            finitesource = FiniteSource.from_usgs_param_file(buf)
    np.testing.assert_almost_equal(finitesource.moment_magnitude,
                                   7.9374427577095901)
    assert finitesource.npointsources == 121

    with io.open(USGS_PARAM_FILE2, "rb") as fh:
        finitesource = FiniteSource.from_usgs_param_file(fh)
    np.testing.assert_almost_equal(finitesource.moment_magnitude,
                                   8.26413197488)
    assert finitesource.npointsources == 400

    with io.open(USGS_PARAM_FILE2, "rb") as fh:
        with io.BytesIO(fh.read()) as buf:
            finitesource = FiniteSource.from_usgs_param_file(buf)
    np.testing.assert_almost_equal(finitesource.moment_magnitude,
                                   8.26413197488)
    assert finitesource.npointsources == 400


def test_Haskell():
    """
    Tests Haskell source.
    """
    latitude, longitude, depth_in_m = 89.9999, 0., 10000.
    strike, dip, rake = 90., 90., 0.
    M0 = 1e20
    fault_length, fault_width = 1000e3, 200.
    rupture_velocity = 1000.
    nl, nw = 3, 3
    finitesource = FiniteSource.from_Haskell(
        latitude, longitude, depth_in_m, strike, dip, rake, M0, fault_length,
        fault_width, rupture_velocity, nl=nl, nw=nw)
    np.testing.assert_almost_equal(finitesource.moment_magnitude,
                                   7.33333333333333)


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


def test_moment2magnitude():
    """
    Tests computation of magnitude
    """
    M0 = 1e22
    Mw = moment2magnitude(M0)
    M0_calc = magnitude2moment(Mw)

    assert M0_calc == M0


def test_M0_finite_source():
    """
    Tests computation of scalar Moment.
    """
    finitesource = FiniteSource.from_srf_file(SRF_FILE, True)
    finitesource.find_hypocenter()

    np.testing.assert_allclose(np.array([finitesource.M0]), np.array([3.2e20]))


def test_CMT_finite_source():
    """
    Tests computation of CMT solution
    """
    finitesource = FiniteSource.from_srf_file(SRF_FILE, True)
    finitesource.compute_centroid()

    np.testing.assert_allclose(
        np.array([-3.918870e+04, 3.909051e+04, 9.819052e+01,
                  1.942254e+04, -5.476000e+03, 3.195987e+20]),
        finitesource.CMT.tensor_voigt, rtol=1E-5)

    np.testing.assert_allclose(np.array([finitesource.CMT.latitude]),
                               np.array([0.0]))
    np.testing.assert_allclose(np.array([finitesource.CMT.longitude]),
                               np.array([4.496608]))


def test_fault_vectors_lmn():
    """
    Tests computation of fault vectors l, m and n
    """
    strike, dip, rake = 35., 70., 17.
    l, m, n = fault_vectors_lmn(strike, dip, rake)

    # vectors should be perpendicular
    np.testing.assert_allclose(np.array([np.dot(l, n)]),
                               np.array([0.0]), atol=1e-15)
    np.testing.assert_allclose(np.array([np.dot(l, m)]),
                               np.array([0.0]), atol=1e-15)
    np.testing.assert_allclose(np.array([np.dot(m, n)]),
                               np.array([0.0]), atol=1e-15)

    np.testing.assert_allclose(np.cross(n, l), m, atol=1e-15)

    # vectors should be normalized
    np.testing.assert_allclose(np.array([1.0]), (l**2).sum())
    np.testing.assert_allclose(np.array([1.0]), (m**2).sum())
    np.testing.assert_allclose(np.array([1.0]), (n**2).sum())


def test_strike_dip_rake_from_ln():
    """
    Tests computation of strike dip and rake from fault vectors l and n
    """
    for strike, dip, rake in zip((42., 180., -140.), (22., 0., 90.),
                                 (17., 32., 120.)):
        l, m, n = fault_vectors_lmn(strike, dip, rake)
        s, d, r = strike_dip_rake_from_ln(l, n)

        np.testing.assert_allclose(np.array([strike]), np.array([s]))
        np.testing.assert_allclose(np.array([dip]), np.array([d]))
        np.testing.assert_allclose(np.array([rake]), np.array([r]))
