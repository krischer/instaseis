#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for source handling.

:copyright:
    Lion Krischer (lion.krischer@gmail.com), 2020
:license:
    GNU Lesser General Public License, Version 3 [non-commercial/academic use]
    (http://www.gnu.org/copyleft/lgpl.html)
"""
import io
import obspy
import os
import numpy as np
import pytest

from instaseis import (
    Source,
    FiniteSource,
    Receiver,
    SourceParseError,
    ForceSource,
)
from instaseis.helpers import elliptic_to_geocentric_latitude
from instaseis.source import moment2magnitude, magnitude2moment
from instaseis.source import (
    fault_vectors_lmn,
    strike_dip_rake_from_ln,
    USGSParamFileParsingException,
)

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
EVENT_FILE = os.path.join(DATA, "GCMT_event_STRAIT_OF_GIBRALTAR.xml")
SRF_FILE = os.path.join(DATA, "strike_slip_eq_10pts.srf")
USGS_PARAM_FILE1 = os.path.join(DATA, "nepal.param")
USGS_PARAM_FILE2 = os.path.join(DATA, "chile.param")
USGS_PARAM_FILE_EMPTY = os.path.join(DATA, "empty.param")


def test_parse_cmtsolutions_file(tmpdir):
    """
    Tests parsing from a CMTSOLUTIONS file.
    """
    filename = os.path.join(str(tmpdir), "CMTSOLUTIONS")
    lines = (
        " PDE 2011  8 23 17 51  4.60  37.9400  -77.9300   6.0 5.9 5.8 "
        "VIRGINIA",
        "event name:     201108231751A",
        "time shift:      1.00",
        "half duration:   1.8000",
        "latitude:       37.9100",
        "longitude:     -77.9300",
        "depth:          12.0000",
        "Mrr:       4.710000e+24",
        "Mtt:       3.810000e+22",
        "Mpp:      -4.740000e+24",
        "Mrt:       3.990000e+23",
        "Mrp:      -8.050000e+23",
        "Mtp:      -1.230000e+24",
    )
    with open(filename, "wt") as fh:
        fh.write("\n".join(lines))

    # This is the hypocentral time + 1 seconds (the time shift in the
    # CMTSOLUTION file).
    origin_time = obspy.UTCDateTime(2011, 8, 23, 17, 51, 5.6)

    src = Source.parse(filename)
    src_params = np.array(
        [
            src.latitude,
            src.longitude,
            src.depth_in_m,
            src.m_rr,
            src.m_tt,
            src.m_pp,
            src.m_rt,
            src.m_rp,
            src.m_tp,
        ],
        dtype="float64",
    )
    # Latitude will have assumed to be WGS84 and converted to geocentric
    # latitude. The import machinery should do that.
    np.testing.assert_allclose(
        src_params,
        np.array(
            (
                elliptic_to_geocentric_latitude(37.91),
                -77.93,
                12000,
                4.71e17,
                3.81e15,
                -4.74e17,
                3.99e16,
                -8.05e16,
                -1.23e17,
            ),
            dtype="float64",
        ),
    )
    assert src.origin_time == origin_time


def _assert_src(src):
    """
    We constantly test the same event in various configurations.
    """
    # Latitude will have been assumed to be WGS84 and converted to geocentric!
    assert (
        src.latitude,
        src.longitude,
        src.depth_in_m,
        src.m_rr,
        src.m_tt,
        src.m_pp,
        src.m_rt,
        src.m_rp,
        src.m_tp,
    ) == (
        elliptic_to_geocentric_latitude(36.97),
        -3.54,
        609800.0,
        -2.16e18,
        5.36e17,
        1.62e18,
        1.3e16,
        3.23e18,
        1.75e18,
    )

    # Also check the time!
    assert src.origin_time == obspy.UTCDateTime("2010-04-11T22:08:12.800000Z")


def test_parse_quakeml():
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


def test_parse_srf_file(tmpdir):
    """
    Tests parsing from a .srf file.
    """
    finitesource = FiniteSource.from_srf_file(SRF_FILE, True)
    assert finitesource.npointsources == 10
    longitudes = np.array(
        [
            0.0,
            0.99925,
            1.99849,
            2.99774,
            3.99698,
            4.99623,
            5.99548,
            6.99472,
            7.99397,
            8.99322,
        ]
    )

    for isrc, src in enumerate(finitesource):
        src_params = np.array(
            [
                src.latitude,
                src.longitude,
                src.depth_in_m,
                src.m_rr,
                src.m_tt,
                src.m_pp,
                src.m_rt,
                src.m_rp,
                src.m_tp,
            ],
            dtype="float64",
        )

        src_params_ref = np.array(
            [
                0.00000000e00,
                longitudes[isrc],
                5.00000000e04,
                0.00000000e00,
                -3.91886976e03,
                3.91886976e03,
                -1.19980783e-13,
                1.95943488e03,
                3.20000000e19,
            ]
        )
        np.testing.assert_allclose(src_params, src_params_ref)

    # Try parsing it again, but this time with a couple of empty lines in
    # front of it.
    filename = os.path.join(tmpdir.strpath, "temp.srf")
    with io.open(SRF_FILE, "rt") as fh_1, io.open(filename, "wt") as fh_2:
        fh_2.write("\n\n\n\n")
        fh_2.write(fh_1.read())
    finitesource_2 = FiniteSource.from_srf_file(filename, True)
    assert finitesource_2.npointsources == 10


def test_parsing_empty_usgs_file():
    """
    Parsing an empty USGS file should fail.
    """
    with pytest.raises(USGSParamFileParsingException) as e:
        FiniteSource.from_usgs_param_file(USGS_PARAM_FILE_EMPTY)

    assert e.value.args[0] == "No point sources found in the file."


def test_parse_usgs_param_file():
    """
    Tests parsing from a .param file.
    """
    # single segment file
    finitesource = FiniteSource.from_usgs_param_file(USGS_PARAM_FILE1)
    np.testing.assert_almost_equal(finitesource.moment_magnitude, 7.87077609)
    assert finitesource.npointsources == 121

    # multi segment file
    finitesource = FiniteSource.from_usgs_param_file(USGS_PARAM_FILE2)
    np.testing.assert_almost_equal(
        finitesource.moment_magnitude, 8.1974653082088
    )
    assert finitesource.npointsources == 400


def test_parse_usgs_param_file_from_bytes_io_and_open_files():
    """
    Tests parsing a USGS file from a BytesIO stream and open files..
    """
    with io.open(USGS_PARAM_FILE1, "rb") as fh:
        finitesource = FiniteSource.from_usgs_param_file(fh)
    np.testing.assert_almost_equal(
        finitesource.moment_magnitude, 7.8707760910429236
    )
    assert finitesource.npointsources == 121

    with io.open(USGS_PARAM_FILE1, "rb") as fh:
        with io.BytesIO(fh.read()) as buf:
            finitesource = FiniteSource.from_usgs_param_file(buf)
    np.testing.assert_almost_equal(finitesource.moment_magnitude, 7.87077609)
    assert finitesource.npointsources == 121

    with io.open(USGS_PARAM_FILE2, "rb") as fh:
        finitesource = FiniteSource.from_usgs_param_file(fh)
    np.testing.assert_almost_equal(
        finitesource.moment_magnitude, 8.1974653082088
    )
    assert finitesource.npointsources == 400

    with io.open(USGS_PARAM_FILE2, "rb") as fh:
        with io.BytesIO(fh.read()) as buf:
            finitesource = FiniteSource.from_usgs_param_file(buf)
    np.testing.assert_almost_equal(
        finitesource.moment_magnitude, 8.19746530820887
    )
    assert finitesource.npointsources == 400


def test_haskell():
    """
    Tests Haskell source.
    """
    latitude, longitude, depth_in_m = 89.9999, 0.0, 10000.0
    strike, dip, rake = 90.0, 90.0, 0.0
    M0 = 1e20  # NOQA
    fault_length, fault_width = 1000e3, 200.0
    rupture_velocity = 1000.0
    nl, nw = 3, 3
    finitesource = FiniteSource.from_Haskell(
        latitude,
        longitude,
        depth_in_m,
        strike,
        dip,
        rake,
        M0,
        fault_length,
        fault_width,
        rupture_velocity,
        nl=nl,
        nw=nw,
    )
    np.testing.assert_almost_equal(
        finitesource.moment_magnitude, 7.26666666666
    )

    # Should raise an error if above ground.
    with pytest.raises(ValueError) as err:
        FiniteSource.from_Haskell(
            latitude,
            longitude,
            -100.0,
            strike,
            dip,
            rake,
            M0,
            fault_length,
            fault_width,
            rupture_velocity,
            nl=nl,
            nw=nw,
        )
    assert err.value.args[0].startswith("At least one source point outside")

    # Manually settings trise and tfall.
    finitesource = FiniteSource.from_Haskell(
        latitude,
        longitude,
        depth_in_m,
        strike,
        dip,
        rake,
        M0,
        fault_length,
        fault_width,
        rupture_velocity,
        nl=nl,
        nw=nw,
        trise=1.0,
        tfall=1.0,
    )
    np.testing.assert_almost_equal(
        finitesource.moment_magnitude, 7.26666666666666666
    )


def test_resample_stf():
    """
    Tests resampling sliprates
    """
    finitesource = FiniteSource.from_srf_file(SRF_FILE, True)
    finitesource.resample_sliprate(dt=1.0, nsamp=10)

    stf_ref = np.array(
        [
            0.00000000e00,
            1.60000000e-05,
            3.20000000e-05,
            4.80000000e-05,
            6.40000000e-05,
            8.00000000e-05,
            1.84000000e-04,
            2.88000000e-04,
            3.92000000e-04,
            4.96000000e-04,
        ]
    )

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

    assert finitesource.rupture_duration == 222.22222


def test_m0():
    """
    Tests computation of scalar Moment.
    """
    strike = 10.0
    dip = 20.0
    rake = 30.0
    m0 = 1e16
    source = Source.from_strike_dip_rake(0.0, 0.0, 0.0, strike, dip, rake, m0)

    assert source.M0 == m0


def test_moment2magnitude():
    """
    Tests computation of magnitude
    """
    m0 = 1e22
    mw = moment2magnitude(m0)
    m0_calc = magnitude2moment(mw)

    assert m0_calc == m0


def test_m0_finite_source():
    """
    Tests computation of scalar Moment.
    """
    finitesource = FiniteSource.from_srf_file(SRF_FILE, True)
    finitesource.find_hypocenter()

    np.testing.assert_allclose(np.array([finitesource.M0]), np.array([3.2e20]))


def test_cmt_finite_source():
    """
    Tests computation of CMT solution
    """
    finitesource = FiniteSource.from_srf_file(SRF_FILE, True)
    finitesource.compute_centroid()

    np.testing.assert_allclose(
        np.array(
            [
                -3.918870e04,
                3.909051e04,
                9.819052e01,
                1.942254e04,
                -5.476000e03,
                3.195987e20,
            ]
        ),
        finitesource.CMT.tensor_voigt,
        rtol=1e-5,
    )

    np.testing.assert_allclose(
        np.array([finitesource.CMT.latitude]), np.array([0.0])
    )
    np.testing.assert_allclose(
        np.array([finitesource.CMT.longitude]), np.array([4.496608])
    )


def test_fault_vectors_lmn():
    """
    Tests computation of fault vectors l, m and n
    """
    strike, dip, rake = 35.0, 70.0, 17.0
    l, m, n = fault_vectors_lmn(strike, dip, rake)

    # vectors should be perpendicular
    np.testing.assert_allclose(
        np.array([np.dot(l, n)]), np.array([0.0]), atol=1e-15
    )
    np.testing.assert_allclose(
        np.array([np.dot(l, m)]), np.array([0.0]), atol=1e-15
    )
    np.testing.assert_allclose(
        np.array([np.dot(m, n)]), np.array([0.0]), atol=1e-15
    )

    np.testing.assert_allclose(np.cross(n, l), m, atol=1e-15)

    # vectors should be normalized
    np.testing.assert_allclose(np.array([1.0]), (l ** 2).sum())
    np.testing.assert_allclose(np.array([1.0]), (m ** 2).sum())
    np.testing.assert_allclose(np.array([1.0]), (n ** 2).sum())


def test_strike_dip_rake_from_ln():
    """
    Tests computation of strike dip and rake from fault vectors l and n
    """
    for strike, dip, rake in zip(
        (42.0, 180.0, -140.0), (22.0, 0.0, 90.0), (17.0, 32.0, 120.0)
    ):
        l, m, n = fault_vectors_lmn(strike, dip, rake)
        s, d, r = strike_dip_rake_from_ln(l, n)

        np.testing.assert_allclose(np.array([strike]), np.array([s]))
        np.testing.assert_allclose(np.array([dip]), np.array([d]))
        np.testing.assert_allclose(np.array([rake]), np.array([r]))


def test_equality_methods():
    """
    Tests the (in)equality methods of source and receiver objects. They both
    inherit from the same base class thus they are tested here - but it
    would also be suitable for the receiver tests.
    """
    src1 = Source(latitude=1, longitude=2)
    src2 = Source(latitude=3, longitude=3)
    rec1 = Receiver(latitude=1, longitude=2)
    rec2 = Receiver(latitude=3, longitude=3)

    assert src1 == src1
    assert src2 == src2
    assert rec1 == rec1
    assert rec2 == rec2

    assert src1 != src2
    assert src1 != rec1
    assert src1 != rec2

    assert src2 != src1
    assert src2 != rec1
    assert src2 != rec2

    assert rec1 != rec2
    assert rec1 != src1
    assert rec1 != src2

    assert rec2 != src1
    assert rec2 != src1
    assert rec2 != src2


def test_radian_calculations():
    src = Source(latitude=90.0, longitude=180.0)
    assert np.isclose(src.latitude, 90.0)
    assert np.isclose(src.latitude_rad, np.pi / 2.0)
    assert np.isclose(src.longitude, 180.0)
    assert np.isclose(src.longitude_rad, np.pi)


def test_event_parsing_failure_states():
    """
    Tests the failures when parsing an event.
    """
    # Random string.
    with pytest.raises(SourceParseError) as err:
        Source.parse("random strings")
    assert err.value.args[0] == "Could not parse the given source."

    # Empty catalog.
    cat = obspy.read_events()
    cat.events = []
    with pytest.raises(SourceParseError) as err:
        Source.parse(cat)
    assert err.value.args[0] == "Event catalog contains zero events."

    # Catalog with lots of events.
    cat = obspy.read_events()
    with pytest.raises(SourceParseError) as err:
        Source.parse(cat)
    assert err.value.args[0].startswith("Event catalog contains 3 events")

    event = obspy.read_events(EVENT_FILE)[0]

    # Event with no origin.
    ev = event.copy()
    ev.origins = []
    with pytest.raises(SourceParseError) as err:
        Source.parse(ev)
    assert err.value.args[0] == "Event must contain an origin."

    # Event with no focmec.
    ev = event.copy()
    ev.focal_mechanisms = []
    with pytest.raises(SourceParseError) as err:
        Source.parse(ev)
    assert err.value.args[0] == "Event must contain a focal mechanism."

    # Event with no moment tensor.
    ev = event.copy()
    ev.focal_mechanisms[0].moment_tensor = None
    # Force this to None to prevent the resource id mechanism to "resurrect"
    # the focal mechanism.
    ev.preferred_focal_mechanism_id = None
    with pytest.raises(SourceParseError) as err:
        Source.parse(ev)
    assert err.value.args[0] == "Event must contain a moment tensor."


def test_sliprate_convenience_methods():
    """
    Tests some convenience methods of sliprates.
    """
    src = Source(latitude=0.0, longitude=90.0)
    src.set_sliprate_dirac(2.0, 5)
    np.testing.assert_allclose(np.array([0.5, 0, 0, 0, 0]), src.sliprate)

    src = Source(latitude=0.0, longitude=90.0)
    src.set_sliprate_lp(2.0, 5, 0.1)
    np.testing.assert_allclose(
        np.array([0.023291, 0.111382, 0.211022, 0.186723, 0.045481]),
        src.sliprate,
        rtol=1e-3,
    )

    src = Source(latitude=0.0, longitude=90.0)
    src.sliprate = np.ones(5)
    src.dt = 0.25
    src.normalize_sliprate()
    np.testing.assert_allclose(np.ones(5), src.sliprate)


def test_sliprate_convenience_methods_finite_source():
    """
    Tests some convenience methods of sliprates for finite sources.
    """
    src = Source(latitude=0.0, longitude=90.0)
    fs = FiniteSource(pointsources=[src])
    fs.set_sliprate_dirac(2.0, 5)
    np.testing.assert_allclose(np.array([0.5, 0, 0, 0, 0]), src.sliprate)

    src = Source(latitude=0.0, longitude=90.0)
    fs = FiniteSource(pointsources=[src])
    fs.set_sliprate_lp(2.0, 5, 0.1)
    np.testing.assert_allclose(
        np.array([0.023291, 0.111382, 0.211022, 0.186723, 0.045481]),
        src.sliprate,
        rtol=1e-3,
    )

    src = Source(latitude=0.0, longitude=90.0)
    src.sliprate = np.ones(5)
    src.dt = 0.25
    fs = FiniteSource(pointsources=[src])
    fs.normalize_sliprate()
    np.testing.assert_allclose(np.ones(5), src.sliprate)


def test_sliprate_convenience_methods_force_source():
    """
    Tests some convenience methods of sliprates for force sources.
    """
    src = ForceSource(latitude=0.0, longitude=90.0)
    src.set_sliprate_dirac(2.0, 5)
    np.testing.assert_allclose(np.array([0.5, 0, 0, 0, 0]), src.sliprate)

    src = ForceSource(latitude=0.0, longitude=90.0)
    src.set_sliprate_lp(2.0, 5, 0.1)
    np.testing.assert_allclose(
        np.array([0.023291, 0.111382, 0.211022, 0.186723, 0.045481]),
        src.sliprate,
        rtol=1e-3,
    )

    src = ForceSource(latitude=0.0, longitude=90.0)
    src.sliprate = np.ones(5)
    src.dt = 0.25
    src.normalize_sliprate()
    np.testing.assert_allclose(np.ones(5), src.sliprate)


def test_str_method_of_src():
    src = Source(latitude=0.0, longitude=90.0)
    assert str(src) == (
        "Instaseis Source:\n"
        "\tOrigin Time      : 1970-01-01T00:00:00.000000Z\n"
        "\tLongitude        :   90.0 deg\n"
        "\tLatitude         :    0.0 deg\n"
        "\tDepth            :  not set km\n"
        "\tMoment Magnitude :   -inf\n"
        "\tScalar Moment    :   0.00e+00 Nm\n"
        "\tMrr              :   0.00e+00 Nm\n"
        "\tMtt              :   0.00e+00 Nm\n"
        "\tMpp              :   0.00e+00 Nm\n"
        "\tMrt              :   0.00e+00 Nm\n"
        "\tMrp              :   0.00e+00 Nm\n"
        "\tMtp              :   0.00e+00 Nm\n"
    )


def test_str_method_of_force_source():
    src = ForceSource(latitude=0.0, longitude=0.0, f_r=1.0, f_t=2.0, f_p=3.0)
    assert str(src) == (
        "Instaseis Force Source:\n"
        "\tOrigin Time      : 1970-01-01T00:00:00.000000Z\n"
        "\tLongitude :    0.0 deg\n"
        "\tLatitude  :    0.0 deg\n"
        "\tFr        :   1.00e+00 N\n"
        "\tFt        :   2.00e+00 N\n"
        "\tFp        :   3.00e+00 N\n"
    )


def test_str_method_of_finite_source():
    finitesource = FiniteSource.from_srf_file(SRF_FILE, True)
    assert str(finitesource) == (
        "Instaseis Finite Source:\n"
        "\tMoment Magnitude     : 7.60\n"
        "\tScalar Moment        :   3.20e+20 Nm\n"
        "\t#Point Sources       : 10\n"
        "\tRupture Duration     :  222.2 s\n"
        "\tTime Shift           :    0.0 s\n"
        "\tMin Depth            : 50000.0 m\n"
        "\tMax Depth            : 50000.0 m\n"
        "\tHypocenter Depth     : 50000.0 m\n"
        "\tMin Latitude         :    0.0 deg\n"
        "\tMax Latitude         :    0.0 deg\n"
        "\tHypocenter Latitude  :    0.0 deg\n"
        "\tMin Longitude        :    0.0 deg\n"
        "\tMax Longitude        :    9.0 deg\n"
        "\tHypocenter Longitude :    0.0 deg\n"
    )


def test_properties_force_source():
    """
    Tests some properties of the force source.
    """
    src = ForceSource(latitude=0.0, longitude=0.0, f_r=1.0, f_t=2.0, f_p=3.0)
    np.testing.assert_allclose(src.force_tpr, [2.0, 3.0, 1.0])
    np.testing.assert_allclose(src.force_rtp, [1.0, 2.0, 3.0])


def test_finite_source_iteration_over_empty_fs():
    """
    Raises a ValueError when trying to iterate over an empty finite source
    object. This is safe-guard against obvious errors so in this particular
    case its better to be explicit instead of just not looping.
    """
    fs = FiniteSource()
    with pytest.raises(ValueError) as err:
        [_i for _i in fs]
    assert err.value.args[0] == "FiniteSource not Initialized"


def test_reading_finite_source_with_slip_along_u2_axis():
    """
    Tests SRF files with slips along the u2 axis with a constructed file.
    """
    # Constructed file with known properties.
    filename = os.path.join(DATA, "strike_slip_eq_2pts.srf")
    finitesource = FiniteSource.from_srf_file(filename, True)
    assert finitesource.npointsources == 3


def test_print_regressions():
    """
    Guard against a regression for printing a source object.
    """
    src = Source.from_strike_dip_rake(
        latitude=27.77,
        longitude=85.37,
        depth_in_m=12000.0,
        M0=1e21,
        strike=32.0,
        dip=62.0,
        rake=90.0,
    )
    assert (str(src)) == (
        "Instaseis Source:\n"
        "\tOrigin Time      : 1970-01-01T00:00:00.000000Z\n"
        "\tLongitude        :   85.4 deg\n"
        "\tLatitude         :   27.8 deg\n"
        "\tDepth            : 1.2e+01 km km\n"
        "\tMoment Magnitude :   7.93\n"
        "\tScalar Moment    :   1.00e+21 Nm\n"
        "\tMrr              :   8.29e+20 Nm\n"
        "\tMtt              :  -2.33e+20 Nm\n"
        "\tMpp              :  -5.96e+20 Nm\n"
        "\tMrt              :   2.96e+20 Nm\n"
        "\tMrp              :   4.74e+20 Nm\n"
        "\tMtp              :  -3.73e+20 Nm\n"
    )
