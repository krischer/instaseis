#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for the finite source route of the Instaseis server.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2015
:license:
    GNU Lesser General Public License, Version 3 [non-commercial/academic use]
    (http://www.gnu.org/copyleft/lgpl.html)
"""
from __future__ import absolute_import

import copy
import io
import os
import zipfile

import obspy
import numpy as np
from .tornado_testing_fixtures import *  # NOQA
from .tornado_testing_fixtures import _assemble_url

import instaseis

# Conditionally import mock either from the stdlib or as a separate library.
import sys
if sys.version_info[0] == 2:
    import mock
else:
    import unittest.mock as mock

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
USGS_PARAM_FILE1 = os.path.join(DATA, "nepal.param")
USGS_PARAM_FILE2 = os.path.join(DATA, "chile.param")


def _parse_finite_source(filename):
    """
    Helper function parsing a finite source exactly how it is parsed in the
    finite source server route.
    """
    fs = instaseis.FiniteSource.from_usgs_param_file(
        filename, npts=10000, dt=0.1, trise_min=1.0)

    # All test databases are the same.
    dominant_period = 100.0
    dt = 24.724845445855724
    npts = 73

    # Things are now on purpose a bit different than in the actual
    # implementation to actually test something....
    for src in fs.pointsources:
        src.sliprate = np.concatenate(
            [np.zeros(10), src.sliprate, np.zeros(10)])
        src.time_shift += 10 * dt

    fs.lp_sliprate(freq=1.0 / dominant_period, zerophase=True)
    fs.resample_sliprate(dt=dt, nsamp=npts + 20)

    return fs


def test_finite_source_retrieval(all_clients):
    """
    Tests if the finite sources requested from the server are identical to
    the one requested with the local instaseis client with some required
    tweaks.
    """
    client = all_clients

    db = instaseis.open_db(client.filepath)

    # Finite sources only work with reciprocal databases.
    if not client.is_reciprocal:
        return

    basic_parameters = {
        "receiverlongitude": 11,
        "receiverlatitude": 22,
        "format": "miniseed"}

    with io.open(USGS_PARAM_FILE1, "rb") as fh:
        body = fh.read()

    # default parameters
    params = copy.deepcopy(basic_parameters)
    request = client.fetch(_assemble_url('finite_source', **params),
                           method="POST", body=body)
    assert request.code == 200
    st_server = obspy.read(request.buffer)
    for tr in st_server:
        assert tr.stats._format == "MSEED"

    # Parse the finite source.
    fs = _parse_finite_source(USGS_PARAM_FILE1)
    rec = instaseis.Receiver(latitude=22, longitude=11, network="XX",
                             station="SYN")

    st_db = db.get_seismograms_finite_source(sources=fs, receiver=rec)
    # The origin time is the time of the first sample in the route.
    for tr in st_db:
        # Cut away the first ten samples as they have been previously added.
        tr.data = tr.data[10:]
        tr.stats.starttime = obspy.UTCDateTime(1900, 1, 1)

    for tr_db, tr_server in zip(st_db, st_server):
        # Sample spacing is very similar but not equal due to floating point
        # accuracy.
        np.testing.assert_allclose(tr_server.stats.delta, tr_db.stats.delta)
        tr_server.stats.delta = tr_db.stats.delta
        del tr_server.stats._format
        del tr_server.stats.mseed

        assert tr_db.stats == tr_server.stats
        np.testing.assert_allclose(tr_db.data, tr_server.data)

    return

    # ObsPy needs the filename to be able to directly unpack zip files. We
    # don't have a filename here so we unpack manually.
    st_server = obspy.Stream()
    zip_obj = zipfile.ZipFile(request.buffer)
    for name in zip_obj.namelist():
        st_server += obspy.read(io.BytesIO(zip_obj.read(name)))
    for tr in st_server:
        assert tr.stats._format == "SAC"

    st_db = db.get_greens_function(
        epicentral_distance_in_degree=params['sourcedistanceindegrees'],
        source_depth_in_m=params['sourcedepthinmeters'], origin_time=time,
        definition="seiscomp")

    for tr_server, tr_db in zip(st_server, st_db):
        # Remove the additional stats from both.
        del tr_server.stats.sac
        del tr_server.stats._format
        del tr_db.stats.instaseis
        # Sample spacing is very similar but not equal due to floating point
        # accuracy.
        np.testing.assert_allclose(tr_server.stats.delta, tr_db.stats.delta)
        tr_server.stats.delta = tr_db.stats.delta
        assert tr_server.stats == tr_db.stats
        # Relative tolerance not particularly useful when testing super
        # small values.
        np.testing.assert_allclose(tr_server.data, tr_db.data,
                                   atol=1E-10 * tr_server.data.ptp())

    # miniseed
    params = copy.deepcopy(basic_parameters)
    params["format"] = "miniseed"
    request = client.fetch(_assemble_url('greens_function', **params))
    assert request.code == 200
    st_server = obspy.read(request.buffer)

    for tr in st_server:
        assert tr.stats._format == "MSEED"

    st_db = db.get_greens_function(
        epicentral_distance_in_degree=params['sourcedistanceindegrees'],
        source_depth_in_m=params['sourcedepthinmeters'], origin_time=time,
        definition="seiscomp")

    for tr_server, tr_db in zip(st_server, st_db):
        # Remove the additional stats from both.
        del tr_server.stats.mseed
        del tr_server.stats._format
        del tr_db.stats.instaseis
        # Sample spacing is very similar but not equal due to floating point
        # accuracy.
        np.testing.assert_allclose(tr_server.stats.delta, tr_db.stats.delta)
        tr_server.stats.delta = tr_db.stats.delta
        assert tr_server.stats == tr_db.stats
        # Relative tolerance not particularly useful when testing super
        # small values.
        np.testing.assert_allclose(tr_server.data, tr_db.data,
                                   atol=1E-10 * tr_server.data.ptp())

    # One with a label.
    params = copy.deepcopy(basic_parameters)
    params["format"] = "miniseed"
    params["label"] = "random_things"
    request = client.fetch(_assemble_url('greens_function', **params))
    assert request.code == 200

    cd = request.headers["Content-Disposition"]
    assert cd.startswith("attachment; filename=random_things_")
    assert cd.endswith(".mseed")

    # One more with resampling parameters and different units.
    params = copy.deepcopy(basic_parameters)
    params["format"] = "miniseed"
    params["dt"] = 0.1
    params["kernelwidth"] = 2
    params["units"] = "acceleration"
    request = client.fetch(_assemble_url('greens_function', **params))
    assert request.code == 200
    st_server = obspy.read(request.buffer)

    st_db = db.get_greens_function(
        epicentral_distance_in_degree=params['sourcedistanceindegrees'],
        source_depth_in_m=params['sourcedepthinmeters'], origin_time=time,
        definition="seiscomp", dt=0.1, kernelwidth=2, kind="acceleration")

    for tr_server, tr_db in zip(st_server, st_db):
        # Remove the additional stats from both.
        del tr_server.stats.mseed
        del tr_server.stats._format
        del tr_db.stats.instaseis
        # Sample spacing is very similar but not equal due to floating point
        # accuracy.
        np.testing.assert_allclose(tr_server.stats.delta, tr_db.stats.delta)
        tr_server.stats.delta = tr_db.stats.delta
        assert tr_server.stats == tr_db.stats
        # Relative tolerance not particularly useful when testing super
        # small values.
        np.testing.assert_allclose(tr_server.data, tr_db.data,
                                   atol=1E-10 * tr_server.data.ptp())

    # One simulating a crash in the underlying function.
    params = copy.deepcopy(basic_parameters)
    params["format"] = "miniseed"

    with mock.patch("instaseis.base_instaseis_db.BaseInstaseisDB"
                    ".get_greens_function") as p:
        def raise_err():
            raise ValueError("random crash")

        p.side_effect = raise_err
        request = client.fetch(_assemble_url('greens_function', **params))

    assert request.code == 400
    assert request.reason == ("Could not extract Green's function. Make "
                              "sure, the parameters are valid, and the depth "
                              "settings are correct.")

    # Two more simulating logic erros that should not be able to happen.
    params = copy.deepcopy(basic_parameters)
    params["format"] = "miniseed"

    with mock.patch("instaseis.base_instaseis_db.BaseInstaseisDB"
                    ".get_greens_function") as p:
        st = obspy.read()
        for tr in st:
            tr.stats.starttime = obspy.UTCDateTime(1E5)

        p.return_value = st
        request = client.fetch(_assemble_url('greens_function', **params))

    assert request.code == 500
    assert request.reason == ("Starttime more than one hour before the "
                              "starttime of the seismograms.")

    params = copy.deepcopy(basic_parameters)
    params["format"] = "miniseed"

    with mock.patch("instaseis.base_instaseis_db.BaseInstaseisDB"
                    ".get_greens_function") as p:
        st = obspy.read()
        for tr in st:
            tr.stats.starttime = obspy.UTCDateTime(0)
            tr.stats.delta = 0.0001

        p.return_value = st
        request = client.fetch(_assemble_url('greens_function', **params))

    assert request.code == 500
    assert request.reason.startswith("Endtime larger then the extracted "
                                     "endtime")
