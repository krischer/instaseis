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
USGS_PARAM_FILE = os.path.join(DATA, "nepal.param")


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
    shift = fs.time_shift
    for src in fs.pointsources:
        src.sliprate = np.concatenate(
            [np.zeros(10), src.sliprate, np.zeros(10)])
        src.time_shift += 10 * dt - shift

    fs.lp_sliprate(freq=1.0 / dominant_period, zerophase=True)
    fs.resample_sliprate(dt=dt, nsamp=npts + 20)

    return fs


def test_sending_non_USGS_file(reciprocal_clients):
    """
    Tests error if a non-USGS file is sent.
    """
    client = reciprocal_clients

    with io.open(__file__) as fh:
        body = fh.read()

    # default parameters
    params = {
        "receiverlongitude": 11,
        "receiverlatitude": 22}
    request = client.fetch(_assemble_url('finite_source', **params),
                           method="POST", body=body)
    assert request.code == 400
    assert request.reason == "Could not parse the body contents. Incorrect " \
                             "USGS param file?"


def test_finite_source_retrieval(reciprocal_clients):
    """
    Tests if the finite sources requested from the server are identical to
    the one requested with the local instaseis client with some required
    tweaks.
    """
    client = reciprocal_clients

    db = instaseis.open_db(client.filepath)

    basic_parameters = {
        "receiverlongitude": 11,
        "receiverlatitude": 22,
        "format": "miniseed"}

    with io.open(USGS_PARAM_FILE, "rb") as fh:
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
    fs = _parse_finite_source(USGS_PARAM_FILE)
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

    # Once again but this time request a SAC file.
    params = copy.deepcopy(basic_parameters)
    params["format"] = "saczip"
    request = client.fetch(_assemble_url('finite_source', **params),
                           method="POST", body=body)
    assert request.code == 200
    st_server = obspy.Stream()
    zip_obj = zipfile.ZipFile(request.buffer)
    for name in zip_obj.namelist():
        st_server += obspy.read(io.BytesIO(zip_obj.read(name)))
    for tr in st_server:
        assert tr.stats._format == "SAC"

    for tr_db, tr_server in zip(st_db, st_server):
        # Sample spacing is very similar but not equal due to floating point
        # accuracy.
        np.testing.assert_allclose(tr_server.stats.delta, tr_db.stats.delta)
        tr_server.stats.delta = tr_db.stats.delta
        del tr_server.stats._format
        del tr_server.stats.sac

        assert tr_db.stats == tr_server.stats
        np.testing.assert_allclose(tr_db.data, tr_server.data)

    # One with a label.
    params = copy.deepcopy(basic_parameters)
    params["label"] = "random_things"
    request = client.fetch(_assemble_url('finite_source', **params),
                           method="POST", body=body)
    assert request.code == 200

    cd = request.headers["Content-Disposition"]
    assert cd.startswith("attachment; filename=random_things_")
    assert cd.endswith(".mseed")

    # One simulating a crash in the underlying function.
    params = copy.deepcopy(basic_parameters)

    with mock.patch("instaseis.base_instaseis_db.BaseInstaseisDB"
                    ".get_seismograms_finite_source") as p:
        def raise_err():
            raise ValueError("random crash")

        p.side_effect = raise_err
        request = client.fetch(_assemble_url('finite_source', **params),
                               method="POST", body=body)

    assert request.code == 400
    assert request.reason == ("Could not extract finite source seismograms. "
                              "Make sure, the parameters are valid, and the "
                              "depth settings are correct.")

    # Simulating a logic error that should not be able to happen.
    params = copy.deepcopy(basic_parameters)
    with mock.patch("instaseis.base_instaseis_db.BaseInstaseisDB"
                    ".get_seismograms_finite_source") as p:
        # Longer than the database returned stream thus the endtime is out
        # of bounds.
        st = obspy.read()

        p.return_value = st
        request = client.fetch(_assemble_url('finite_source', **params),
                               method="POST", body=body)

    assert request.code == 500
    assert request.reason.startswith("Endtime larger than the extracted "
                                     "endtime")

    # One more with resampling parameters and different units.
    params = copy.deepcopy(basic_parameters)
    # We must have a sampling rate that cleanly fits in the existing one,
    # otherwise we cannot fake the cutting.
    dt_new = 24.724845445855724 / 10
    params["dt"] = dt_new
    params["kernelwidth"] = 2
    params["units"] = "acceleration"

    st_db = db.get_seismograms_finite_source(sources=fs, receiver=rec,
                                             dt=dt_new, kernelwidth=2,
                                             kind="acceleration")
    # The origin time is the time of the first sample in the route.
    for tr in st_db:
        # Cut away the first ten samples as they have been previously added.
        tr.data = tr.data[100:]
        tr.stats.starttime = obspy.UTCDateTime(1900, 1, 1)

    request = client.fetch(_assemble_url('finite_source', **params),
                           method="POST", body=body)
    assert request.code == 200
    st_server = obspy.read(request.buffer)

    # Cut some parts in the middle to avoid any potential boundary effects.
    st_db.trim(obspy.UTCDateTime(1900, 1, 1, 0, 4),
               obspy.UTCDateTime(1900, 1, 1, 0, 14))
    st_server.trim(obspy.UTCDateTime(1900, 1, 1, 0, 4),
                   obspy.UTCDateTime(1900, 1, 1, 0, 14))

    for tr_db, tr_server in zip(st_db, st_server):
        # Sample spacing and times are very similar but not identical due to
        # floating point inaccuracies in the arithmetics.
        np.testing.assert_allclose(tr_server.stats.delta, tr_db.stats.delta)
        np.testing.assert_allclose(tr_server.stats.starttime.timestamp,
                                   tr_db.stats.starttime.timestamp)
        tr_server.stats.delta = tr_db.stats.delta
        tr_server.stats.starttime = tr_db.stats.starttime
        del tr_server.stats._format
        del tr_server.stats.mseed
        del tr_server.stats.processing
        del tr_db.stats.processing

        np.testing.assert_allclose(tr_server.stats.delta, tr_db.stats.delta)

        assert tr_db.stats == tr_server.stats
        np.testing.assert_allclose(tr_db.data, tr_server.data)
