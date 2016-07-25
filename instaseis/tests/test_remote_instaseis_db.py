#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for the instaseis remote database.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
:license:
    GNU Lesser General Public License, Version 3 [non-commercial/academic use]
    (http://www.gnu.org/copyleft/lgpl.html)
"""
from __future__ import absolute_import

import copy
import numpy as np
import responses
import warnings
import pytest

import instaseis
from .tornado_testing_fixtures import *  # NOQA
from .tornado_testing_fixtures import _add_callback

# Conditionally import mock either from the stdlib or as a separate library.
import sys
if sys.version_info[0] == 2:  # pragma: no cover
    import mock
else:  # pragma: no cover
    import unittest.mock as mock


@responses.activate
def test_info(all_remote_dbs):
    """
    Make sure the info is identical no matter if it comes from a local or
    from a remote database.
    """
    # Remote and local database.
    r_db = all_remote_dbs
    l_db = instaseis.open_db(r_db._client.filepath)
    # Mock responses to get the tornado testing to work.
    _add_callback(r_db._client)

    r_info = copy.deepcopy(r_db.info)
    l_info = copy.deepcopy(l_db.info)

    np.testing.assert_allclose(r_info.slip, l_info.slip)
    np.testing.assert_allclose(r_info.sliprate, l_info.sliprate)

    for key in ["directory", "slip", "sliprate"]:
        del r_info[key]
        del l_info[key]

    assert r_info.__dict__ == l_info.__dict__


def _compare_streams(r_db, l_db, kwargs):
    """
    Helper function comparing streams extracted from local and remote
    instaseis databases.
    """
    r_st = r_db.get_seismograms(**kwargs)
    l_st = l_db.get_seismograms(**kwargs)

    assert len(r_st) == len(l_st)

    for r_tr, l_tr in zip(r_st, l_st):
        assert r_tr.stats.__dict__ == l_tr.stats.__dict__
        # Very small values have issues with floating point accuracy. 7
        # orders of magnitude should be more than accurate enough.
        np.testing.assert_allclose(r_tr.data, l_tr.data,
                                   atol=1E-6 * r_tr.data.ptp())


@responses.activate
def test_seismogram_extraction(all_remote_dbs):
    """
    Test the seismogram extraction from local and remote databases.
    """
    # Remote and local database.
    r_db = all_remote_dbs
    l_db = instaseis.open_db(r_db._client.filepath)
    # Mock responses to get the tornado testing to work.
    _add_callback(r_db._client)

    source = instaseis.Source(
        latitude=4., longitude=3.0, depth_in_m=0, m_rr=4.71e+17, m_tt=3.81e+17,
        m_pp=-4.74e+17, m_rt=3.99e+17, m_rp=-8.05e+17, m_tp=-1.23e+17)

    receiver = instaseis.Receiver(latitude=10., longitude=20., depth_in_m=None)

    components = r_db.available_components

    kwargs = {"source": source, "receiver": receiver,
              "components": components}
    _compare_streams(r_db, l_db, kwargs)

    # Test velocity and acceleration.
    kwargs = {"source": source, "receiver": receiver,
              "components": components, "kind": "velocity"}
    _compare_streams(r_db, l_db, kwargs)
    kwargs = {"source": source, "receiver": receiver,
              "components": components, "kind": "acceleration"}
    _compare_streams(r_db, l_db, kwargs)

    # Test remove source shift.
    kwargs = {"source": source, "receiver": receiver,
              "components": components,
              "remove_source_shift": False}
    _compare_streams(r_db, l_db, kwargs)

    # Test resampling.
    kwargs = {"source": source, "receiver": receiver,
              "components": components,
              "dt": 1.0, "kernelwidth": 6}
    _compare_streams(r_db, l_db, kwargs)

    # Test force source.
    if "displ_only" in r_db._client.filepath:
        source = instaseis.ForceSource(
            latitude=89.91, longitude=0.0, depth_in_m=12000,
            f_r=1.23E10,
            f_t=2.55E10,
            f_p=1.73E10)
        kwargs = {"source": source, "receiver": receiver}
        _compare_streams(r_db, l_db, kwargs)

    # Fix receiver depth, network, and station codes.
    receiver = instaseis.Receiver(latitude=10., longitude=20.,
                                  depth_in_m=0.0, station="ALTM",
                                  network="BW")

    kwargs = {"source": source, "receiver": receiver,
              "components": components}
    _compare_streams(r_db, l_db, kwargs)


def test_initialization_failures():
    """
    Tests various initialization failures for the remote instaseis db.
    """
    # Random error during init.
    with mock.patch("instaseis.database_interfaces.remote_instaseis_db"
                    ".RemoteInstaseisDB._download_url") as p:
        p.side_effect = ValueError("random")
        with pytest.raises(instaseis.InstaseisError) as err:
            instaseis.open_db("http://localhost:8765432")

    assert err.value.args[0] == ("Failed to connect to remote Instaseis "
                                 "server due to: random")

    # Invalid JSON returned.
    with mock.patch("instaseis.database_interfaces.remote_instaseis_db"
                    ".RemoteInstaseisDB._download_url") as p:
        p.return_value = {"a": "b"}
        with pytest.raises(instaseis.InstaseisError) as err:
            instaseis.open_db("http://localhost:8765432")

    assert err.value.args[0].startswith("Instaseis server responded with "
                                        "invalid response:")

    # Incompatible version number - should raise a warning.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with mock.patch("instaseis.database_interfaces.remote_instaseis_db"
                        ".RemoteInstaseisDB._download_url") as p_1:
            p_1.return_value = {"type": "Instaseis Remote Server",
                                "datetime": "2001-01-01",
                                "version": "test version"}
            try:
                instaseis.open_db("http://localhost:8765432")
            except:
                pass

    assert len(w) == 1
    assert w[0].message.args[0].startswith('Instaseis versions on server')


@responses.activate
def test_source_depth_error_handling(all_remote_dbs):
    """
    Test the seismogram extraction from local and remote databases.
    """
    db = all_remote_dbs

    # Skip forward databases.
    if "100s_db_fwd" in db._client.filepath:
        return

    # Mock responses to get the tornado testing to work.
    _add_callback(db._client)

    # 900 km is deeper than any test database.
    src = instaseis.Source(latitude=4., longitude=3.0, depth_in_m=900000,
                           m_rr=4.71e+17, m_tt=3.81e+17, m_pp=-4.74e+17,
                           m_rt=3.99e+17, m_rp=-8.05e+17, m_tp=-1.23e+17)
    rec = instaseis.Receiver(latitude=10., longitude=20., depth_in_m=0)

    with pytest.raises(ValueError) as err:
        db.get_seismograms(source=src, receiver=rec)

    assert err.value.args[0] == (
        "Source too deep. Source would be located at a radius of 5471000.0 "
        "meters. The database supports source radii from 6000000.0 to "
        "6371000.0 meters.")

    # Too shallow.
    src = instaseis.Source(latitude=4., longitude=3.0, depth_in_m=-10000,
                           m_rr=4.71e+17, m_tt=3.81e+17, m_pp=-4.74e+17,
                           m_rt=3.99e+17, m_rp=-8.05e+17, m_tp=-1.23e+17)
    rec = instaseis.Receiver(latitude=10., longitude=20., depth_in_m=0)

    with pytest.raises(ValueError) as err:
        db.get_seismograms(source=src, receiver=rec)

    assert err.value.args[0] == (
        "Source is too shallow. Source would be located at a radius of "
        "6381000.0 meters. The database supports source radii from "
        "6000000.0 to 6371000.0 meters.")
