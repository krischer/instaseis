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

import instaseis
from .tornado_testing_fixtures import *  # NOQA
from .tornado_testing_fixtures import _add_callback


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
                                   atol=1E-7 * r_tr.data.ptp())


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

    kwargs = {"source": source, "receiver": receiver,
              "components": ["Z", "N", "E", "R", "T"]}
    _compare_streams(r_db, l_db, kwargs)

    # Test velocity and acceleration.
    kwargs = {"source": source, "receiver": receiver,
              "components": ["Z", "N", "E", "R", "T"], "kind": "velocity"}
    _compare_streams(r_db, l_db, kwargs)
    kwargs = {"source": source, "receiver": receiver,
              "components": ["Z", "N", "E", "R", "T"], "kind": "acceleration"}
    _compare_streams(r_db, l_db, kwargs)

    # Test remove source shift.
    kwargs = {"source": source, "receiver": receiver,
              "components": ["Z", "N", "E", "R", "T"],
              "remove_source_shift": False}
    _compare_streams(r_db, l_db, kwargs)

    # Test lanzcos resampling.
    kwargs = {"source": source, "receiver": receiver,
              "components": ["Z", "N", "E", "R", "T"],
              "dt": 1.0, "a_lanczos": 6}
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
