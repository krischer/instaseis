#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test for the Instaseis server.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
from __future__ import absolute_import

import copy
import json
import numpy as np
import instaseis
from .tornado_testing_fixtures import *  # NOQA


def _assemble_url(**kwargs):
    """
    Helper function.
    """
    url = "/seismograms_raw?"
    url += "&".join("%s=%s" % (key, value) for key, value in kwargs.items())
    return url


def test_root_route(all_clients):
    """
    Shows very basic information and the version of the client. Test is run
    for all clients.
    """
    client = all_clients
    request = client.fetch("/")
    assert request.code == 200
    result = json.loads(str(request.body.decode("utf8")))
    assert result == {
        "type": "Instaseis Remote Server", "version": instaseis.__version__}


def test_info_route(all_clients):
    """
    Tests that the /info route returns the information dictionary and does
    not mess with anything.

    Test is parameterized to run for all test databases.
    """
    client = all_clients
    # Load the result via the webclient.
    request = client.fetch("/info")
    assert request.code == 200
    result = json.loads(str(request.body.decode("utf8")))
    # Convert list to arrays.
    client_slip = np.array(result["slip"])
    client_sliprate = np.array(result["sliprate"])
    del result["slip"]
    del result["sliprate"]

    # Make sure it is identical to one from a local client.
    db = instaseis.open_db(client.filepath, read_on_demand=True)

    db_slip = list(db.info.slip)
    db_sliprate = list(db.info.sliprate)
    del db.info["slip"]
    del db.info["sliprate"]

    # Directory from server is empty.
    db.info["directory"] = ""
    # Datetime is a string.
    db.info["datetime"] = str(db.info["datetime"])

    # Make sure the information dictionary is the same, no matter where it
    # comes from.
    assert dict(db.info) == result
    # Same for slip and sliprate.
    np.testing.assert_allclose(client_slip, db_slip)
    np.testing.assert_allclose(client_sliprate, db_sliprate)


def test_raw_seismograms_error_handling(all_clients):
    """
    Tests error handling of the /seismograms_raw route. Potentially outwards
    facing thus tested rather well.
    """
    client = all_clients

    basic_parameters = {
        "source_latitude": 10,
        "source_longitude": 10,
        "receiver_latitude": -10,
        "receiver_longitude": -10}

    # Remove the source latitude, a required parameter.
    params = copy.deepcopy(basic_parameters)
    del params["source_latitude"]
    request = client.fetch(_assemble_url(**params))
    assert request.code == 400
    assert request.reason == \
        "Required parameter 'source_latitude' not given."

    # Invalid type.
    params = copy.deepcopy(basic_parameters)
    params["source_latitude"] = "A"
    request = client.fetch(_assemble_url(**params))
    assert request.code == 400
    assert "could not be converted" in request.reason
    assert "source_latitude" in request.reason

    # No source.
    request = client.fetch(_assemble_url(**basic_parameters))
    assert request.code == 400
    assert request.reason == "No/insufficient source parameters specified"

    # Invalid MT source.
    params = copy.deepcopy(basic_parameters)
    params["m_tt"] = "100000"
    params["m_pp"] = "100000"
    params["m_rr"] = "100000"
    params["m_rt"] = "100000"
    params["m_rp"] = "100000"
    params["m_tp"] = "100000"
    params["source_latitude"] = "100"
    request = client.fetch(_assemble_url(**params))
    assert request.code == 400
    assert "could not construct moment tensor source" in request.reason.lower()

    # Invalid strike/dip/rake
    params = copy.deepcopy(basic_parameters)
    params["strike"] = "45"
    params["dip"] = "45"
    params["rake"] = "45"
    params["M0"] = "450000"
    params["source_latitude"] = "100"
    request = client.fetch(_assemble_url(**params))
    assert request.code == 400
    assert "could not construct the source" in request.reason.lower()
    assert "strike/dip/rake" in request.reason.lower()

    # Invalid force source.
    params = copy.deepcopy(basic_parameters)
    params["f_r"] = "100000"
    params["f_t"] = "100000"
    params["f_p"] = "100000"
    params["source_latitude"] = "100"
    request = client.fetch(_assemble_url(**params))
    assert request.code == 400
    assert "could not construct force source" in request.reason.lower()

    # Could not extract seismogram.
    params = copy.deepcopy(basic_parameters)
    params["m_tt"] = "100000"
    params["m_pp"] = "100000"
    params["m_rr"] = "100000"
    params["m_rt"] = "100000"
    params["m_rp"] = "100000"
    params["m_tp"] = "100000"
    params["components"] = "ABC"
    request = client.fetch(_assemble_url(**params))
    assert request.code == 400
    assert "could not extract seismogram" in request.reason.lower()
