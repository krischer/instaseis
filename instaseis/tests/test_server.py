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

import json
import numpy as np
import instaseis
from .tornado_testing_fixtures import *  # NOQA


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
