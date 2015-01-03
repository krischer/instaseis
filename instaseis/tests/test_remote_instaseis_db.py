#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for the instaseis remote database.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
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
