#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Basic integration tests for the AxiSEM database Python interface.

:copyright:
    Martin van Driel (Martin@vanDriel.de), 2014
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
:license:
    GNU Lesser General Public License, Version 3 [non-commercial/academic use]
    (http://www.gnu.org/copyleft/lgpl.html)
"""
from __future__ import absolute_import

import inspect
import io
import math
import numpy as np
import obspy
import os
import pytest
import shutil

import instaseis
from instaseis import InstaseisError, InstaseisNotFoundError
from instaseis.database_interfaces import find_and_open_files
from instaseis.database_interfaces.base_instaseis_db import \
    _get_seismogram_times
from instaseis import Source, Receiver, ForceSource
from instaseis.helpers import (get_band_code, elliptic_to_geocentric_latitude,
                               geocentric_to_elliptic_latitude, sizeof_fmt)

from .testdata import BWD_TEST_DATA, FWD_TEST_DATA
from .testdata import BWD_STRAIN_ONLY_TEST_DATA, BWD_FORCE_TEST_DATA


# Most generic way to get the data folder path.
DATA = os.path.join(os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe()))), "data")

DBS = [os.path.join(DATA, "100s_db_fwd"),
       os.path.join(DATA, "100s_db_fwd_deep"),
       os.path.join(DATA, "100s_db_bwd_displ_only")]

TEST_DATA = {
    os.path.join(DATA, "100s_db_fwd"): FWD_TEST_DATA,
    os.path.join(DATA, "100s_db_bwd_displ_only"): BWD_TEST_DATA
}


# Add all automatically created repacked databases to the test suite.
for name, path in pytest.config.dbs["databases"].items():
    DBS.append(path)
    if "bwd" in name:
        test_data = BWD_TEST_DATA
    elif "horizontal_only" in name or "vertical_only" in name:
        test_data = BWD_TEST_DATA
    elif "fwd" in name:
        test_data = FWD_TEST_DATA
    else:  # pragma: no cover
        raise NotImplementedError
    TEST_DATA[path] = test_data

BW_DISPL_DBS = [_i for _i in DBS if "_db_bwd_displ_" in _i]


@pytest.mark.parametrize("bwd_db", BW_DISPL_DBS)
def test_fwd_vs_bwd(bwd_db):
    """
    Test fwd against bwd mode
    """
    instaseis_fwd = find_and_open_files(os.path.join(DATA, "100s_db_fwd"))
    instaseis_bwd = find_and_open_files(bwd_db)

    source_fwd = Source(latitude=4., longitude=3.0, depth_in_m=None,
                        m_rr=4.71e+17, m_tt=3.81e+17, m_pp=-4.74e+17,
                        m_rt=3.99e+17, m_rp=-8.05e+17, m_tp=-1.23e+17)
    source_bwd = Source(latitude=4., longitude=3.0, depth_in_m=0,
                        m_rr=4.71e+17, m_tt=3.81e+17, m_pp=-4.74e+17,
                        m_rt=3.99e+17, m_rp=-8.05e+17, m_tp=-1.23e+17)

    receiver_fwd = Receiver(latitude=10., longitude=20., depth_in_m=0)
    receiver_bwd = Receiver(latitude=10., longitude=20., depth_in_m=None)

    st_fwd = instaseis_fwd.get_seismograms(
        source=source_fwd, receiver=receiver_fwd,
        components=('Z', 'N', 'E', 'R', 'T'))
    st_bwd = instaseis_bwd.get_seismograms(
        source=source_bwd, receiver=receiver_bwd,
        components=('Z', 'N', 'E', 'R', 'T'))

    st_bwd.filter('lowpass', freq=0.002)
    st_fwd.filter('lowpass', freq=0.002)

    np.testing.assert_allclose(st_fwd.select(component="Z")[0].data,
                               st_bwd.select(component="Z")[0].data,
                               rtol=1E-3, atol=1E-10)

    np.testing.assert_allclose(st_fwd.select(component="N")[0].data,
                               st_bwd.select(component="N")[0].data,
                               rtol=1E-3, atol=1E-10)

    np.testing.assert_allclose(st_fwd.select(component="E")[0].data,
                               st_bwd.select(component="E")[0].data,
                               rtol=1E-3, atol=1E-10)

    np.testing.assert_allclose(st_fwd.select(component="R")[0].data,
                               st_bwd.select(component="R")[0].data,
                               rtol=1E-3, atol=1E-10)

    np.testing.assert_allclose(st_fwd.select(component="T")[0].data,
                               st_bwd.select(component="T")[0].data,
                               rtol=1E-3, atol=1E-10)


@pytest.mark.parametrize("bwd_db", BW_DISPL_DBS)
def test_fwd_vs_bwd_axial(bwd_db):
    """
    Test fwd against bwd mode, axial element. Differences are a bit larger then
    in non axial case, presumably because the close source, which is not
    exactly a point source in the SEM representation.
    """
    instaseis_fwd = find_and_open_files(os.path.join(DATA, "100s_db_fwd_deep"))
    instaseis_bwd = find_and_open_files(bwd_db)

    source_fwd = Source(latitude=0., longitude=0., depth_in_m=None,
                        m_rr=4.71e+17, m_tt=3.81e+17, m_pp=-4.74e+17,
                        m_rt=3.99e+17, m_rp=-8.05e+17, m_tp=-1.23e+17)
    source_bwd = Source(latitude=0., longitude=0., depth_in_m=310000,
                        m_rr=4.71e+17, m_tt=3.81e+17, m_pp=-4.74e+17,
                        m_rt=3.99e+17, m_rp=-8.05e+17, m_tp=-1.23e+17)

    receiver_fwd = Receiver(latitude=0., longitude=0.1, depth_in_m=0)
    receiver_bwd = Receiver(latitude=0., longitude=0.1, depth_in_m=None)

    st_fwd = instaseis_fwd.get_seismograms(
        source=source_fwd, receiver=receiver_fwd,
        components=('Z', 'N', 'E', 'R', 'T'))
    st_bwd = instaseis_bwd.get_seismograms(
        source=source_bwd, receiver=receiver_bwd,
        components=('Z', 'N', 'E', 'R', 'T'))

    st_bwd.filter('lowpass', freq=0.01)
    st_fwd.filter('lowpass', freq=0.01)
    st_bwd.filter('lowpass', freq=0.01)
    st_fwd.filter('lowpass', freq=0.01)
    st_bwd.differentiate()
    st_fwd.differentiate()

    np.testing.assert_allclose(st_fwd.select(component="Z")[0].data,
                               st_bwd.select(component="Z")[0].data,
                               rtol=1E-2, atol=5E-9)

    np.testing.assert_allclose(st_fwd.select(component="N")[0].data,
                               st_bwd.select(component="N")[0].data,
                               rtol=1E-2, atol=5E-9)

    np.testing.assert_allclose(st_fwd.select(component="E")[0].data,
                               st_bwd.select(component="E")[0].data,
                               rtol=1E-2, atol=6E-9)

    np.testing.assert_allclose(st_fwd.select(component="R")[0].data,
                               st_bwd.select(component="R")[0].data,
                               rtol=1E-2, atol=6E-9)

    np.testing.assert_allclose(st_fwd.select(component="T")[0].data,
                               st_bwd.select(component="T")[0].data,
                               rtol=1E-2, atol=5E-9)


@pytest.mark.parametrize("bwd_db", BW_DISPL_DBS)
def test_incremental_bwd(bwd_db):
    """
    incremental tests of bwd mode with displ_only db
    """
    instaseis_bwd = find_and_open_files(bwd_db)

    receiver = Receiver(latitude=42.6390, longitude=74.4940)
    source = Source(
        latitude=89.91, longitude=0.0, depth_in_m=12000,
        m_rr=4.710000e+24 / 1E7,
        m_tt=3.810000e+22 / 1E7,
        m_pp=-4.740000e+24 / 1E7,
        m_rt=3.990000e+23 / 1E7,
        m_rp=-8.050000e+23 / 1E7,
        m_tp=-1.230000e+24 / 1E7)

    st_bwd = instaseis_bwd.get_seismograms(
        source=source, receiver=receiver, components=('Z', 'N', 'E', 'R', 'T'))

    np.testing.assert_allclose(st_bwd.select(component='Z')[0].data,
                               BWD_TEST_DATA["Z"], rtol=1E-7, atol=1E-12)
    np.testing.assert_allclose(st_bwd.select(component='N')[0].data,
                               BWD_TEST_DATA["N"], rtol=1E-7, atol=1E-12)
    np.testing.assert_allclose(st_bwd.select(component='E')[0].data,
                               BWD_TEST_DATA["E"], rtol=1E-7, atol=1E-12)
    np.testing.assert_allclose(st_bwd.select(component='R')[0].data,
                               BWD_TEST_DATA["R"], rtol=1E-7, atol=1E-12)
    np.testing.assert_allclose(st_bwd.select(component='T')[0].data,
                               BWD_TEST_DATA["T"], rtol=1E-7, atol=1E-12)
    if hasattr(instaseis_bwd.meshes, "px"):
        assert instaseis_bwd.meshes.px.strain_buffer.efficiency == 0.0
        assert instaseis_bwd.meshes.pz.strain_buffer.efficiency == 0.0
    else:
        assert instaseis_bwd.meshes.merged.strain_buffer.efficiency == 0.0

    # read on init
    instaseis_bwd = find_and_open_files(bwd_db, read_on_demand=False)

    st_bwd = instaseis_bwd.get_seismograms(
        source=source, receiver=receiver, components=('Z', 'N', 'E', 'R', 'T'))

    np.testing.assert_allclose(st_bwd.select(component='Z')[0].data,
                               BWD_TEST_DATA["Z"], rtol=1E-7, atol=1E-12)
    np.testing.assert_allclose(st_bwd.select(component='N')[0].data,
                               BWD_TEST_DATA["N"], rtol=1E-7, atol=1E-12)
    np.testing.assert_allclose(st_bwd.select(component='E')[0].data,
                               BWD_TEST_DATA["E"], rtol=1E-7, atol=1E-12)
    np.testing.assert_allclose(st_bwd.select(component='R')[0].data,
                               BWD_TEST_DATA["R"], rtol=1E-7, atol=1E-12)
    np.testing.assert_allclose(st_bwd.select(component='T')[0].data,
                               BWD_TEST_DATA["T"], rtol=1E-7, atol=1E-12)
    if hasattr(instaseis_bwd.meshes, "px"):
        assert instaseis_bwd.meshes.px.strain_buffer.efficiency == 0.0
        assert instaseis_bwd.meshes.pz.strain_buffer.efficiency == 0.0
    else:
        assert instaseis_bwd.meshes.merged.strain_buffer.efficiency == 0.0

    # read the same again to test buffer
    st_bwd = instaseis_bwd.get_seismograms(
        source=source, receiver=receiver, components=('Z', 'N', 'E', 'R', 'T'))
    np.testing.assert_allclose(st_bwd.select(component='Z')[0].data,
                               BWD_TEST_DATA["Z"], rtol=1E-7, atol=1E-12)
    np.testing.assert_allclose(st_bwd.select(component='N')[0].data,
                               BWD_TEST_DATA["N"], rtol=1E-7, atol=1E-12)
    np.testing.assert_allclose(st_bwd.select(component='E')[0].data,
                               BWD_TEST_DATA["E"], rtol=1E-7, atol=1E-12)
    np.testing.assert_allclose(st_bwd.select(component='R')[0].data,
                               BWD_TEST_DATA["R"], rtol=1E-7, atol=1E-12)
    np.testing.assert_allclose(st_bwd.select(component='T')[0].data,
                               BWD_TEST_DATA["T"], rtol=1E-7, atol=1E-12)
    if hasattr(instaseis_bwd.meshes, "px"):
        assert instaseis_bwd.meshes.px.strain_buffer.efficiency == 1.0 / 2.0
        assert instaseis_bwd.meshes.pz.strain_buffer.efficiency == 1.0 / 2.0
    else:
        assert instaseis_bwd.meshes.merged.strain_buffer.efficiency == \
            1.0 / 2.0

    # test resampling with a no-op interpolation.
    dt = instaseis_bwd.info.dt
    st_bwd = instaseis_bwd.get_seismograms(
        source=source, receiver=receiver, components=('Z'), dt=dt,
        kernelwidth=5)

    np.testing.assert_allclose(st_bwd.select(component='Z')[0].data,
                               BWD_TEST_DATA["Z"], rtol=1E-7, atol=1E-12)


def test_vertical_only_db(tmpdir):
    """
    Everything should work even if only the vertical component is present.
    """
    # Copy only the vertical component data.
    tmpdir = str(tmpdir)
    path = os.path.join(tmpdir, "PZ", "Data", "ordered_output.nc4")
    os.makedirs(os.path.dirname(path))
    shutil.copy(
        os.path.join(DATA, "100s_db_bwd_displ_only", "PZ", "Data",
                     "ordered_output.nc4"), path)

    receiver = Receiver(latitude=42.6390, longitude=74.4940)
    source = Source(
        latitude=89.91, longitude=0.0, depth_in_m=12000,
        m_rr=4.710000e+24 / 1E7,
        m_tt=3.810000e+22 / 1E7,
        m_pp=-4.740000e+24 / 1E7,
        m_rt=3.990000e+23 / 1E7,
        m_rp=-8.050000e+23 / 1E7,
        m_tp=-1.230000e+24 / 1E7)

    # vertical only DB
    instaseis_bwd = find_and_open_files(tmpdir)

    st_bwd = instaseis_bwd.get_seismograms(
        source=source, receiver=receiver, components=('Z'))

    np.testing.assert_allclose(st_bwd.select(component='Z')[0].data,
                               BWD_TEST_DATA["Z"], rtol=1E-7, atol=1E-12)


def test_horizontal_only_db(tmpdir):
    """
    Everything should work even if only the horizontal component is present.
    """
    # Copy only the horizontal component data.
    tmpdir = str(tmpdir)
    path = os.path.join(tmpdir, "PX", "Data", "ordered_output.nc4")
    os.makedirs(os.path.dirname(path))
    shutil.copy(
        os.path.join(DATA, "100s_db_bwd_displ_only", "PX", "Data",
                     "ordered_output.nc4"),
        path)

    receiver = Receiver(latitude=42.6390, longitude=74.4940)
    source = Source(
        latitude=89.91, longitude=0.0, depth_in_m=12000,
        m_rr=4.710000e+24 / 1E7,
        m_tt=3.810000e+22 / 1E7,
        m_pp=-4.740000e+24 / 1E7,
        m_rt=3.990000e+23 / 1E7,
        m_rp=-8.050000e+23 / 1E7,
        m_tp=-1.230000e+24 / 1E7)

    # vertical only DB
    instaseis_bwd = find_and_open_files(tmpdir)

    st_bwd = instaseis_bwd.get_seismograms(
        source=source, receiver=receiver, components=('N'))

    np.testing.assert_allclose(st_bwd.select(component='N')[0].data,
                               BWD_TEST_DATA["N"], rtol=1E-7, atol=1E-12)


def test_requesting_wrong_component_horizontal(tmpdir):
    # Copy only the horizontal component data.
    tmpdir = str(tmpdir)
    path = os.path.join(tmpdir, "PX", "Data", "ordered_output.nc4")
    os.makedirs(os.path.dirname(path))
    shutil.copy(
        os.path.join(DATA, "100s_db_bwd_displ_only", "PX", "Data",
                     "ordered_output.nc4"),
        path)

    receiver = Receiver(latitude=42.6390, longitude=74.4940)
    source = Source(
        latitude=89.91, longitude=0.0, depth_in_m=12000,
        m_rr=4.710000e+24 / 1E7,
        m_tt=3.810000e+22 / 1E7,
        m_pp=-4.740000e+24 / 1E7,
        m_rt=3.990000e+23 / 1E7,
        m_rp=-8.050000e+23 / 1E7,
        m_tp=-1.230000e+24 / 1E7)

    # vertical only DB
    instaseis_bwd = find_and_open_files(tmpdir)

    with pytest.raises(ValueError):
        instaseis_bwd.get_seismograms(
            source=source, receiver=receiver, components=('Z'))


def test_requesting_wrong_component_vertical(tmpdir):
    # Copy only the horizontal component data.
    tmpdir = str(tmpdir)
    path = os.path.join(tmpdir, "PZ", "Data", "ordered_output.nc4")
    os.makedirs(os.path.dirname(path))
    shutil.copy(
        os.path.join(DATA, "100s_db_bwd_displ_only", "PZ", "Data",
                     "ordered_output.nc4"),
        path)

    receiver = Receiver(latitude=42.6390, longitude=74.4940)
    source = Source(
        latitude=89.91, longitude=0.0, depth_in_m=12000,
        m_rr=4.710000e+24 / 1E7,
        m_tt=3.810000e+22 / 1E7,
        m_pp=-4.740000e+24 / 1E7,
        m_rt=3.990000e+23 / 1E7,
        m_rp=-8.050000e+23 / 1E7,
        m_tp=-1.230000e+24 / 1E7)

    # vertical only DB
    instaseis_bwd = find_and_open_files(tmpdir)

    with pytest.raises(ValueError):
        instaseis_bwd.get_seismograms(
            source=source, receiver=receiver, components=('E'))
    with pytest.raises(ValueError):
        instaseis_bwd.get_seismograms(
            source=source, receiver=receiver, components=('N'))
    with pytest.raises(ValueError):
        instaseis_bwd.get_seismograms(
            source=source, receiver=receiver, components=('T'))
    with pytest.raises(ValueError):
        instaseis_bwd.get_seismograms(
            source=source, receiver=receiver, components=('R'))


def test_incremental_fwd():
    """
    incremental tests of fwd mode
    """
    instaseis_fwd = find_and_open_files(os.path.join(DATA, "100s_db_fwd"))

    receiver = Receiver(latitude=42.6390, longitude=74.4940)
    source = Source(
        latitude=89.91, longitude=0.0, depth_in_m=None,
        m_rr=4.710000e+24 / 1E7,
        m_tt=3.810000e+22 / 1E7,
        m_pp=-4.740000e+24 / 1E7,
        m_rt=3.990000e+23 / 1E7,
        m_rp=-8.050000e+23 / 1E7,
        m_tp=-1.230000e+24 / 1E7)

    st_fwd = instaseis_fwd.get_seismograms(
        source=source, receiver=receiver, components=('Z', 'N', 'E', 'R', 'T'))

    np.testing.assert_allclose(st_fwd.select(component='Z')[0].data,
                               FWD_TEST_DATA["Z"], rtol=1E-7, atol=1E-16)
    np.testing.assert_allclose(st_fwd.select(component='N')[0].data,
                               FWD_TEST_DATA["N"], rtol=1E-7, atol=1E-16)
    np.testing.assert_allclose(st_fwd.select(component='E')[0].data,
                               FWD_TEST_DATA["E"], rtol=1E-7, atol=1E-16)
    np.testing.assert_allclose(st_fwd.select(component='R')[0].data,
                               FWD_TEST_DATA["R"], rtol=1E-7, atol=1E-16)
    np.testing.assert_allclose(st_fwd.select(component='T')[0].data,
                               FWD_TEST_DATA["T"], rtol=1E-7, atol=1E-16)
    assert instaseis_fwd.meshes.m1.displ_buffer.efficiency == 0.0
    assert instaseis_fwd.meshes.m2.displ_buffer.efficiency == 0.0
    assert instaseis_fwd.meshes.m3.displ_buffer.efficiency == 0.0
    assert instaseis_fwd.meshes.m4.displ_buffer.efficiency == 0.0

    # read the same again to test buffer
    st_fwd = instaseis_fwd.get_seismograms(
        source=source, receiver=receiver, components=('Z', 'N', 'E', 'R', 'T'))
    np.testing.assert_allclose(st_fwd.select(component='Z')[0].data,
                               FWD_TEST_DATA["Z"], rtol=1E-7, atol=1E-16)
    np.testing.assert_allclose(st_fwd.select(component='N')[0].data,
                               FWD_TEST_DATA["N"], rtol=1E-7, atol=1E-16)
    np.testing.assert_allclose(st_fwd.select(component='E')[0].data,
                               FWD_TEST_DATA["E"], rtol=1E-7, atol=1E-16)
    np.testing.assert_allclose(st_fwd.select(component='R')[0].data,
                               FWD_TEST_DATA["R"], rtol=1E-7, atol=1E-16)
    np.testing.assert_allclose(st_fwd.select(component='T')[0].data,
                               FWD_TEST_DATA["T"], rtol=1E-7, atol=1E-16)
    assert instaseis_fwd.meshes.m1.displ_buffer.efficiency == 1.0 / 2.0
    assert instaseis_fwd.meshes.m2.displ_buffer.efficiency == 1.0 / 2.0
    assert instaseis_fwd.meshes.m3.displ_buffer.efficiency == 1.0 / 2.0
    assert instaseis_fwd.meshes.m4.displ_buffer.efficiency == 1.0 / 2.0


def test_incremental_bwd_strain_only():
    """
    incremental tests of bwd mode with strain_only DB
    """
    instaseis_bwd = find_and_open_files(
        os.path.join(DATA, "100s_db_bwd_strain_only"))

    receiver = Receiver(latitude=42.6390, longitude=74.4940)
    source = Source(
        latitude=89.91, longitude=0.0, depth_in_m=12000,
        m_rr=4.710000e+24 / 1E7,
        m_tt=3.810000e+22 / 1E7,
        m_pp=-4.740000e+24 / 1E7,
        m_rt=3.990000e+23 / 1E7,
        m_rp=-8.050000e+23 / 1E7,
        m_tp=-1.230000e+24 / 1E7)

    st_bwd = instaseis_bwd.get_seismograms(
        source=source, receiver=receiver, components=('Z', 'N', 'E', 'R', 'T'))

    np.testing.assert_allclose(st_bwd.select(component='Z')[0].data,
                               BWD_STRAIN_ONLY_TEST_DATA["Z"], rtol=1E-7,
                               atol=1E-12)
    np.testing.assert_allclose(st_bwd.select(component='N')[0].data,
                               BWD_STRAIN_ONLY_TEST_DATA["N"], rtol=1E-7,
                               atol=1E-12)
    np.testing.assert_allclose(st_bwd.select(component='E')[0].data,
                               BWD_STRAIN_ONLY_TEST_DATA["E"], rtol=1E-7,
                               atol=1E-12)
    np.testing.assert_allclose(st_bwd.select(component='R')[0].data,
                               BWD_STRAIN_ONLY_TEST_DATA["R"], rtol=1E-7,
                               atol=1E-12)
    np.testing.assert_allclose(st_bwd.select(component='T')[0].data,
                               BWD_STRAIN_ONLY_TEST_DATA["T"], rtol=1E-7,
                               atol=1E-12)


@pytest.mark.parametrize("db", BW_DISPL_DBS)
def test_incremental_bwd_force_source(db):
    """
    incremental tests of bwd mode with source force
    """
    instaseis_bwd = find_and_open_files(db)

    receiver = Receiver(latitude=42.6390, longitude=74.4940)
    source = ForceSource(
        latitude=89.91, longitude=0.0, depth_in_m=12000,
        f_r=1.23E10,
        f_t=2.55E10,
        f_p=1.73E10)

    st_bwd = instaseis_bwd.get_seismograms(
        source=source, receiver=receiver, components=('Z', 'N', 'E', 'R', 'T'),
        kind='velocity')

    np.testing.assert_allclose(st_bwd.select(component='Z')[0].data,
                               BWD_FORCE_TEST_DATA["Z"], rtol=1E-7, atol=1E-12)
    np.testing.assert_allclose(st_bwd.select(component='N')[0].data,
                               BWD_FORCE_TEST_DATA["N"], rtol=1E-7, atol=1E-12)
    np.testing.assert_allclose(st_bwd.select(component='E')[0].data,
                               BWD_FORCE_TEST_DATA["E"], rtol=1E-7, atol=1E-12)
    np.testing.assert_allclose(st_bwd.select(component='R')[0].data,
                               BWD_FORCE_TEST_DATA["R"], rtol=1E-7, atol=1E-12)
    np.testing.assert_allclose(st_bwd.select(component='T')[0].data,
                               BWD_FORCE_TEST_DATA["T"], rtol=1E-7, atol=1E-12)

    # Force source does not work with strain databases.
    db_strain = find_and_open_files(
        os.path.join(DATA, "100s_db_bwd_strain_only"))

    receiver = Receiver(latitude=42.6390, longitude=74.4940)
    source = ForceSource(
        latitude=89.91, longitude=0.0, depth_in_m=12000,
        f_r=1.23E10,
        f_t=2.55E10,
        f_p=1.73E10)

    with pytest.raises(ValueError) as err:
        db_strain.get_seismograms(source=source, receiver=receiver)

    assert err.value.args[0] == "Force sources only in displ_only mode"

    # Force source, extract single component.
    for comp in ["Z", "N", "E", "R", "T"]:
        st = instaseis_bwd.get_seismograms(
            source=source, receiver=receiver, components=[comp])
        assert [tr.stats.channel[-1] for tr in st] == [comp]


@pytest.mark.parametrize("bwd_db", BW_DISPL_DBS)
def test_get_greens_vs_get_seismogram(bwd_db):
    """
    Test get_greens_function() against default get_seismograms()
    """
    db = find_and_open_files(bwd_db)

    depth_in_m = 1000
    epicentral_distance_degree = 20
    azimuth = 177.

    # some 'random' moment tensor
    Mxx, Myy, Mzz, Mxz, Myz, Mxy = 1e20, 0.5e20, 0.3e20, 0.7e20, 0.4e20, 0.6e20

    # in Minson & Dreger, 2008, z is up, so we have
    # Mtt = Mxx, Mpp = Myy, Mrr = Mzz
    # Mrp = Myz, Mrt = Mxz, Mtp = Mxy
    source = Source(latitude=90., longitude=0., depth_in_m=depth_in_m,
                    m_tt=Mxx, m_pp=Myy, m_rr=Mzz, m_rt=Mxz, m_rp=Myz, m_tp=Mxy)
    receiver = Receiver(latitude=90. - epicentral_distance_degree,
                        longitude=azimuth)

    st_ref = db.get_seismograms(source, receiver, components=('Z', 'R', 'T'))

    st_greens = db.get_greens_function(epicentral_distance_degree,
                                       depth_in_m, definition="seiscomp")

    TSS = st_greens.select(channel="TSS")[0].data
    ZSS = st_greens.select(channel="ZSS")[0].data
    RSS = st_greens.select(channel="RSS")[0].data
    TDS = st_greens.select(channel="TDS")[0].data
    ZDS = st_greens.select(channel="ZDS")[0].data
    RDS = st_greens.select(channel="RDS")[0].data
    ZDD = st_greens.select(channel="ZDD")[0].data
    RDD = st_greens.select(channel="RDD")[0].data
    ZEP = st_greens.select(channel="ZEP")[0].data
    REP = st_greens.select(channel="REP")[0].data

    az = np.deg2rad(azimuth)
    # eq (6) in Minson & Dreger, 2008
    uz = Mxx * (ZSS / 2. * np.cos(2 * az) - ZDD / 6. + ZEP / 3.) \
        + Myy * (-ZSS / 2. * np.cos(2 * az) - ZDD / 6. + ZEP / 3.) \
        + Mzz * (ZDD / 3. + ZEP / 3.) \
        + Mxy * ZSS * np.sin(2 * az) \
        + Mxz * ZDS * np.cos(az) \
        + Myz * ZDS * np.sin(az)

    # eq (7) in Minson & Dreger, 2008
    ur = Mxx * (RSS / 2. * np.cos(2 * az) - RDD / 6. + REP / 3.) \
        + Myy * (-RSS / 2. * np.cos(2 * az) - RDD / 6. + REP / 3.) \
        + Mzz * (RDD / 3. + REP / 3.) \
        + Mxy * RSS * np.sin(2 * az) \
        + Mxz * RDS * np.cos(az) \
        + Myz * RDS * np.sin(az)

    # eq (8) in Minson & Dreger, 2008
    ut = Mxx * TSS / 2. * np.sin(2 * az) \
        - Myy * TSS / 2. * np.sin(2 * az) \
        - Mxy * TSS * np.cos(2 * az) \
        + Mxz * TDS * np.sin(az) \
        - Myz * TDS * np.cos(az)

    np.testing.assert_allclose(st_ref.select(component="Z")[0].data,
                               uz, rtol=1E-3, atol=1E-10)

    np.testing.assert_allclose(st_ref.select(component="R")[0].data,
                               ur, rtol=1E-3, atol=1E-10)

    np.testing.assert_allclose(st_ref.select(component="T")[0].data,
                               ut, rtol=1E-3, atol=1E-10)

    # Assure it also works with just the data and not necessarily an ObsPy
    # Stream object.
    greens_data = db.get_greens_function(epicentral_distance_degree,
                                         depth_in_m, definition="seiscomp",
                                         return_obspy_stream=False)
    assert isinstance(greens_data, dict)


@pytest.mark.parametrize("bwd_db", BW_DISPL_DBS)
def test_greens_function_failures(bwd_db):
    """
    Tests some failures for the greens function calculation.
    """
    db = find_and_open_files(bwd_db)

    depth_in_m = 1000
    epicentral_distance_degree = 20.0

    # Wrong kind.
    with pytest.raises(ValueError) as err:
        db.get_greens_function(epicentral_distance_degree, depth_in_m,
                               definition="seiscomp", kind="random")
    assert err.value.args[0] == "unknown kind 'random'."

    # Wrong epicentral distance
    with pytest.raises(ValueError) as err:
        db.get_greens_function(1000.0, depth_in_m, definition="seiscomp")
    assert err.value.args[0] == ("epicentral_distance_degree should be in "
                                 "[0.0, 180.0]")

    # Source depth has to be positive.
    with pytest.raises(ValueError) as err:
        db.get_greens_function(100.0, -20, definition="seiscomp")
    assert err.value.args[0] == (
        "Source is too shallow. Source would be located at a radius of "
        "6371020.0 meters. The database supports source radii from 6000000.0 "
        "to 6371000.0 meters.")

    # Requires a reciprocal database.
    db.info.is_reciprocal = False
    with pytest.raises(ValueError) as err:
        db.get_greens_function(epicentral_distance_degree, depth_in_m,
                               definition="seiscomp")
    assert err.value.args[0] == ("forward DB cannot be used with "
                                 "get_greens_function()")

    # Requires both components
    db.info.is_reciprocal = True
    db.info.components = "vertical"
    with pytest.raises(ValueError) as err:
        db.get_greens_function(epicentral_distance_degree, depth_in_m,
                               definition="seiscomp")
    assert err.value.args[0] == ("get_greens_function() needs a DB with both "
                                 "vertical and horizontal components")


@pytest.mark.parametrize("bwd_db", BW_DISPL_DBS)
def test_finite_source(bwd_db):
    """
    incremental tests of bwd mode with source force
    """
    from obspy.signal.filter import lowpass
    instaseis_bwd = find_and_open_files(bwd_db)

    receiver = Receiver(latitude=42.6390, longitude=74.4940)

    source = Source(
        latitude=89.91, longitude=0.0, depth_in_m=12000,
        m_rr=4.710000e+24 / 1E7,
        m_tt=3.810000e+22 / 1E7,
        m_pp=-4.740000e+24 / 1E7,
        m_rt=3.990000e+23 / 1E7,
        m_rp=-8.050000e+23 / 1E7,
        m_tp=-1.230000e+24 / 1E7)

    dt = instaseis_bwd.info.dt
    sliprate = np.zeros(1000)
    sliprate[0] = 1.
    sliprate = lowpass(sliprate, 1./100., 1./dt, corners=4)

    source.set_sliprate(sliprate, dt, time_shift=0., normalize=True)

    # We can only do a dt that is a clean multiple of the original dt as
    # otherwise we will have small time shifts.
    dt = instaseis_bwd.info.dt / 4
    st_fin = instaseis_bwd.get_seismograms_finite_source(
        sources=[source], receiver=receiver,
        components=('Z', 'N', 'E', 'R', 'T'), dt=dt,
        kernelwidth=1)
    st_ref = instaseis_bwd.get_seismograms(
        source=source, receiver=receiver,
        components=('Z', 'N', 'E', 'R', 'T'), dt=dt, reconvolve_stf=True,
        remove_source_shift=False, kernelwidth=1)

    np.testing.assert_allclose(st_fin.select(component='Z')[0].data,
                               st_ref.select(component='Z')[0].data,
                               rtol=1E-7, atol=1E-12)
    np.testing.assert_allclose(st_fin.select(component='N')[0].data,
                               st_ref.select(component='N')[0].data,
                               rtol=1E-7, atol=1E-12)
    np.testing.assert_allclose(st_fin.select(component='E')[0].data,
                               st_ref.select(component='E')[0].data,
                               rtol=1E-7, atol=1E-12)
    np.testing.assert_allclose(st_fin.select(component='R')[0].data,
                               st_ref.select(component='R')[0].data,
                               rtol=1E-7, atol=1E-12)
    np.testing.assert_allclose(st_fin.select(component='T')[0].data,
                               st_ref.select(component='T')[0].data,
                               rtol=1E-7, atol=1E-12)

    # Test receiver settings in finite source route.
    receiver = Receiver(latitude=10.0, longitude=20.0)
    st = instaseis_bwd.get_seismograms_finite_source(
        sources=[source], receiver=receiver, components=('Z'))
    assert len(st) == 1
    tr = st[0]
    assert tr.stats.network == ""
    assert tr.stats.station == ""
    assert tr.stats.location == ""
    assert tr.stats.channel[-1] == "Z"

    # Receiver with everything.
    receiver = Receiver(latitude=10.0, longitude=20.0, network="BW",
                        station="ALTM", location="SY")
    st = instaseis_bwd.get_seismograms_finite_source(
        sources=[source], receiver=receiver, components=('Z'))

    assert len(st) == 1
    tr = st[0]
    assert tr.stats.network == "BW"
    assert tr.stats.station == "ALTM"
    assert tr.stats.location == "SY"
    assert tr.stats.channel[-1] == "Z"

    # Make sure the correct_mu setting actually does something.
    st_2 = instaseis_bwd.get_seismograms_finite_source(
        sources=[source], receiver=receiver, components=('Z'),
        correct_mu=True)
    assert st != st_2


def test_get_band_code_method():
    """
    Dummy test assuring the band code is determined correctly.
    """
    codes = {
        0.0005: "F",
        0.001: "F",
        0.0011: "C",
        0.004: "C",
        0.0041: "H",
        0.0125: "H",
        0.0126: "B",
        0.1: "B",
        0.11: "M",
        0.99: "M",
        1.0: "L",
        10.0: "L",
        33.0: "L"
    }
    for dt, letter in codes.items():
        assert get_band_code(dt) == letter


@pytest.mark.parametrize("bwd_db", BW_DISPL_DBS)
def test_origin_time_of_resulting_seismograms(bwd_db):
    """
    Makes sure that the origin time is passed to the seismograms.
    """
    instaseis_bwd = find_and_open_files(bwd_db)

    receiver = Receiver(latitude=42.6390, longitude=74.4940)
    source = Source(
        latitude=89.91, longitude=0.0, depth_in_m=12000,
        m_rr=4.710000e+24 / 1E7,
        m_tt=3.810000e+22 / 1E7,
        m_pp=-4.740000e+24 / 1E7,
        m_rt=3.990000e+23 / 1E7,
        m_rp=-8.050000e+23 / 1E7,
        m_tp=-1.230000e+24 / 1E7)

    st = instaseis_bwd.get_seismograms(
        source=source, receiver=receiver, components=('Z'))

    # Default time is it timestamp 0.
    assert st[0].stats.starttime == obspy.UTCDateTime(0)

    # Set some custom time.
    org_time = obspy.UTCDateTime(2014, 1, 1)
    source = Source(
        latitude=89.91, longitude=0.0, depth_in_m=12000,
        m_rr=4.710000e+24 / 1E7,
        m_tt=3.810000e+22 / 1E7,
        m_pp=-4.740000e+24 / 1E7,
        m_rt=3.990000e+23 / 1E7,
        m_rp=-8.050000e+23 / 1E7,
        m_tp=-1.230000e+24 / 1E7,
        origin_time=org_time)
    st = instaseis_bwd.get_seismograms(
        source=source, receiver=receiver, components=('Z'))

    # Default time is it timestamp 0.
    assert st[0].stats.starttime == org_time


@pytest.mark.parametrize("bwd_db", BW_DISPL_DBS)
def test_higher_level_event_and_receiver_parsing(bwd_db):
    """
    Tests that events and receivers can be parsed from different supported
    formats.
    """
    # Create an event and modify it to match the settings of the test data.
    event = obspy.read_events(os.path.join(
        DATA, "GCMT_event_STRAIT_OF_GIBRALTAR.xml"))[0]
    event.origins = event.origins[:1]
    event.magnitudes = event.magnitudes[:1]
    event.focal_mechanisms = event.focal_mechanisms[:1]

    org = event.origins[0]
    mt = event.focal_mechanisms[0].moment_tensor.tensor

    # Convert everything to WGS84 values. Instaseis will assume them to be
    # wgs84 and convert it to geocentric.

    org.latitude = geocentric_to_elliptic_latitude(89.91)
    org.longitude = 0.0
    org.depth = 12000
    org.time = obspy.UTCDateTime(2014, 1, 5)
    mt.m_rr = 4.710000e+24 / 1E7
    mt.m_tt = 3.810000e+22 / 1E7
    mt.m_pp = -4.740000e+24 / 1E7
    mt.m_rt = 3.990000e+23 / 1E7
    mt.m_rp = -8.050000e+23 / 1E7
    mt.m_tp = -1.230000e+24 / 1E7

    # Same with the receiver objects.
    inv = obspy.read_inventory(os.path.join(DATA, "TA.Q56A..BH.xml"))
    inv[0][0].latitude = geocentric_to_elliptic_latitude(42.6390)
    inv[0][0].longitude = 74.4940
    inv[0][0].elevation = 0.0
    inv[0][0].channels = []

    # receiver = Receiver(latitude=42.6390, longitude=74.4940)
    instaseis_bwd = find_and_open_files(bwd_db)

    st = instaseis_bwd.get_seismograms(source=event, receiver=inv,
                                       components=('Z'))
    np.testing.assert_allclose(st.select(component='Z')[0].data,
                               BWD_TEST_DATA["Z"], rtol=1E-7, atol=1E-12)

    # Make sure the attributes are correct.
    assert st[0].stats.starttime == org.time
    assert st[0].stats.network == "TA"
    assert st[0].stats.station == "Q56A"


@pytest.mark.parametrize("db", DBS)
def test_available_components_decorator(db):
    db = find_and_open_files(db)
    if "vertical" in db.info.components and "horizontal" in db.info.components:
        assert db.available_components == ["Z", "N", "E", "R", "T"]
    elif "4 elemental moment tensors" in db.info.components:
        assert db.available_components == ["Z", "N", "E", "R", "T"]
    elif "vertical" in db.info.components:
        assert db.available_components == ["Z"]
    elif "horizontal" in db.info.components:
        assert db.available_components == ["N", "E", "R", "T"]
    else:  # pragma: no cover
        raise NotImplementedError


@pytest.mark.parametrize("db", DBS)
def test_resampling_and_time_settings(db):
    """
    This tests should assure that the origin time is always the peak of the
    source time function.
    """
    db = find_and_open_files(db)

    origin_time = obspy.UTCDateTime(2015, 1, 1, 1, 1)

    source = Source(latitude=4., longitude=3.0, depth_in_m=0,
                    m_rr=4.71e+17, m_tt=3.81e+17, m_pp=-4.74e+17,
                    m_rt=3.99e+17, m_rp=-8.05e+17, m_tp=-1.23e+17,
                    origin_time=origin_time)
    receiver = Receiver(latitude=10., longitude=20., depth_in_m=0)

    # The `remove_source_shift` argument will cut away the first couple of
    # samples. This results in the first sample being the origin time.
    st_r_shift = db.get_seismograms(source=source, receiver=receiver,
                                    remove_source_shift=True,
                                    components=db.available_components)
    for tr in st_r_shift:
        assert tr.stats.starttime == origin_time
    length = st_r_shift[0].stats.npts

    # Now if we don't remove it we should have a couple more samples. If we
    # don't resample it should be more or less exact.
    st = db.get_seismograms(source=source, receiver=receiver,
                            remove_source_shift=False,
                            components=db.available_components)
    for tr in st:
        assert tr.stats.starttime == origin_time - \
            (db.info.src_shift_samples * db.info.dt)
    # This should now contain 7 more samples.
    assert st[0].stats.npts == length + 7

    # Make sure the shift does what is is supposed to do.
    for tr_r, tr in zip(st_r_shift, st):
        np.testing.assert_allclose(tr_r.data, tr[7:], atol=1E-9)

    # This becomes a bit trickier if resampling is involved. It will always
    # resample in a way that the given origin time is at the peak of the
    # source time function.
    st_r_shift = db.get_seismograms(source=source, receiver=receiver,
                                    remove_source_shift=True, dt=12,
                                    kernelwidth=1,
                                    components=db.available_components)
    for tr in st_r_shift:
        assert tr.stats.starttime == origin_time
    length = st_r_shift[0].stats.npts

    st = db.get_seismograms(source=source, receiver=receiver,
                            remove_source_shift=False, dt=12,
                            kernelwidth=1,
                            components=db.available_components)
    for tr in st:
        assert tr.stats.starttime == origin_time - 14 * 12

    # Now this should have exactly 14 samples more then without removing the
    # source shift.
    assert length + 14 == tr.stats.npts

    # Make sure the shift does what is is supposed to do.
    for tr_r, tr in zip(st_r_shift, st):
        np.testing.assert_allclose(tr_r.data, tr[14:], atol=1E-9)


@pytest.mark.parametrize("db", DBS)
def test_time_settings_with_resample_stf(db):
    """
    Test the time settings with resampling an stf. In that case the rules
    are pretty simple: The first sample will always be set to the origin time.
    """
    from obspy.signal.filter import lowpass

    origin_time = obspy.UTCDateTime(2015, 1, 1, 1, 1)

    db = find_and_open_files(db)

    receiver = Receiver(latitude=42.6390, longitude=74.4940)
    source = Source(
        latitude=89.91, longitude=0.0, depth_in_m=12000,
        m_rr=4.710000e+24 / 1E7,
        m_tt=3.810000e+22 / 1E7,
        m_pp=-4.740000e+24 / 1E7,
        m_rt=3.990000e+23 / 1E7,
        m_rp=-8.050000e+23 / 1E7,
        m_tp=-1.230000e+24 / 1E7,
        origin_time=origin_time)

    dt = db.info.dt
    sliprate = np.zeros(1000)
    sliprate[0] = 1.
    sliprate = lowpass(sliprate, 1./100., 1./dt, corners=4)

    source.set_sliprate(sliprate, dt, time_shift=0., normalize=True)

    # Using both reconvolve stf and remove source shift results in an error.
    with pytest.raises(ValueError) as err:
        db.get_seismograms(
            source=source, receiver=receiver,
            components=db.available_components, dt=0.1, reconvolve_stf=True,
            remove_source_shift=True)
    assert isinstance(err.value, ValueError)
    assert err.value.args[0] == ("'remove_source_shift' argument not "
                                 "compatible with 'reconvolve_stf'.")

    # No matter the dt, the first sample will always be set to the origin time.
    st = db.get_seismograms(
        source=source, receiver=receiver,
        components=db.available_components, reconvolve_stf=True,
        remove_source_shift=False)
    for tr in st:
        assert tr.stats.starttime == origin_time

    st = db.get_seismograms(
        source=source, receiver=receiver,
        components=db.available_components, reconvolve_stf=True,
        remove_source_shift=False, dt=0.1)
    for tr in st:
        assert tr.stats.starttime == origin_time

    st = db.get_seismograms(
        source=source, receiver=receiver,
        components=db.available_components, reconvolve_stf=True,
        remove_source_shift=False, dt=1.0)
    for tr in st:
        assert tr.stats.starttime == origin_time


@pytest.mark.parametrize("db", DBS)
def test_remove_samples_at_end_for_interpolation(db):
    """
    Remove some samples at the end for the resampling to avoid boundary
    effects.
    """
    db = find_and_open_files(db)

    origin_time = obspy.UTCDateTime(2015, 1, 1, 1, 1)

    source = Source(latitude=4., longitude=3.0, depth_in_m=0,
                    m_rr=4.71e+17, m_tt=3.81e+17, m_pp=-4.74e+17,
                    m_rt=3.99e+17, m_rp=-8.05e+17, m_tp=-1.23e+17,
                    origin_time=origin_time)
    receiver = Receiver(latitude=10., longitude=20., depth_in_m=0)

    st = db.get_seismograms(source=source, receiver=receiver,
                            remove_source_shift=True)

    # The "identify" interpolation will still perform the interpolation but
    # nothing will be cut. This is a special, hard-coded case mainly for
    # testing. But it also not wrong.
    st_2 = db.get_seismograms(source=source, receiver=receiver,
                              remove_source_shift=True, dt=db.info.dt,
                              kernelwidth=1)
    for tr, tr_2 in zip(st, st_2):
        assert tr == tr_2

    # The original dt is a bit more then 24, so a dt of twelve should result
    # in three missing samples for a=1, and 5 for a=2.
    samples = int((st_2[0].stats.endtime - st_2[0].stats.starttime) / 12.0) + 1

    st_3 = db.get_seismograms(source=source, receiver=receiver,
                              remove_source_shift=True, dt=12,
                              kernelwidth=1)
    for tr in st_3:
        assert tr.stats.npts == samples - 3

    st_4 = db.get_seismograms(source=source, receiver=receiver,
                              remove_source_shift=True, dt=12,
                              kernelwidth=2)
    for tr in st_4:
        assert tr.stats.npts == samples - 5

    st_5 = db.get_seismograms(source=source, receiver=receiver,
                              remove_source_shift=True, dt=12,
                              kernelwidth=3)
    for tr in st_5:
        assert tr.stats.npts == samples - 7


@pytest.mark.parametrize("db", DBS)
def test_get_time_information(db):
    """
    Tests the _get_seismogram_times() function. Also make sure it is
    consistent with the actually produces seismograms.
    """
    db = find_and_open_files(db)

    origin_time = obspy.UTCDateTime(2015, 1, 1, 1, 1)

    source = Source(latitude=4., longitude=3.0, depth_in_m=0,
                    m_rr=4.71e+17, m_tt=3.81e+17, m_pp=-4.74e+17,
                    m_rt=3.99e+17, m_rp=-8.05e+17, m_tp=-1.23e+17,
                    origin_time=origin_time)
    receiver = Receiver(latitude=10., longitude=20., depth_in_m=0)

    # First case is very simple.
    par = {"remove_source_shift": True,
           "dt": None,
           "kernelwidth": 5}
    tr = db.get_seismograms(source=source, receiver=receiver,
                            components=db.available_components, **par)[0]
    times = _get_seismogram_times(info=db.info, origin_time=origin_time, **par)

    assert tr.stats.starttime == times["starttime"]
    assert tr.stats.starttime == origin_time
    assert tr.stats.endtime == times["endtime"]
    assert tr.stats.endtime == origin_time + 65 * db.info.dt
    assert times["samples_cut_at_end"] == 0
    assert times["ref_sample"] == 7
    assert times["time_shift_at_beginning"] == 0
    assert times["npts_before_shift_removal"] == 73
    assert times["npts"] == 66
    assert tr.stats.npts == times["npts"]

    # Now the same, but without removal of the the source shift.
    par = {"remove_source_shift": False,
           "dt": None,
           "kernelwidth": 5}
    tr = db.get_seismograms(source=source, receiver=receiver,
                            components=db.available_components, **par)[0]
    times = _get_seismogram_times(info=db.info, origin_time=origin_time, **par)

    assert tr.stats.starttime == times["starttime"]
    assert tr.stats.starttime == origin_time - 7 * db.info.dt
    assert tr.stats.endtime == times["endtime"]
    assert tr.stats.endtime == origin_time + 65 * db.info.dt
    assert times["samples_cut_at_end"] == 0
    assert times["ref_sample"] == 7
    assert times["time_shift_at_beginning"] == 0
    assert times["npts_before_shift_removal"] == 73
    assert times["npts"] == 73
    assert tr.stats.npts == times["npts"]

    # The "identify" interpolation will still perform the interpolation but
    # nothing will be cut. This is a special, hard-coded case mainly for
    # testing. But it also not wrong.
    par = {"remove_source_shift": True,
           "dt": db.info.dt,
           "kernelwidth": 2}
    tr = db.get_seismograms(source=source, receiver=receiver,
                            components=db.available_components, **par)[0]
    times = _get_seismogram_times(info=db.info, origin_time=origin_time, **par)

    assert tr.stats.starttime == times["starttime"]
    assert tr.stats.starttime == origin_time
    assert tr.stats.endtime == times["endtime"]
    assert tr.stats.endtime == origin_time + 65 * db.info.dt
    assert times["samples_cut_at_end"] == 0
    assert times["ref_sample"] == 7
    assert times["time_shift_at_beginning"] == 0
    assert times["npts_before_shift_removal"] == 73
    assert times["npts"] == 66
    assert tr.stats.npts == times["npts"]

    par = {"remove_source_shift": False,
           "dt": db.info.dt,
           "kernelwidth": 5}
    tr = db.get_seismograms(source=source, receiver=receiver,
                            components=db.available_components, **par)[0]
    times = _get_seismogram_times(info=db.info, origin_time=origin_time, **par)

    assert tr.stats.starttime == times["starttime"]
    assert tr.stats.starttime == origin_time - 7 * db.info.dt
    assert tr.stats.endtime == times["endtime"]
    assert tr.stats.endtime == origin_time + 65 * db.info.dt
    assert times["samples_cut_at_end"] == 0
    assert times["ref_sample"] == 7
    assert times["time_shift_at_beginning"] == 0
    assert times["npts_before_shift_removal"] == 73
    assert times["npts"] == 73
    assert tr.stats.npts == times["npts"]

    # Now resampling with various values of a.
    par = {"remove_source_shift": True,
           "dt": 12.0,
           "kernelwidth": 1}
    tr = db.get_seismograms(source=source, receiver=receiver,
                            components=db.available_components, **par)[0]
    times = _get_seismogram_times(info=db.info, origin_time=origin_time, **par)

    assert tr.stats.starttime == times["starttime"]
    assert tr.stats.starttime == origin_time
    assert tr.stats.endtime == times["endtime"]
    # 65 sample intervals fit with db.info.dt.
    endtime_with_dt = int((65 * db.info.dt) / 12.0) * 12.0
    endtime_minus_a = endtime_with_dt - 1 * db.info.dt
    assert tr.stats.endtime == origin_time + int(endtime_minus_a / 12.0) * 12.0
    assert times["samples_cut_at_end"] == 3
    assert times["ref_sample"] == 14
    assert round(times["time_shift_at_beginning"], 4) == \
        round(((7 * db.info.dt) / 12.0) % 1 * 12.0, 4)
    assert times["npts_before_shift_removal"] == 145
    assert times["npts"] == 131
    assert tr.stats.npts == times["npts"]

    par = {"remove_source_shift": True,
           "dt": 12.0,
           "kernelwidth": 2}
    tr = db.get_seismograms(source=source, receiver=receiver,
                            components=db.available_components, **par)[0]
    times = _get_seismogram_times(info=db.info, origin_time=origin_time, **par)

    assert tr.stats.starttime == times["starttime"]
    assert tr.stats.starttime == origin_time
    assert tr.stats.endtime == times["endtime"]
    # 65 sample intervals fit with db.info.dt.
    endtime_with_dt = int((65 * db.info.dt) / 12.0) * 12.0
    endtime_minus_a = endtime_with_dt - 2 * db.info.dt
    assert tr.stats.endtime == origin_time + int(endtime_minus_a / 12.0) * 12.0
    assert times["samples_cut_at_end"] == 5
    assert times["ref_sample"] == 14
    assert round(times["time_shift_at_beginning"], 4) == \
        round(((7 * db.info.dt) / 12.0) % 1 * 12.0, 4)
    assert times["npts_before_shift_removal"] == 143
    assert times["npts"] == 129
    assert tr.stats.npts == times["npts"]

    par = {"remove_source_shift": True,
           "dt": 12.0,
           "kernelwidth": 7}
    tr = db.get_seismograms(source=source, receiver=receiver,
                            components=db.available_components, **par)[0]
    times = _get_seismogram_times(info=db.info, origin_time=origin_time, **par)

    assert tr.stats.starttime == times["starttime"]
    assert tr.stats.starttime == origin_time
    assert tr.stats.endtime == times["endtime"]
    endtime_with_dt = int((65 * db.info.dt) / 12.0) * 12.0
    endtime_minus_a = endtime_with_dt - 7 * db.info.dt
    assert tr.stats.endtime == origin_time + int(endtime_minus_a / 12.0) * 12.0
    assert times["samples_cut_at_end"] == 15
    assert times["ref_sample"] == 14
    assert round(times["time_shift_at_beginning"], 4) == \
        round(((7 * db.info.dt) / 12.0) % 1 * 12.0, 4)
    assert times["npts_before_shift_removal"] == 133
    assert times["npts"] == 119
    assert tr.stats.npts == times["npts"]

    # Same once again but this time no source time shift
    par = {"remove_source_shift": False,
           "dt": 12.0,
           "kernelwidth": 1}
    tr = db.get_seismograms(source=source, receiver=receiver,
                            components=db.available_components, **par)[0]
    times = _get_seismogram_times(info=db.info, origin_time=origin_time, **par)

    assert tr.stats.starttime == times["starttime"]
    assert tr.stats.starttime == origin_time - 14 * 12.0
    assert tr.stats.endtime == times["endtime"]
    # 65 sample intervals fit with db.info.dt.
    endtime_with_dt = int((65 * db.info.dt) / 12.0) * 12.0
    endtime_minus_a = endtime_with_dt - 1 * db.info.dt
    assert tr.stats.endtime == origin_time + int(endtime_minus_a / 12.0) * 12.0
    assert times["samples_cut_at_end"] == 3
    assert times["ref_sample"] == 14
    assert round(times["time_shift_at_beginning"], 4) == \
        round(((7 * db.info.dt) / 12.0) % 1 * 12.0, 4)
    assert times["npts_before_shift_removal"] == 145
    assert times["npts"] == 145
    assert tr.stats.npts == times["npts"]

    par = {"remove_source_shift": False,
           "dt": 12.0,
           "kernelwidth": 2}
    tr = db.get_seismograms(source=source, receiver=receiver,
                            components=db.available_components, **par)[0]
    times = _get_seismogram_times(info=db.info, origin_time=origin_time, **par)

    assert tr.stats.starttime == times["starttime"]
    assert tr.stats.starttime == origin_time - 14 * 12
    assert tr.stats.endtime == times["endtime"]
    # 65 sample intervals fit with db.info.dt.
    endtime_with_dt = int((65 * db.info.dt) / 12.0) * 12.0
    endtime_minus_a = endtime_with_dt - 2 * db.info.dt
    assert tr.stats.endtime == origin_time + int(endtime_minus_a / 12.0) * 12.0
    assert times["samples_cut_at_end"] == 5
    assert times["ref_sample"] == 14
    assert round(times["time_shift_at_beginning"], 4) == \
        round(((7 * db.info.dt) / 12.0) % 1 * 12.0, 4)
    assert times["npts_before_shift_removal"] == 143
    assert times["npts"] == 143
    assert tr.stats.npts == times["npts"]

    par = {"remove_source_shift": False,
           "dt": 12.0,
           "kernelwidth": 7}
    tr = db.get_seismograms(source=source, receiver=receiver,
                            components=db.available_components, **par)[0]
    times = _get_seismogram_times(info=db.info, origin_time=origin_time, **par)

    assert tr.stats.starttime == times["starttime"]
    assert tr.stats.starttime == origin_time - 14 * 12
    assert tr.stats.endtime == times["endtime"]
    endtime_with_dt = int((65 * db.info.dt) / 12.0) * 12.0
    endtime_minus_a = endtime_with_dt - 7 * db.info.dt
    assert tr.stats.endtime == origin_time + int(endtime_minus_a / 12.0) * 12.0
    assert times["samples_cut_at_end"] == 15
    assert times["ref_sample"] == 14
    assert round(times["time_shift_at_beginning"], 4) == \
        round(((7 * db.info.dt) / 12.0) % 1 * 12.0, 4)
    assert times["npts_before_shift_removal"] == 133
    assert times["npts"] == 133
    assert tr.stats.npts == times["npts"]


@pytest.mark.parametrize("db", DBS)
def test_get_time_information_reconvolve_stf(db):
    """
    Tests the _get_seismogram_times() function but with reconvolve_stf = True.
    In that case time shifts and what not are no longer applied.
    """
    from obspy.signal.filter import lowpass
    db = find_and_open_files(db)

    origin_time = obspy.UTCDateTime(2015, 1, 1, 1, 1)

    source = Source(latitude=4., longitude=3.0, depth_in_m=0,
                    m_rr=4.71e+17, m_tt=3.81e+17, m_pp=-4.74e+17,
                    m_rt=3.99e+17, m_rp=-8.05e+17, m_tp=-1.23e+17,
                    origin_time=origin_time)
    receiver = Receiver(latitude=10., longitude=20., depth_in_m=0)
    dt = db.info.dt
    sliprate = np.zeros(1000)
    sliprate[0] = 1.
    sliprate = lowpass(sliprate, 1./100., 1./dt, corners=4)

    source.set_sliprate(sliprate, dt, time_shift=0., normalize=True)

    # remove_source_shift and reconvolve_stf cannot be true at the same time.
    par = {"remove_source_shift": True,
           "reconvolve_stf": True,
           "dt": None,
           "kernelwidth": 5}
    with pytest.raises(ValueError) as err:
        _get_seismogram_times(info=db.info, origin_time=origin_time, **par)
    assert err.value.args[0] == (
        "'remove_source_shift' argument not compatible with 'reconvolve_stf'.")

    par = {"remove_source_shift": False,
           "reconvolve_stf": True,
           "dt": None,
           "kernelwidth": 5}
    tr = db.get_seismograms(source=source, receiver=receiver,
                            components=db.available_components, **par)[0]
    times = _get_seismogram_times(info=db.info, origin_time=origin_time, **par)

    # The seismogram will always start with the origin time and the
    # ref_sample is always 0.
    assert tr.stats.starttime == times["starttime"]
    assert tr.stats.starttime == origin_time
    assert tr.stats.endtime == times["endtime"]
    assert tr.stats.endtime == origin_time + 72 * db.info.dt
    assert times["samples_cut_at_end"] == 0
    assert times["ref_sample"] == 0
    assert times["time_shift_at_beginning"] == 0
    assert times["npts_before_shift_removal"] == 73
    assert times["npts"] == 73
    assert times["npts"] == times["npts_before_shift_removal"]
    assert tr.stats.npts == times["npts"]

    # A couple more with a given interpolation kernel width.
    par = {"remove_source_shift": False,
           "reconvolve_stf": True,
           "dt": 12.0,
           "kernelwidth": 1}
    tr = db.get_seismograms(source=source, receiver=receiver,
                            components=db.available_components, **par)[0]
    times = _get_seismogram_times(info=db.info, origin_time=origin_time, **par)

    assert tr.stats.starttime == times["starttime"]
    assert tr.stats.starttime == origin_time
    assert tr.stats.endtime == times["endtime"]
    # 72 sample intervals fit with db.info.dt.
    endtime_with_dt = int((72 * db.info.dt) / 12.0) * 12.0
    endtime_minus_a = \
        endtime_with_dt - int(math.ceil(1 * db.info.dt / 12.0)) * 12.0
    assert tr.stats.endtime == origin_time + int(endtime_minus_a / 12.0) * 12.0
    assert times["samples_cut_at_end"] == 3
    assert times["ref_sample"] == 0
    assert times["time_shift_at_beginning"] == 0
    assert times["npts_before_shift_removal"] == 146
    assert times["npts"] == 146
    assert times["npts"] == times["npts_before_shift_removal"]
    assert tr.stats.npts == times["npts"]

    par = {"remove_source_shift": False,
           "reconvolve_stf": True,
           "dt": 12.0,
           "kernelwidth": 2}
    tr = db.get_seismograms(source=source, receiver=receiver,
                            components=db.available_components, **par)[0]
    times = _get_seismogram_times(info=db.info, origin_time=origin_time, **par)

    assert tr.stats.starttime == times["starttime"]
    assert tr.stats.starttime == origin_time
    assert tr.stats.endtime == times["endtime"]
    # 65 sample intervals fit with db.info.dt.
    endtime_with_dt = int((72 * db.info.dt) / 12.0) * 12.0
    endtime_minus_a = endtime_with_dt - 2 * db.info.dt
    assert tr.stats.endtime == origin_time + int(endtime_minus_a / 12.0) * 12.0
    assert times["samples_cut_at_end"] == 5
    assert times["ref_sample"] == 0
    assert times["time_shift_at_beginning"] == 0
    assert times["npts_before_shift_removal"] == 144
    assert times["npts"] == 144
    assert times["npts"] == times["npts_before_shift_removal"]
    assert tr.stats.npts == times["npts"]

    par = {"remove_source_shift": False,
           "reconvolve_stf": True,
           "dt": 12.0,
           "kernelwidth": 7}
    tr = db.get_seismograms(source=source, receiver=receiver,
                            components=db.available_components, **par)[0]
    times = _get_seismogram_times(info=db.info, origin_time=origin_time, **par)

    assert tr.stats.starttime == times["starttime"]
    assert tr.stats.starttime == origin_time
    assert tr.stats.endtime == times["endtime"]
    endtime_with_dt = int((72 * db.info.dt) / 12.0) * 12.0
    endtime_minus_a = endtime_with_dt - 7 * db.info.dt
    assert tr.stats.endtime == origin_time + int(endtime_minus_a / 12.0) * 12.0
    assert times["samples_cut_at_end"] == 15
    assert times["ref_sample"] == 0
    assert times["time_shift_at_beginning"] == 0
    assert times["npts_before_shift_removal"] == 134
    assert times["npts"] == 134
    assert times["npts"] == times["npts_before_shift_removal"]
    assert tr.stats.npts == times["npts"]


def test_wgs84_to_geocentric():
    """
    Tests the utility function.
    """
    assert elliptic_to_geocentric_latitude(0.0) == 0.0
    assert elliptic_to_geocentric_latitude(90.0) == 90.0
    assert elliptic_to_geocentric_latitude(-90.0) == -90.0

    # Difference minimal close to the poles and the equator.
    assert abs(elliptic_to_geocentric_latitude(0.1) - 0.1) < 1E-3
    assert abs(elliptic_to_geocentric_latitude(-0.1) + 0.1) < 1E-3
    assert abs(elliptic_to_geocentric_latitude(89.9) - 89.9) < 1E-3
    assert abs(elliptic_to_geocentric_latitude(-89.9) + 89.9) < 1E-3

    # The geographic latitude is larger then the geocentric on the northern
    # hemisphere.
    for _i in range(1, 90):
        assert _i > elliptic_to_geocentric_latitude(_i)

    # The opposite is true on the southern hemisphere.
    for _i in range(-1, -90, -1):
        assert _i < elliptic_to_geocentric_latitude(_i)

    # Small check to test the approximate ranges.
    assert 0.19 < 45.0 - elliptic_to_geocentric_latitude(45.0) < 0.2
    assert -0.19 > -45 - elliptic_to_geocentric_latitude(-45.0) > -0.2


def test_geocentric_to_wgs84():
    """
    Tests the utility function.
    """
    assert geocentric_to_elliptic_latitude(0.0) == 0.0
    assert geocentric_to_elliptic_latitude(90.0) == 90.0
    assert geocentric_to_elliptic_latitude(-90.0) == -90.0

    # Difference minimal close to the poles and the equator.
    assert abs(geocentric_to_elliptic_latitude(0.1) - 0.1) < 1E-3
    assert abs(geocentric_to_elliptic_latitude(-0.1) + 0.1) < 1E-3
    assert abs(geocentric_to_elliptic_latitude(89.9) - 89.9) < 1E-3
    assert abs(geocentric_to_elliptic_latitude(-89.9) + 89.9) < 1E-3

    # The geographic latitude is larger then the geocentric on the northern
    # hemisphere.
    for _i in range(1, 90):
        assert _i < geocentric_to_elliptic_latitude(_i)

    # The opposite is true on the southern hemisphere.
    for _i in range(-1, -90, -1):
        assert _i > geocentric_to_elliptic_latitude(_i)

    # Small check to test the approximate ranges.
    assert -0.19 > 45.0 - geocentric_to_elliptic_latitude(45.0) > -0.2
    assert 0.19 < -45 - geocentric_to_elliptic_latitude(-45.0) < 0.2


def test_coordinate_conversions_round_trips():
    """
    Tests round tripping of the coordinate conversion routines.
    """
    values = np.linspace(-90, 90, 100)
    for value in values:
        value = float(value)
        there = elliptic_to_geocentric_latitude(value)
        back = geocentric_to_elliptic_latitude(there)
        assert abs(back - value) < 1E-12

        there = geocentric_to_elliptic_latitude(value)
        back = elliptic_to_geocentric_latitude(there)
        assert abs(back - value) < 1E-12


def test_receiver_settings():
    db = find_and_open_files(os.path.join(DATA, "100s_db_fwd"))

    source = Source(latitude=4., longitude=3.0, depth_in_m=None,
                    m_rr=4.71e+17, m_tt=3.81e+17, m_pp=-4.74e+17,
                    m_rt=3.99e+17, m_rp=-8.05e+17, m_tp=-1.23e+17)

    # Receiver with neither network, nor station, nor receiver code.
    receiver = Receiver(latitude=10.0, longitude=20.0)

    st = db.get_seismograms(source=source, receiver=receiver,
                            components=["Z"])
    assert len(st) == 1
    tr = st[0]
    assert tr.stats.network == ""
    assert tr.stats.station == ""
    assert tr.stats.location == ""
    assert tr.stats.channel[-1] == "Z"

    # Receiver with everything.
    receiver = Receiver(latitude=10.0, longitude=20.0, network="BW",
                        station="ALTM", location="SY")

    st = db.get_seismograms(source=source, receiver=receiver,
                            components=["Z"])
    assert len(st) == 1
    tr = st[0]
    assert tr.stats.network == "BW"
    assert tr.stats.station == "ALTM"
    assert tr.stats.location == "SY"
    assert tr.stats.channel[-1] == "Z"


def test_some_failure_conditions():
    """
    Tests some failure conditions.
    """
    db = find_and_open_files(os.path.join(DATA, "100s_db_fwd"))
    source = Source(latitude=4., longitude=3.0, depth_in_m=None,
                    m_rr=4.71e+17, m_tt=3.81e+17, m_pp=-4.74e+17,
                    m_rt=3.99e+17, m_rp=-8.05e+17, m_tp=-1.23e+17)
    receiver = Receiver(latitude=10.0, longitude=20.0)

    # Reconvolving with stf requires a source time function.
    with pytest.raises(ValueError) as err:
        db.get_seismograms(
            source=source, receiver=receiver,
            components=('Z', 'N', 'E', 'R', 'T'), dt=2.0, reconvolve_stf=True,
            remove_source_shift=False, kernelwidth=1)

    assert err.value.args[0] == "source has no source time function"

    # Incompatible source time function.
    source.sliprate = np.arange(100)
    source.dt = 0.123
    with pytest.raises(ValueError) as err:
        db.get_seismograms(
            source=source, receiver=receiver,
            components=('Z', 'N', 'E', 'R', 'T'), dt=2.0, reconvolve_stf=True,
            remove_source_shift=False, kernelwidth=1)

    assert err.value.args[0] == "dt of the source not compatible"

    # Multiple receivers.
    with pytest.raises(ValueError) as err:
        db.get_seismograms(
            source=source, receiver=obspy.read_inventory(),
            components=('Z', 'N', 'E', 'R', 'T'))
    assert err.value.args[0].startswith("Receiver object/file contains "
                                        "multiple stations.")

    # Wrong kind.
    with pytest.raises(ValueError) as err:
        db.get_seismograms(
            source=source, receiver=receiver, kind="random")
    assert err.value.args[0] == "unknown kind 'random'"


def test_sizeof_fmt_function():
    assert sizeof_fmt(1024) == "1.0 KB"
    assert sizeof_fmt(1024 ** 2) == "1.0 MB"
    assert sizeof_fmt(1024 ** 3) == "1.0 GB"
    assert sizeof_fmt(1024 ** 4) == "1.0 TB"


def test_failures_when_opening_databases(tmpdir):
    """
    Tests various failures when opening databases.
    """
    # Add a deep folder that should not be tested.
    os.makedirs(os.path.join(tmpdir.strpath, "1", "2", "3", "4", "5"))

    # Nothing there currently.
    with pytest.raises(InstaseisNotFoundError) as err:
        find_and_open_files(tmpdir.strpath)
    assert err.value.args[0].startswith("No suitable netCDF files")

    # Three files are no good.
    f_1 = os.path.join(tmpdir.strpath, "1", "ordered_output.nc4")
    f_2 = os.path.join(tmpdir.strpath, "2", "ordered_output.nc4")
    f_3 = os.path.join(tmpdir.strpath, "3", "ordered_output.nc4")

    for f in [f_1, f_2, f_3]:
        if not os.path.exists(os.path.dirname(f)):
            os.makedirs(os.path.dirname(f))
        with io.open(f, "wb") as fh:
            fh.write(b" ")

    with pytest.raises(InstaseisError) as err:
        find_and_open_files(tmpdir.strpath)
    assert err.value.args[0].startswith("1, 2 or 4 netCDF must be present in "
                                        "the folder structure.")

    # Two netcdf files but in funny places.
    os.remove(f_3)
    with pytest.raises(InstaseisError) as err:
        find_and_open_files(tmpdir.strpath)
    assert err.value.args[0].startswith(
        "Could not find any suitable netCDF files. Did you pass the correct "
        "directory?")

    # Two PX files.
    os.remove(f_1)
    os.remove(f_2)
    f_1 = os.path.join(tmpdir.strpath, "1", "PX", "ordered_output.nc4")
    f_2 = os.path.join(tmpdir.strpath, "2", "PX", "ordered_output.nc4")
    for f in [f_1, f_2]:
        if not os.path.exists(os.path.dirname(f)):
            os.makedirs(os.path.dirname(f))
        with io.open(f, "wb") as fh:
            fh.write(b" ")

    with pytest.raises(InstaseisError) as err:
        find_and_open_files(tmpdir.strpath)
    assert err.value.args[0].startswith("Found 2 files for component PX:")

    # Only two moment tensor components.
    os.remove(f_1)
    os.remove(f_2)
    f_1 = os.path.join(tmpdir.strpath, "1", "MZZ", "ordered_output.nc4")
    f_2 = os.path.join(tmpdir.strpath, "2", "MXX_P_MYY", "ordered_output.nc4")
    for f in [f_1, f_2]:
        if not os.path.exists(os.path.dirname(f)):
            os.makedirs(os.path.dirname(f))
        with io.open(f, "wb") as fh:
            fh.write(b" ")

    with pytest.raises(InstaseisError) as err:
        find_and_open_files(tmpdir.strpath)
    assert err.value.args[0] == ("Expecting all four elemental moment tensor "
                                 "subfolders to be present.")


@pytest.mark.parametrize("database_folder", DBS)
@pytest.mark.parametrize("read_on_demand", [True, False])
def test_read_on_demand(database_folder, read_on_demand):
    """
    Make sure that databases work in read_on_demand mode.
    """
    # The test data is not valid for deep forward DBs.
    if "fwd_deep" in database_folder:
        return

    db = find_and_open_files(database_folder, read_on_demand=read_on_demand)

    # Test requires all 3 components.
    if "only" in db.info.components:
        return

    receiver = Receiver(latitude=42.6390, longitude=74.4940)

    if "100s_db_fwd" in database_folder:
        depth_in_m = None
    else:
        depth_in_m = 12000

    source = Source(
        latitude=89.91, longitude=0.0, depth_in_m=depth_in_m,
        m_rr=4.710000e+24 / 1E7,
        m_tt=3.810000e+22 / 1E7,
        m_pp=-4.740000e+24 / 1E7,
        m_rt=3.990000e+23 / 1E7,
        m_rp=-8.050000e+23 / 1E7,
        m_tp=-1.230000e+24 / 1E7)

    st = db.get_seismograms(source=source, receiver=receiver,
                            components=('Z', 'N', 'E', 'R', 'T'))

    if database_folder in TEST_DATA:
        td = TEST_DATA[database_folder]

    np.testing.assert_allclose(st.select(component='Z')[0].data,
                               td["Z"], rtol=1E-7, atol=1E-12)
    np.testing.assert_allclose(st.select(component='N')[0].data,
                               td["N"], rtol=1E-7, atol=1E-12)
    np.testing.assert_allclose(st.select(component='E')[0].data,
                               td["E"], rtol=1E-7, atol=1E-12)
    np.testing.assert_allclose(st.select(component='R')[0].data,
                               td["R"], rtol=1E-7, atol=1E-12)
    np.testing.assert_allclose(st.select(component='T')[0].data,
                               td["T"], rtol=1E-7, atol=1E-12)


@pytest.mark.skipif("merged_100s_db_fwd" not in pytest.config.dbs["databases"],
                    reason="requires generated tests databases.")
def test_merged_forward_database_layout():
    """
    Make sure the merged fwd database layout returns the same result as then
    default forward layout.
    """
    fwd_db = os.path.join(DATA, "100s_db_fwd")
    fwd_db_m = pytest.config.dbs["databases"]["merged_100s_db_fwd"]
    fwd_db = instaseis.open_db(fwd_db)
    fwd_db_m = instaseis.open_db(fwd_db_m)

    depths = [0, 10, 100, 150, 400]
    lngs = [-50, 0, 50, 100]
    lats = [-10, 0, 60]

    for rec_d in depths:
        for lat in lats:
            for lng in lngs:
                receiver = Receiver(latitude=10.0, longitude=20.0,
                                    depth_in_m=rec_d)
                source = Source(
                    latitude=lat, longitude=lng,
                    m_rr=4.710000e+24 / 1E7,
                    m_tt=3.810000e+22 / 1E7,
                    m_pp=-4.740000e+24 / 1E7,
                    m_rt=3.990000e+23 / 1E7,
                    m_rp=-8.050000e+23 / 1E7,
                    m_tp=-1.230000e+24 / 1E7)

                st_fwd = fwd_db.get_seismograms(
                    source=source, receiver=receiver,
                    components=('Z', 'N', 'E', 'R', 'T'))
                st_fwd_m = fwd_db_m.get_seismograms(
                    source=source, receiver=receiver,
                    components=('Z', 'N', 'E', 'R', 'T'))

                assert st_fwd == st_fwd_m


@pytest.mark.parametrize("bwd_db", BW_DISPL_DBS)
def test_error_handling_source_too_deep(bwd_db):
    """
    Tests the error handling if the source is too deep.
    """
    db = find_and_open_files(bwd_db)
    # 900 km is deeper than any test database.
    src = Source(latitude=4., longitude=3.0, depth_in_m=900000,
                 m_rr=4.71e+17, m_tt=3.81e+17, m_pp=-4.74e+17,
                 m_rt=3.99e+17, m_rp=-8.05e+17, m_tp=-1.23e+17)
    rec = Receiver(latitude=10., longitude=20., depth_in_m=0)

    with pytest.raises(ValueError) as err:
        db.get_seismograms(source=src, receiver=rec)

    assert err.value.args[0] == (
        "Source too deep. Source would be located at a radius of 5471000.0 "
        "meters. The database supports source radii from 6000000.0 to "
        "6371000.0 meters.")


@pytest.mark.parametrize("bwd_db", BW_DISPL_DBS)
def test_error_handling_source_too_shallow(bwd_db):
    """
    Tests the error handling if the source is too shallow.
    """
    db = find_and_open_files(bwd_db)
    src = Source(latitude=4., longitude=3.0, depth_in_m=-10000,
                 m_rr=4.71e+17, m_tt=3.81e+17, m_pp=-4.74e+17,
                 m_rt=3.99e+17, m_rp=-8.05e+17, m_tp=-1.23e+17)
    rec = Receiver(latitude=10., longitude=20., depth_in_m=0)

    with pytest.raises(ValueError) as err:
        db.get_seismograms(source=src, receiver=rec)

    assert err.value.args[0] == (
        "Source is too shallow. Source would be located at a radius of "
        "6381000.0 meters. The database supports source radii from "
        "6000000.0 to 6371000.0 meters.")


def test_receiver_too_deep_or_shallow_forward_database():
    """
    Test error handling for a too deep or too shallow for the forward mode.
    """
    db = find_and_open_files(os.path.join(DATA, "100s_db_fwd"))

    src = Source(latitude=4., longitude=3.0, depth_in_m=None,
                 m_rr=4.71e+17, m_tt=3.81e+17, m_pp=-4.74e+17,
                 m_rt=3.99e+17, m_rp=-8.05e+17, m_tp=-1.23e+17)

    # Receiver too deep.
    rec = Receiver(latitude=10., longitude=20., depth_in_m=1000000)

    with pytest.raises(ValueError) as err:
        db.get_seismograms(source=src, receiver=rec)

    assert err.value.args[0] == (
        "Receiver too deep. Receiver would be located at a radius of "
        "5371000.0 meters. The database supports receiver radii from "
        "6000000.0 to 6371000.0 meters.")

    # Receiver too shallow.
    rec = Receiver(latitude=10., longitude=20., depth_in_m=-10000)

    with pytest.raises(ValueError) as err:
        db.get_seismograms(source=src, receiver=rec)

    assert err.value.args[0] == (
        "Receiver is too shallow. Receiver would be located at a radius of "
        "6381000.0 meters. The database supports receiver radii from "
        "6000000.0 to 6371000.0 meters.")


@pytest.mark.parametrize("bwd_db", BW_DISPL_DBS)
def test_epicentral_distance_not_in_db(bwd_db):
    db = find_and_open_files(bwd_db)
    src = Source(latitude=0.0, longitude=0.0, depth_in_m=10000,
                 m_rr=4.71e+17, m_tt=3.81e+17, m_pp=-4.74e+17,
                 m_rt=3.99e+17, m_rp=-8.05e+17, m_tp=-1.23e+17)
    rec = Receiver(latitude=0.0, longitude=180.0, depth_in_m=0)

    # Works.
    db.get_seismograms(source=src, receiver=rec)

    # Hack!
    db.info.max_d = 170.0
    with pytest.raises(ValueError) as err:
        db.get_seismograms(source=src, receiver=rec)
    assert err.value.args[0] == ("Epicentral distance is 180.0 but should be "
                                 "in [0.0, 170.0].")

    rec.longitude = 10.0
    db.info.max_d = 180.0
    db.info.min_d = 20.0
    with pytest.raises(ValueError) as err:
        db.get_seismograms(source=src, receiver=rec)
    assert err.value.args[0] == ("Epicentral distance is 10.0 but should be "
                                 "in [20.0, 180.0].")


@pytest.mark.parametrize("bwd_db", BW_DISPL_DBS)
def test_source_depth_greens_function_error_handling(bwd_db):
    """
    Tests the error handling for the greens functions for too deep or too
    shallow sources.
    """
    db = find_and_open_files(bwd_db)

    # all good.
    db.get_greens_function(epicentral_distance_in_degree=10.0,
                           source_depth_in_m=100.0, definition="seiscomp")

    # too deep.
    with pytest.raises(ValueError) as err:
        db.get_greens_function(epicentral_distance_in_degree=10.0,
                               source_depth_in_m=900000.0,
                               definition="seiscomp")

    assert err.value.args[0] == (
        "Source too deep. Source would be located at a radius of 5471000.0 "
        "meters. The database supports source radii from 6000000.0 to "
        "6371000.0 meters.")

    # too shallow.
    with pytest.raises(ValueError) as err:
        db.get_greens_function(epicentral_distance_in_degree=10.0,
                               source_depth_in_m=-10000.0,
                               definition="seiscomp")

    assert err.value.args[0] == (
        "Source is too shallow. Source would be located at a radius of "
        "6381000.0 meters. The database supports source radii from "
        "6000000.0 to 6371000.0 meters.")


@pytest.mark.parametrize("bwd_db", BW_DISPL_DBS)
def test_dt_must_be_larger_than_zero(bwd_db):
    """
    dt must be larger than zero!
    """
    db = find_and_open_files(bwd_db)

    src = Source(latitude=4., longitude=3.0, depth_in_m=0,
                 m_rr=4.71e+17, m_tt=3.81e+17, m_pp=-4.74e+17,
                 m_rt=3.99e+17, m_rp=-8.05e+17, m_tp=-1.23e+17)
    rec = Receiver(latitude=10., longitude=20., depth_in_m=0)

    with pytest.raises(ValueError) as err:
        db.get_seismograms(source=src, receiver=rec, dt=0)
    assert err.value.args[0] == "dt must be bigger than 0."

    with pytest.raises(ValueError) as err:
        db.get_seismograms(source=src, receiver=rec, dt=-1.0)
    assert err.value.args[0] == "dt must be bigger than 0."

    with pytest.raises(ValueError) as err:
        db.get_greens_function(epicentral_distance_in_degree=10.0,
                               source_depth_in_m=0,
                               definition="seiscomp", dt=0)
    assert err.value.args[0] == "dt must be bigger than 0."
    with pytest.raises(ValueError) as err:
        db.get_greens_function(epicentral_distance_in_degree=10.0,
                               source_depth_in_m=0,
                               definition="seiscomp", dt=-1.0)
    assert err.value.args[0] == "dt must be bigger than 0."


@pytest.mark.parametrize("bwd_db", BW_DISPL_DBS)
def test_no_downsampling(bwd_db):
    """
    Make sure downsampling is not possible.
    """
    db = find_and_open_files(bwd_db)

    src = Source(latitude=4., longitude=3.0, depth_in_m=0,
                 m_rr=4.71e+17, m_tt=3.81e+17, m_pp=-4.74e+17,
                 m_rt=3.99e+17, m_rp=-8.05e+17, m_tp=-1.23e+17)
    rec = Receiver(latitude=10., longitude=20., depth_in_m=0)

    dt = db.info.dt

    # Same dt works.
    db.get_seismograms(source=src, receiver=rec, dt=dt)

    # A smaller one as well - this is an upsampling operation.
    db.get_seismograms(source=src, receiver=rec, dt=dt / 1.1)

    # But a larger one should raise.
    with pytest.raises(ValueError) as err:
        db.get_seismograms(source=src, receiver=rec, dt=dt * 1.1)
    assert err.value.args[0] == (
        "The database is sampled with a sample spacing of 24.725 seconds. You "
        "must not pass a 'dt' larger than that as that would be a "
        "downsampling operation which Instaseis does not do.")

    # Same with the greens functions.
    db.get_greens_function(epicentral_distance_in_degree=10.0,
                           source_depth_in_m=0,
                           definition="seiscomp", dt=dt)
    db.get_greens_function(epicentral_distance_in_degree=10.0,
                           source_depth_in_m=0,
                           definition="seiscomp", dt=dt / 1.1)
    # But a larger one should raise.
    with pytest.raises(ValueError) as err:
        db.get_greens_function(epicentral_distance_in_degree=10.0,
                               source_depth_in_m=0,
                               definition="seiscomp", dt=dt * 1.1)
    assert err.value.args[0] == (
        "The database is sampled with a sample spacing of 24.725 seconds. You "
        "must not pass a 'dt' larger than that as that would be a "
        "downsampling operation which Instaseis does not do.")
