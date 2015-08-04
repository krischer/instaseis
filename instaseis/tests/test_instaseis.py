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
import numpy as np
import obspy
import os
import pytest
import shutil

from instaseis.instaseis_db import InstaseisDB
from instaseis.base_instaseis_db import get_seismogram_times
from instaseis import Source, Receiver, ForceSource
from instaseis.helpers import get_band_code

from .testdata import BWD_TEST_DATA, FWD_TEST_DATA
from .testdata import BWD_STRAIN_ONLY_TEST_DATA, BWD_FORCE_TEST_DATA


# Most generic way to get the data folder path.
DATA = os.path.join(os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe()))), "data")

DBS = [os.path.join(DATA, "100s_db_fwd"),
       os.path.join(DATA, "100s_db_bwd_displ_only")]


def test_fwd_vs_bwd():
    """
    Test fwd against bwd mode
    """
    instaseis_fwd = InstaseisDB(os.path.join(DATA, "100s_db_fwd"))

    instaseis_bwd = InstaseisDB(os.path.join(DATA, "100s_db_bwd_displ_only"))

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


def test_fwd_vs_bwd_axial():
    """
    Test fwd against bwd mode, axial element. Differences are a bit larger then
    in non axial case, presumably because the close source, which is not
    exactly a point source in the SEM representation.
    """
    instaseis_fwd = InstaseisDB(os.path.join(DATA, "100s_db_fwd_deep"))

    instaseis_bwd = InstaseisDB(os.path.join(DATA, "100s_db_bwd_displ_only"))

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


def test_incremental_bwd():
    """
    incremental tests of bwd mode with displ_only db
    """
    instaseis_bwd = InstaseisDB(os.path.join(DATA, "100s_db_bwd_displ_only"))

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
    assert instaseis_bwd.meshes.px.strain_buffer.efficiency == 0.0
    assert instaseis_bwd.meshes.pz.strain_buffer.efficiency == 0.0

    # read on init
    instaseis_bwd = InstaseisDB(os.path.join(DATA, "100s_db_bwd_displ_only"),
                                read_on_demand=False)

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
    assert instaseis_bwd.meshes.px.strain_buffer.efficiency == 0.0
    assert instaseis_bwd.meshes.pz.strain_buffer.efficiency == 0.0

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
    assert instaseis_bwd.meshes.px.strain_buffer.efficiency == 1.0 / 2.0
    assert instaseis_bwd.meshes.pz.strain_buffer.efficiency == 1.0 / 2.0

    # test resampling with a no-op interpolation.
    dt = instaseis_bwd.info.dt
    st_bwd = instaseis_bwd.get_seismograms(
        source=source, receiver=receiver, components=('Z'), dt=dt,
        a_lanczos=5)

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
    instaseis_bwd = InstaseisDB(tmpdir)

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
    instaseis_bwd = InstaseisDB(tmpdir)

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
    instaseis_bwd = InstaseisDB(tmpdir)

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
    instaseis_bwd = InstaseisDB(tmpdir)

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
    instaseis_fwd = InstaseisDB(os.path.join(DATA, "100s_db_fwd"))

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
    instaseis_bwd = InstaseisDB(os.path.join(DATA, "100s_db_bwd_strain_only"))

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


def test_incremental_bwd_force_source():
    """
    incremental tests of bwd mode with source force
    """
    instaseis_bwd = InstaseisDB(os.path.join(DATA, "100s_db_bwd_displ_only"))

    receiver = Receiver(latitude=42.6390, longitude=74.4940)
    source = ForceSource(
        latitude=89.91, longitude=0.0, depth_in_m=12000,
        f_r=1.23E10,
        f_t=2.55E10,
        f_p=1.73E10)

    st_bwd = instaseis_bwd.get_seismograms(
        source=source, receiver=receiver, components=('Z', 'N', 'E', 'R', 'T'))

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


def test_finite_source():
    """
    incremental tests of bwd mode with source force
    """
    from obspy.signal.filter import lowpass
    instaseis_bwd = InstaseisDB(os.path.join(DATA, "100s_db_bwd_displ_only"))

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
        a_lanczos=1)
    st_ref = instaseis_bwd.get_seismograms(
        source=source, receiver=receiver,
        components=('Z', 'N', 'E', 'R', 'T'), dt=dt, reconvolve_stf=True,
        remove_source_shift=False, a_lanczos=1)

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


def test_origin_time_of_resulting_seismograms():
    """
    Makes sure that the origin time is passed to the seismograms.
    """
    instaseis_bwd = InstaseisDB(os.path.join(DATA, "100s_db_bwd_displ_only"))

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


def test_higher_level_event_and_receiver_parsing():
    """
    Tests that events and receivers can be parsed from different supported
    formats.
    """
    # Create an event and modify it to match the settings of the test data.
    event = obspy.readEvents(os.path.join(
        DATA, "GCMT_event_STRAIT_OF_GIBRALTAR.xml"))[0]
    event.origins = event.origins[:1]
    event.magnitudes = event.magnitudes[:1]
    event.focal_mechanisms = event.focal_mechanisms[:1]

    org = event.origins[0]
    mt = event.focal_mechanisms[0].moment_tensor.tensor

    org.latitude = 89.91
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
    inv[0][0].latitude = 42.6390
    inv[0][0].longitude = 74.4940
    inv[0][0].elevation = 0.0
    inv[0][0].channels = []

    # receiver = Receiver(latitude=42.6390, longitude=74.4940)
    instaseis_bwd = InstaseisDB(os.path.join(DATA, "100s_db_bwd_displ_only"))

    st = instaseis_bwd.get_seismograms(source=event, receiver=inv,
                                       components=('Z'))
    np.testing.assert_allclose(st.select(component='Z')[0].data,
                               BWD_TEST_DATA["Z"], rtol=1E-7, atol=1E-12)

    # Make sure the attributes are correct.
    assert st[0].stats.starttime == org.time
    assert st[0].stats.network == "TA"
    assert st[0].stats.station == "Q56A"


@pytest.mark.parametrize("db", DBS)
def test_resampling_and_time_settings(db):
    """
    This tests should assure that the origin time is always the peak of the
    source time function.
    """
    db = InstaseisDB(db)

    origin_time = obspy.UTCDateTime(2015, 1, 1, 1, 1)

    source = Source(latitude=4., longitude=3.0, depth_in_m=0,
                    m_rr=4.71e+17, m_tt=3.81e+17, m_pp=-4.74e+17,
                    m_rt=3.99e+17, m_rp=-8.05e+17, m_tp=-1.23e+17,
                    origin_time=origin_time)
    receiver = Receiver(latitude=10., longitude=20., depth_in_m=0)

    # The `remove_source_shift` argument will cut away the first couple of
    # samples. This results in the first sample being the origin time.
    st_r_shift = db.get_seismograms(source=source, receiver=receiver,
                                    remove_source_shift=True)
    for tr in st_r_shift:
        assert tr.stats.starttime == origin_time
    length = st_r_shift[0].stats.npts

    # Now if we don't remove it we should have a couple more samples. If we
    # don't resample it should be more or less exact.
    st = db.get_seismograms(source=source, receiver=receiver,
                            remove_source_shift=False)
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
                                    a_lanczos=1)
    for tr in st_r_shift:
        assert tr.stats.starttime == origin_time
    length = st_r_shift[0].stats.npts

    st = db.get_seismograms(source=source, receiver=receiver,
                            remove_source_shift=False, dt=12,
                            a_lanczos=1)
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

    db = InstaseisDB(db)

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
            components=('Z', 'N', 'E', 'R', 'T'), dt=0.1, reconvolve_stf=True,
            remove_source_shift=True)
    assert isinstance(err.value, ValueError)
    assert err.value.args[0] == ("'remove_source_shift' argument not "
                                 "compatible with 'reconvolve_stf'.")

    # No matter the dt, the first sample will always be set to the origin time.
    st = db.get_seismograms(
        source=source, receiver=receiver,
        components=('Z', 'N', 'E', 'R', 'T'), reconvolve_stf=True,
        remove_source_shift=False)
    for tr in st:
        assert tr.stats.starttime == origin_time

    st = db.get_seismograms(
        source=source, receiver=receiver,
        components=('Z', 'N', 'E', 'R', 'T'), reconvolve_stf=True,
        remove_source_shift=False, dt=0.1)
    for tr in st:
        assert tr.stats.starttime == origin_time

    st = db.get_seismograms(
        source=source, receiver=receiver,
        components=('Z', 'N', 'E', 'R', 'T'), reconvolve_stf=True,
        remove_source_shift=False, dt=1.0)
    for tr in st:
        assert tr.stats.starttime == origin_time


@pytest.mark.parametrize("db", DBS)
def test_remove_samples_at_end_for_lanczos(db):
    """
    Remove some samples at the end for the Lanczos resampling to avoid
    boundary effects.
    """
    db = InstaseisDB(db)

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
                              a_lanczos=1)
    for tr, tr_2 in zip(st, st_2):
        assert tr == tr_2

    # The original dt is a bit more then 24, so a dt of twelve should result
    # in three missing samples for a=1, and 5 for a=2.
    samples = int((st_2[0].stats.endtime - st_2[0].stats.starttime) / 12.0) + 1

    st_3 = db.get_seismograms(source=source, receiver=receiver,
                              remove_source_shift=True, dt=12,
                              a_lanczos=1)
    for tr in st_3:
        assert tr.stats.npts == samples - 3

    st_4 = db.get_seismograms(source=source, receiver=receiver,
                              remove_source_shift=True, dt=12,
                              a_lanczos=2)
    for tr in st_4:
        assert tr.stats.npts == samples - 5

    st_5 = db.get_seismograms(source=source, receiver=receiver,
                              remove_source_shift=True, dt=12,
                              a_lanczos=3)
    for tr in st_5:
        assert tr.stats.npts == samples - 7


@pytest.mark.parametrize("db", DBS)
def test_get_time_information(db):
    """
    Tests the get_seismogram_times() function. Also make sure it is
    consistent with the actually produces seismograms.
    """
    db = InstaseisDB(db)

    origin_time = obspy.UTCDateTime(2015, 1, 1, 1, 1)

    source = Source(latitude=4., longitude=3.0, depth_in_m=0,
                    m_rr=4.71e+17, m_tt=3.81e+17, m_pp=-4.74e+17,
                    m_rt=3.99e+17, m_rp=-8.05e+17, m_tp=-1.23e+17,
                    origin_time=origin_time)
    receiver = Receiver(latitude=10., longitude=20., depth_in_m=0)

    # First case is very simple.
    par = {"remove_source_shift": True,
           "dt": None,
           "a_lanczos": 5}
    tr = db.get_seismograms(source=source, receiver=receiver,
                            components=["Z"], **par)[0]
    times = get_seismogram_times(info=db.info, origin_time=origin_time, **par)

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
           "a_lanczos": 5}
    tr = db.get_seismograms(source=source, receiver=receiver,
                            components=["Z"], **par)[0]
    times = get_seismogram_times(info=db.info, origin_time=origin_time, **par)

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
           "a_lanczos": 2}
    tr = db.get_seismograms(source=source, receiver=receiver,
                            components=["Z"], **par)[0]
    times = get_seismogram_times(info=db.info, origin_time=origin_time, **par)

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
           "a_lanczos": 5}
    tr = db.get_seismograms(source=source, receiver=receiver,
                            components=["Z"], **par)[0]
    times = get_seismogram_times(info=db.info, origin_time=origin_time, **par)

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
           "a_lanczos": 1}
    tr = db.get_seismograms(source=source, receiver=receiver,
                            components=["Z"], **par)[0]
    times = get_seismogram_times(info=db.info, origin_time=origin_time, **par)

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
           "a_lanczos": 2}
    tr = db.get_seismograms(source=source, receiver=receiver,
                            components=["Z"], **par)[0]
    times = get_seismogram_times(info=db.info, origin_time=origin_time, **par)

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
           "a_lanczos": 7}
    tr = db.get_seismograms(source=source, receiver=receiver,
                            components=["Z"], **par)[0]
    times = get_seismogram_times(info=db.info, origin_time=origin_time, **par)

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
