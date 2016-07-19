#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mesh object also taking care of opening and closing the netCDF files.

Please note that this module actually uses h5py instead of the Python
netcdf4 library to read the files. This enables us to skip one layer of
software. E.g.

* HDF5 -> C netCDF -> Python netcdf

instead of

* HDF5 -> h5py


:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
    Martin van Driel (Martin@vanDriel.de), 2014
:license:
    GNU Lesser General Public License, Version 3 [non-commercial/academic use]
    (http://www.gnu.org/copyleft/lgpl.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from collections import OrderedDict

import h5py
import numpy as np
from obspy import UTCDateTime
from scipy.spatial import cKDTree


class Buffer(object):
    """
    A simple memory-limited buffer with a dictionary-like interface.
    Implemented as a kind of priority queue where priority is highest for
    recently accessed items. Thus the "stalest" items are removed first once
    the memory limit it reached.
    """
    def __init__(self, max_size_in_mb=100):
        self._max_size_in_bytes = max_size_in_mb * 1024 ** 2
        self._total_size = 0
        self._buffer = OrderedDict()
        self._hits = 0
        self._fails = 0

    def __contains__(self, key):
        contains = key in self._buffer
        if contains:
            self._hits += 1
        else:
            self._fails += 1
        return contains

    def get(self, key):
        """
        Return an item from the buffer and move it to the end, so it is removed
        last.
        """
        value = self._buffer.pop(key)
        self._buffer[key] = value
        return value

    def _get_nbytes(self, value):
        # Works with single arrays and iterables of arrays.
        try:
            return value.nbytes
        except:
            return sum(_i.nbytes for _i in value if _i is not None)

    def add(self, key, value):
        """
        Add an item to the buffer and make sure that the buffer does not exceed
        the maximum size in memory.
        """
        self._buffer[key] = value
        # Assuming value is a numpy array
        self._total_size += self._get_nbytes(value)

        # Remove existing values, until the size limit is fulfilled.
        while self._total_size > self._max_size_in_bytes:
            _, v = self._buffer.popitem(last=False)
            self._total_size -= self._get_nbytes(v)

    def get_size_mb(self):
        return float(self._total_size) / 1024 ** 2

    @property
    def efficiency(self):
        """
        Return the fraction of calls to the __contains__() routine that
        returned True.
        """
        if (self._hits + self._fails) == 0:
            return 0.0
        else:
            return float(self._hits) / float(self._hits + self._fails)


def get_time_axis(ds, ndumps):
    """
    Helper function to determine the time axis of the mesh.
    """
    if ds.shape[0] == ds.shape[1]:
        raise NotImplementedError("Both dimensions in the dataset "
                                  "are identical. This is currently "
                                  "not supported.")
    elif ds.shape[0] == ndumps:
        return 0
    elif ds.shape[1] == ndumps:
        return 1
    else:  # pragma: no cover
        raise ValueError("Could not determine the time axis in the "
                         "2D array. It has an incompatible shape.")


class Mesh(object):
    """
    A class to handle the actual netCDF files written by AxiSEM.
    """
    # Minimal acceptable version of the netCDF database files.
    MIN_FILE_VERSION = 7

    def __init__(self, filename, full_parse=False,
                 strain_buffer_size_in_mb=0, displ_buffer_size_in_mb=0,
                 read_on_demand=True):
        self.f = h5py.File(filename, "r")
        self.filename = filename
        self.read_on_demand = read_on_demand
        self._parse(full_parse=full_parse)
        self._find_time_axis()
        self.strain_buffer = Buffer(strain_buffer_size_in_mb)
        self.displ_buffer = Buffer(displ_buffer_size_in_mb)

    def _get_str_attr(self, name):
        attr = self.f.attrs[name]
        if isinstance(attr, np.ndarray):
            attr = attr[0]
        try:
            return attr.decode()
        except:
            return attr

    def _find_time_axis(self):
        # Merged databases are always the same and don't have a Snapshots key.
        if "Snapshots" not in self.f:
            return

        self.time_axis = {}
        for key, value in self.f["Snapshots"].items():
            if "stf" not in key:
                self.time_axis[key] = get_time_axis(value, self.ndumps)
        else:
            return
        raise NotImplementedError  # pragma: no cover

    def _parse(self, full_parse=False):
        self.dump_type = self._get_str_attr(
            "dump type (displ_only, displ_velo, fullfields)")
        if (self.dump_type != "displ_only" and
                self.dump_type != "fullfields" and
                self.dump_type != "strain_only"):
            raise NotImplementedError

        self.npol = self.f.attrs["npol"][0]

        try:
            self.file_version = self.f.attrs["file version"][0]
        except AttributeError:  # pragma: no cover
            raise ValueError("Database file so old that it does not even have "
                             "a version number. Please update AxiSEM or get "
                             "new databases.")

        if self.file_version < self.MIN_FILE_VERSION:  # pragma: no cover
            raise ValueError("Database file too old. Minimum file version "
                             "expected: %d, found: %d." %
                             (self.MIN_FILE_VERSION, self.file_version))

        self.ndumps = self.f.attrs["number of strain dumps"][0]
        self.excitation_type = self._get_str_attr("excitation type")

        # The rest is not needed for every mesh.

        if full_parse is False:
            return

        # Read some basic information to have easier access later on.
        self.source_type = self._get_str_attr("source type")
        self.amplitude = self.f.attrs["scalar source magnitude"][0]
        self.dt = self.f.attrs["strain dump sampling rate in sec"][0]
        self.source_shift = self.f.attrs["source shift factor in sec"][0]
        self.source_shift_samp = self.f.attrs[
            "source shift factor for deltat_coarse"][0]

        # Search /Snapshots first - then the root group and then the legacy
        # /Surface group.
        possible_stf_groups = ["Snapshots", "/", "Surface"]
        found_stf = False
        for g in possible_stf_groups:
            if g not in self.f:  # pragma: no cover
                continue
            group = self.f[g]

            if "stf_d_dump" not in group or \
                    "stf_dump" not in group:  # pragma: no cover
                continue

            stf_d_dump = group["stf_d_dump"][:]
            stf_dump = group["stf_dump"][:]

            if np.ma.is_masked(stf_d_dump) or \
                    np.ma.is_masked(stf_dump) or \
                    np.isnan(np.sum(stf_d_dump)) or \
                    np.isnan(np.sum(stf_dump)):  # pragma: no cover
                continue

            found_stf = True
            break

        if found_stf is False:  # pragma: no cover
            raise ValueError("Could not extract valid slip and sliprates "
                             "from the netcdf files.")

        self.stf_d = stf_d_dump
        self.stf = stf_dump

        self.stf_d_norm = self.stf_d / self.amplitude
        self.stf_norm = self.stf / self.amplitude

        self.npoints = self.f.attrs["npoints"][0]

        self.background_model = self._get_str_attr("background model")
        if self.file_version >= 8:  # pragma: no cover
            self.external_model_name = \
                self._get_str_attr("external model name")
        else:
            if self.background_model == 'external':  # pragma: no cover
                self.external_model_name = 'unknown'
            else:
                self.external_model_name = ''
        self.attenuation = bool(self.f.attrs["attenuation"][0])
        self.planet_radius = self.f.attrs["planet radius"][0] * 1e3
        self.dominant_period = self.f.attrs["dominant source period"][0]
        self.axisem_version = self._get_str_attr("git commit hash")
        self.creation_time = UTCDateTime(self._get_str_attr("datetime"))
        self.axisem_compiler = "%s %s" % (
            self._get_str_attr("compiler brand"),
            self._get_str_attr("compiler version"))
        self.axisem_user = "%s on %s" % (
            self._get_str_attr("user name"),
            self._get_str_attr("host name"))

        self.kwf_rmin = self.f.attrs["kernel wavefield rmin"][0]
        self.kwf_rmax = self.f.attrs["kernel wavefield rmax"][0]
        self.kwf_colatmin = self.f.attrs["kernel wavefield colatmin"][0]
        self.kwf_colatmax = self.f.attrs["kernel wavefield colatmax"][0]
        self.time_scheme = self._get_str_attr("time scheme")
        self.source_depth = self.f.attrs["source depth in km"][0]
        self.stf_kind = self._get_str_attr("source time function")

        if self.dump_type == "displ_only":
            self.gll_points = self.f["Mesh"]["gll"][:]
            self.glj_points = self.f["Mesh"]["glj"][:]
            self.G0 = self.f["Mesh"]["G0"][:]
            self.G1 = self.f["Mesh"]["G1"][:].T
            self.G2 = self.f["Mesh"]["G2"][:].T

            self.G1T = np.require(self.G1.transpose(),
                                  requirements=["F_CONTIGUOUS"])
            self.G2T = np.require(self.G2.transpose(),
                                  requirements=["F_CONTIGUOUS"])

            # Build a kdtree of the element midpoints.
            self.s_mp = self.f["Mesh"]["mp_mesh_S"]
            self.z_mp = self.f["Mesh"]["mp_mesh_Z"]

            self.mesh = np.empty((self.s_mp.shape[0], 2),
                                 dtype=self.s_mp.dtype)
            self.mesh[:, 0] = self.s_mp[:]
            self.mesh[:, 1] = self.z_mp[:]

            self.kdtree = cKDTree(data=self.mesh)

            # Store some more index types in memory. While this increases
            # memory use it should be acceptable and result in much less netCDF
            # reads.
            if not self.read_on_demand:
                self.fem_mesh = self.f["Mesh"]["fem_mesh"][:]
                self.eltypes = self.f["Mesh"]["eltype"][:]
                self.mesh_S = self.f["Mesh"]["mesh_S"][:]
                self.mesh_Z = self.f["Mesh"]["mesh_Z"][:]
                self.sem_mesh = self.f["Mesh"]["sem_mesh"][:]
                self.axis = self.f["Mesh"]["axis"][:]
                self.mesh_mu = self.f["Mesh"]["mesh_mu"][:]

        elif self.dump_type == "fullfields" or self.dump_type == "strain_only":
            # Build a kdtree of the stored gll points.
            self.mesh_S = self.f["Mesh"]["mesh_S"]
            self.mesh_Z = self.f["Mesh"]["mesh_Z"]

            self.mesh = np.empty((self.mesh_S.shape[0], 2),
                                 dtype=self.mesh_S.dtype)
            self.mesh[:, 0] = self.mesh_S[:]
            self.mesh[:, 1] = self.mesh_Z[:]

            self.kdtree = cKDTree(data=self.mesh)

            if not self.read_on_demand:
                self.mesh_mu = self.f["Mesh"]["mesh_mu"][:]
