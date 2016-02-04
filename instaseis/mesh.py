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

import h5py
import numpy as np
from obspy import UTCDateTime
from scipy.spatial import cKDTree

# Python 2.6 compat.
try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict


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

    def add(self, key, value):
        """
        Add an item to the buffer and make sure that the buffer does not exceed
        the maximum size in memory.
        """
        self._buffer[key] = value
        # Assuming value is a numpy array
        self._total_size += value.nbytes

        # Remove existing values, until the size limit is fulfilled.
        while self._total_size > self._max_size_in_bytes:
            _, v = self._buffer.popitem(last=False)
            self._total_size -= v.nbytes

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
        self.strain_buffer = Buffer(strain_buffer_size_in_mb)
        self.displ_buffer = Buffer(displ_buffer_size_in_mb)
        self._create_mesh_dict()

    def _create_mesh_dict(self):
        """
        Creates a dictionary of the required data.

        Either creates memory maps if possible with the data or just points
        towards the HDF5 groups.
        """
        if "unrolled_snapshots" in self.f:
            ds = self.f["unrolled_snapshots"]
            offset = ds.id.get_offset()
            if ds.chunks is None and ds.compression is None and \
                    offset is not None:
                self.mesh_dict = {}
                try:
                    self.mesh_dict["unrolled_snapshots"] = \
                        np.memmap(self.filename, mode='r', shape=ds.shape,
                                  offset=offset, dtype=ds.dtype, order="C")
                except:
                    self.mesh_dict["unrolled_snapshots"] = ds
            else:
                raise NotImplementedError("Unrolled snapshots must not be "
                                          "chunked in the netCDF file.")
            return
        elif "merged_snapshots" in self.f:
            ds = self.f["merged_snapshots"]
            offset = ds.id.get_offset()
            if ds.chunks is None and ds.compression is None and \
                            offset is not None:
                self.mesh_dict = {}
                try:
                    self.mesh_dict["merged_snapshots"] = \
                        np.memmap(self.filename, mode='r', shape=ds.shape,
                                  offset=offset, dtype=ds.dtype, order="C")
                except:
                    self.mesh_dict["merged_snapshots"] = ds
            else:
                raise NotImplementedError("Merged snapshots must not be "
                                          "chunked in the netCDF file.")
            return

        mesh_dict = {}

        def get_time_axis(ds):
            """Helper function to determine the time axis of the mesh."""
            if ds.shape[0] == ds.shape[1]:
                raise NotImplementedError("Both dimensions in the dataset "
                                          "are identical. This is currently "
                                          "not supported.")
            elif ds.shape[0] == self.ndumps:
                return 0
            elif ds.shape[1] == self.ndumps:
                return 1
            else:
                raise ValueError("Could not determine the time axis in the "
                                 "2D array. It has an incompatible shape.")

        for key, value in self.f["Snapshots"].items():
            offset = value.id.get_offset()
            time_axis = get_time_axis(value)
            if value.chunks is None and value.compression is None and \
                    offset is not None:
                try:
                    mesh_dict[key] = \
                        np.memmap(self.filename, mode='r', shape=value.shape,
                                  offset=offset, dtype=value.dtype, order="C")
                except:
                    mesh_dict[key] = value
            else:
                if time_axis != 0:
                    raise NotImplementedError(
                        "The current implementation requires chunked "
                        "netCDF files to have the time as the first axis and "
                        "the gll points as the second.")
                mesh_dict[key] = value

            mesh_dict[key].time_axis = time_axis

        self.mesh_dict = mesh_dict

    def _parse(self, full_parse=False):
        # Cheap sanity check. No need to parse the rest.
        self.dump_type = \
            self.f.attrs["dump type "
                         "(displ_only, displ_velo, fullfields)"].decode()
        if (self.dump_type != "displ_only" and
                self.dump_type != "fullfields" and
                self.dump_type != "strain_only"):
            raise NotImplementedError

        self.npol = self.f.attrs["npol"][0]

        try:
            self.file_version = self.f.attrs["file version"][0]
        except AttributeError:
            raise ValueError("Database file so old that it does not even have "
                             "a version number. Please update AxiSEM or get "
                             "new databases.")

        if self.file_version < self.MIN_FILE_VERSION:
            raise ValueError("Database file too old. Minimum file version "
                             "expected: %d, found: %d." %
                             (self.MIN_FILE_VERSION, self.file_version))

        self.ndumps = self.f.attrs["number of strain dumps"][0]
        self.excitation_type = self.f.attrs["excitation type"].decode()

        # The rest is not needed for every mesh.

        if full_parse is False:
            return

        # Read some basic information to have easier access later on.
        self.source_type = self.f.attrs["source type"].decode()
        self.amplitude = self.f.attrs["scalar source magnitude"][0]
        self.dt = self.f.attrs["strain dump sampling rate in sec"][0]
        self.source_shift = self.f.attrs["source shift factor in sec"][0]
        self.source_shift_samp = self.f.attrs[
            "source shift factor for deltat_coarse"][0]

        possible_stf_groups = ["Surface", "Snapshots"]
        found_stf = False
        for g in possible_stf_groups:
            if g not in self.f:
                continue
            group = self.f[g]

            if "stf_d_dump" not in group or \
                    "stf_dump" not in group:
                continue

            stf_d_dump = group["stf_d_dump"][:]
            stf_dump = group["stf_dump"][:]

            if np.ma.is_masked(stf_d_dump) or \
                    np.ma.is_masked(stf_dump) or \
                    np.isnan(np.sum(stf_d_dump)) or \
                    np.isnan(np.sum(stf_dump)):
                continue

            found_stf = True
            break

        if found_stf is False:
            raise ValueError("Could not extract valid slip and sliprates "
                             "from the netcdf files.")

        self.stf_d = stf_d_dump
        self.stf = stf_dump

        self.stf_d_norm = self.stf_d / self.amplitude
        self.stf_norm = self.stf / self.amplitude

        self.npoints = self.f.attrs["npoints"][0]

        self.background_model = self.f.attrs["background model"].decode()
        if self.file_version >= 8:
            self.external_model_name = \
                self.f.attrs["external model name"].decode()
        else:
            if self.background_model == 'external':
                self.external_model_name = 'unknown'
            else:
                self.external_model_name = ''
        self.attenuation = bool(self.f.attrs["attenuation"][0])
        self.planet_radius = self.f.attrs["planet radius"][0] * 1e3
        self.dominant_period = self.f.attrs["dominant source period"][0]
        self.axisem_version = self.f.attrs["git commit hash"].decode()
        self.creation_time = UTCDateTime(self.f.attrs["datetime"].decode())
        self.axisem_compiler = "%s %s" % (
            self.f.attrs["compiler brand"].decode(),
            self.f.attrs["compiler version"].decode())
        self.axisem_user = "%s on %s" % (
            self.f.attrs["user name"].decode(),
            self.f.attrs["host name"].decode())

        self.kwf_rmin = self.f.attrs["kernel wavefield rmin"][0]
        self.kwf_rmax = self.f.attrs["kernel wavefield rmax"][0]
        self.kwf_colatmin = self.f.attrs["kernel wavefield colatmin"][0]
        self.kwf_colatmax = self.f.attrs["kernel wavefield colatmax"][0]
        self.time_scheme = self.f.attrs["time scheme"].decode()
        self.source_depth = self.f.attrs["source depth in km"][0]
        self.stf_kind = self.f.attrs["source time function"].decode()

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
