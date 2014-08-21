#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mesh object also taking care of opening and closing the netCDF files.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
from __future__ import absolute_import

import netCDF4
import numpy as np
from scipy.spatial import cKDTree

from . import spectral_basis


class Mesh(object):
    def __init__(self, filename, full_parse=False):
        self.f = netCDF4.Dataset(filename, "r", format="NETCDF4")
        self._parse(full_parse=full_parse)

    def __del__(self):
        try:
            self.f.close()
        except:
            pass

    def _parse(self, full_parse=False):
        # Cheap sanity check. No need to parse the rest.
        self.dump_type = \
            getattr(self.f, "dump type (displ_only, displ_velo, fullfields)")
        if self.dump_type != "displ_only":
            raise NotImplementedError

        self.npol = self.f.npol
        self.ndumps = getattr(self.f, "number of strain dumps")
        self.chunks = \
            self.f.groups["Snapshots"].variables.values()[0].chunking()
        self.excitation_type = getattr(self.f, "excitation type")

        # The rest is not needed for every mesh.

        if full_parse is False:
            return

        # Read some basic information to have easier access later on.
        self.source_type = getattr(self.f, "source type")
        self.amplitude = getattr(self.f, "scalar source magnitude")
        self.dt = getattr(self.f, "strain dump sampling rate in sec")
        self.npoints = self.f.npoints
        self.compression_level = \
            self.f.groups["Snapshots"].variables["disp_s"]\
            .filters()["complevel"]

        self.gll_points = spectral_basis.zelegl(self.npol)
        self.glj_points = spectral_basis.zemngl2(self.npol)
        self.G0, self.G1 = spectral_basis.def_lagrange_derivs_glj(self.npol)
        self.G2 = spectral_basis.def_lagrange_derivs_gll(self.npol)
        self.G1T = np.require(self.G1.transpose(),
                              requirements=["F_CONTIGUOUS"])
        self.G2T = np.require(self.G2.transpose(),
                              requirements=["F_CONTIGUOUS"])

        # Build a kdtree of the element midpoints.
        self.s_mp = self.f.groups["Mesh"].variables["mp_mesh_S"]
        self.z_mp = self.f.groups["Mesh"].variables["mp_mesh_Z"]

        self.mesh = np.empty((self.s_mp.shape[0], 2), dtype=self.s_mp.dtype)
        self.mesh[:, 0] = self.s_mp[:]
        self.mesh[:, 1] = self.z_mp[:]

        self.kdtree = cKDTree(data=self.mesh)

    def get_n_closests_points(self, s, z, n=6):
        _, idx = self.kdtree.query([s, z], k=6)
        return self.mesh[idx]
