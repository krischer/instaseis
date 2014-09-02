#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Python interface to an AxiSEM database in a netCDF file.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
from __future__ import absolute_import

import collections
import numpy as np
from obspy.core import Stream, Trace
import os

from . import finite_elem_mapping
from . import mesh
from . import rotations
from . import sem_derivatives
from . import spectral_basis
from . import lanczos


MeshCollection = collections.namedtuple("MeshCollection", ["px", "pz"])

DEFAULT_MU = 32e9


class AxiSEMDB(object):
    def __init__(self, folder, buffer_size_in_mb=100, read_on_demand=True):
        self.folder = folder
        self.buffer_size_in_mb = buffer_size_in_mb
        self.read_on_demand = read_on_demand
        self._find_and_open_files()

    def _find_and_open_files(self):
        px = os.path.join(self.folder, "PX")
        pz = os.path.join(self.folder, "PZ")
        if not os.path.exists(px) and not os.path.exists(pz):
            raise ValueError(
                "Expecting the 'PX' or 'PZ' subfolders to be present.")
        px_file = os.path.join(px, "Data", "ordered_output.nc4")
        pz_file = os.path.join(pz, "Data", "ordered_output.nc4")
        x_exists, z_exists = False, False
        if os.path.exists(px_file):
            x_exists = True
        if os.path.exists(pz_file):
            z_exists = True

        # full_parse will force the kd-tree to be built
        if x_exists and z_exists:
            px_m = mesh.Mesh(px_file, full_parse=True,
                             buffer_size_in_mb=self.buffer_size_in_mb,
                             read_on_demand=self.read_on_demand)
            pz_m = mesh.Mesh(pz_file, full_parse=False,
                             buffer_size_in_mb=self.buffer_size_in_mb,
                             read_on_demand=self.read_on_demand)
            self.parsed_mesh = px_m
        elif x_exists:
            px_m = mesh.Mesh(px_file, full_parse=True,
                             buffer_size_in_mb=self.buffer_size_in_mb,
                             read_on_demand=self.read_on_demand)
            pz_m = None
            self.parsed_mesh = px_m
        elif z_exists:
            px_m = None
            pz_m = mesh.Mesh(pz_file, full_parse=True,
                             buffer_size_in_mb=self.buffer_size_in_mb,
                             read_on_demand=self.read_on_demand)
            self.parsed_mesh = pz_m
        else:
            raise ValueError("ordered_output.nc4 files must exist in the "
                             "PZ/Data and/or PX/Data subfolders")

        self.meshes = MeshCollection(px_m, pz_m)

    def get_seismograms(self, source, receiver, components=("Z", "N", "E"),
                        remove_source_shift=True, reconvolve_stf=False,
                        return_obspy_stream=True, dt=None, a_lanczos=5):

        rotmesh_s, rotmesh_phi, rotmesh_z = rotations.rotate_frame_rd(
            source.x * 1000.0, source.y * 1000.0, source.z * 1000.0,
            receiver.longitude, receiver.colatitude)

        nextpoints = self.parsed_mesh.kdtree.query([rotmesh_s, rotmesh_z], k=6)

        # Find the element containing the point of interest.
        mesh = self.parsed_mesh.f.groups["Mesh"]
        for idx in nextpoints[1]:
            corner_points = np.empty((4, 2), dtype="float64")

            if not self.read_on_demand:
                corner_point_ids = self.parsed_mesh.fem_mesh[idx][:4]
                eltype = self.parsed_mesh.eltypes[idx]
                corner_points[:, 0] = self.parsed_mesh.mesh_S[corner_point_ids]
                corner_points[:, 1] = self.parsed_mesh.mesh_Z[corner_point_ids]
            else:
                corner_point_ids = mesh.variables["fem_mesh"][idx][:4]
                eltype = mesh.variables["eltype"][idx]
                corner_points[:, 0] = mesh.variables["mesh_S"][corner_point_ids]
                corner_points[:, 1] = mesh.variables["mesh_Z"][corner_point_ids]

            isin, xi, eta = finite_elem_mapping.inside_element(
                rotmesh_s, rotmesh_z, corner_points, eltype,
                tolerance=1E-3)
            if isin:
                id_elem = idx
                break
        else:
            raise ValueError("Element not found")

        if not self.read_on_demand:
            gll_point_ids = self.parsed_mesh.sem_mesh[id_elem]
            axis = bool(self.parsed_mesh.axis[id_elem])
        else:
            gll_point_ids = mesh.variables["sem_mesh"][id_elem]
            axis = bool(mesh.variables["axis"][id_elem])

        if axis:
            G = self.parsed_mesh.G2
            GT = self.parsed_mesh.G1T
            col_points_xi = self.parsed_mesh.glj_points
            col_points_eta = self.parsed_mesh.gll_points
        else:
            G = self.parsed_mesh.G2
            GT = self.parsed_mesh.G2T
            col_points_xi = self.parsed_mesh.gll_points
            col_points_eta = self.parsed_mesh.gll_points

        strain_x = None
        strain_z = None

        # Minor optimization: Only read if actually requested.
        if "Z" in components:
            strain_z = self.__get_strain(self.meshes.pz, id_elem,
                                         gll_point_ids, G, GT, col_points_xi,
                                         col_points_eta,
                                         corner_points, eltype, axis, xi, eta)
        if "N" in components or "E" in components:
            strain_x = self.__get_strain(self.meshes.px, id_elem,
                                         gll_point_ids, G, GT, col_points_xi,
                                         col_points_eta, corner_points, eltype,
                                         axis, xi, eta)

        mij = rotations.rotate_symm_tensor_voigt_xyz_src_to_xyz_earth_1d(
            source.tensor_voigt, np.deg2rad(source.longitude),
            np.deg2rad(source.colatitude))
        mij = rotations.rotate_symm_tensor_voigt_xyz_earth_to_xyz_src_1d(
            mij, np.deg2rad(receiver.longitude),
            np.deg2rad(receiver.colatitude))
        mij = rotations.rotate_symm_tensor_voigt_xyz_to_src_1d(
            mij, rotmesh_phi)
        mij /= self.parsed_mesh.amplitude

        data = {}

        if "Z" in components:
            final = np.zeros(strain_z.shape[0], dtype="float64")
            for i in xrange(3):
                final += mij[i] * strain_z[:, i]
            final += 2.0 * mij[4] * strain_z[:, 4]
            data["Z"] = final

        ax_map = {"N": np.array([0.0, 1.0, 0.0]),
                  "E": np.array([0.0, 0.0, 1.0])}

        for comp in ["E", "N"]:
            if comp not in components:
                continue

            fac_1 = rotations.azim_factor_bw(rotmesh_phi, ax_map[comp], 2, 1)
            fac_2 = rotations.azim_factor_bw(rotmesh_phi, ax_map[comp], 2, 2)

            final = np.zeros(strain_x.shape[0], dtype="float64")
            final += strain_x[:, 0] * mij[0] * 1.0 * fac_1
            final += strain_x[:, 1] * mij[1] * 1.0 * fac_1
            final += strain_x[:, 2] * mij[2] * 1.0 * fac_1
            final += strain_x[:, 3] * mij[3] * 2.0 * fac_2
            final += strain_x[:, 4] * mij[4] * 2.0 * fac_1
            final += strain_x[:, 5] * mij[5] * 2.0 * fac_2
            if comp == "N":
                final *= -1.0
            data[comp] = final

        for comp in components:
            if remove_source_shift and not reconvolve_stf:
                data[comp] = data[comp][self.parsed_mesh.source_shift_samp:]
            elif reconvolve_stf:
                if source.dt is None or source.sliprate is None:
                    raise RuntimeError("source has no source time function")

                stf_deconv_f = np.fft.rfft(
                    self.sliprate, n=self.ndumps * 2)

                if abs((source.dt - self.dt) / self.dt) > 1e-7:
                    raise ValueError("dt of the source not compatible")

                stf_conv_f = np.fft.rfft(source.sliprate,
                                         n=self.ndumps * 2)

                if source.time_shift is not None:
                    stf_conv_f *= \
                        np.exp(- 1j * np.fft.rfftfreq(self.ndumps * 2)
                               * 2. * np.pi * source.time_shift /
                               self.dt)

                # TODO: double check wether a taper is needed at the end of the
                #       trace
                dataf = np.fft.rfft(data[comp], n=self.ndumps * 2)

                data[comp] = np.fft.irfft(
                    dataf * stf_conv_f / stf_deconv_f)[:self.ndumps]

            if dt is not None:
                data[comp] = lanczos.lanczos_resamp(
                    data[comp], self.parsed_mesh.dt, dt, a_lanczos)

        if return_obspy_stream:
            # Convert to an ObsPy Stream object.
            st = Stream()
            if dt is None:
                dt = self.parsed_mesh.dt
            band_code = self._get_band_code(dt)
            for comp in components:
                tr = Trace(data=data[comp],
                           header={"delta": dt,
                                   "station": receiver.name,
                                   "network": receiver.network,
                                   "channel": "%sX%s" % (band_code, comp)})
                st += tr
            return st
        else:
            npol = self.parsed_mesh.npol
            if not self.read_on_demand:
                mu = self.parsed_mesh.mesh_mu[gll_point_ids[npol/2, npol/2]]
            else:
                mu = mesh.variables["mesh_mu"][gll_point_ids[npol/2, npol/2]]
            return data, mu

    def get_seismograms_finite_source(self, sources, receiver,
                                      components=("Z", "N", "E"), dt=None,
                                      a_lanczos=5):
        data_summed = {}
        for source in sources:
            data, mu = self.get_seismograms(
                source, receiver, components, reconvolve_stf=True,
                return_obspy_stream=False)
            for comp in components:
                if comp in data_summed:
                    data_summed[comp] += data[comp] * mu / DEFAULT_MU
                else:
                    data_summed[comp] = data[comp] * mu / DEFAULT_MU

        if dt is not None:
            for comp in components:
                data_summed[comp] = lanczos.lanczos_resamp(
                    data_summed[comp], self.parsed_mesh.dt, dt, a_lanczos)

        # Convert to an ObsPy Stream object.
        st = Stream()
        if dt is None:
            dt = self.parsed_mesh.dt
        band_code = self._get_band_code(dt)
        for comp in components:
            tr = Trace(data=data_summed[comp],
                       header={"delta": dt,
                               "station": receiver.name,
                               "network": receiver.network,
                               "channel": "%sX%s" % (band_code, comp)})
            st += tr
        return st

    def _get_band_code(self, dt):
        """
        Figure out the channel band code. Done as in SPECFEM.
        """
        sr = 1.0 / dt
        if sr <= 0.001:
            band_code = "F"
        elif sr <= 0.004:
            band_code = "C"
        elif sr <= 0.0125:
            band_code = "H"
        elif sr <= 0.1:
            band_code = "B"
        elif sr <= 1:
            band_code = "M"
        else:
            band_code = "L"
        return band_code

    def __get_strain(self, mesh, id_elem, gll_point_ids, G, GT, col_points_xi,
                     col_points_eta, corner_points, eltype, axis, xi, eta):
        if id_elem not in mesh.strain_buffer:
            # Single precision in the NetCDF files but the later interpolation
            # routines require double precision. Assignment to this array will
            # force a cast.
            utemp = np.zeros((mesh.ndumps, mesh.npol + 1, mesh.npol + 1, 3),
                             dtype=np.float64, order="F")

            mesh_dict = mesh.f.groups["Snapshots"].variables

            # Load displacement from all GLL points.
            for i, var in enumerate(["disp_s", "disp_p", "disp_z"]):
                if var not in mesh_dict:
                    continue
                temp = mesh_dict[var][:, gll_point_ids.flatten()]
                for ipol in xrange(mesh.npol + 1):
                    for jpol in xrange(mesh.npol + 1):
                        utemp[:, jpol, ipol, i] = temp[:, ipol * 5 + jpol]

            strain_fct_map = {
                "monopole": sem_derivatives.strain_monopole_td,
                "dipole": sem_derivatives.strain_dipole_td,
                "quadpole": sem_derivatives.strain_quadpole_td}

            strain = strain_fct_map[mesh.excitation_type](
                utemp, G, GT, col_points_xi, col_points_eta, mesh.npol,
                mesh.ndumps, corner_points, eltype, axis)

            mesh.strain_buffer.add(id_elem, strain)
        else:
            strain = mesh.strain_buffer.get(id_elem)

        final_strain = np.empty((strain.shape[0], 6), order="F")

        for i in xrange(6):
            final_strain[:, i] = spectral_basis.lagrange_interpol_2D_td(
                col_points_xi, col_points_eta, strain[:, :, :, i], xi, eta)

        if not mesh.excitation_type == "monopole":
            final_strain[:, 3] *= -1.0
            final_strain[:, 5] *= -1.0

        return final_strain

    @property
    def dt(self):
        return self.parsed_mesh.dt

    @property
    def ndumps(self):
        return self.parsed_mesh.ndumps

    @property
    def background_model(self):
        return self.parsed_mesh.background_model

    @property
    def sliprate(self):
        return self.parsed_mesh.stf_d_norm

    @property
    def slip(self):
        return self.parsed_mesh.stf
