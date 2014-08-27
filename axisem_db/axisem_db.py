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
import obspy
from obspy.core import Stream, Trace
import os

from . import finite_elem_mapping
from . import mesh
from . import rotations
from . import sem_derivatives
from . import spectral_basis


MeshCollection = collections.namedtuple("MeshCollection", ["px", "pz"])


class AxiSEMDB(object):
    def __init__(self, folder):
        self.folder = folder
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
            px_m = mesh.Mesh(px_file, full_parse=True)
            pz_m = mesh.Mesh(pz_file, full_parse=False)
            self.parsed_mesh = px_m
        elif x_exists:
            px_m = mesh.Mesh(px_file, full_parse=True)
            pz_m = None
            self.parsed_mesh = px_m
        elif z_exists:
            px_m = None
            pz_m = mesh.Mesh(pz_file, full_parse=True)
            self.parsed_mesh = pz_m
        else:
            raise ValueError("ordered_output.nc4 files must exist in the "
                             "PZ/Data and/or PX/Data subfolders")

        self.meshes = MeshCollection(px_m, pz_m)

    def get_seismograms(self, source, receiver, components=("Z", "N", "E"),
                        remove_source_shift=True, reconvolve_stf=False):

        rotmesh_s, rotmesh_phi, rotmesh_z = rotations.rotate_frame_rd(
            source.x * 1000.0, source.y * 1000.0, source.z * 1000.0,
            receiver.longitude, receiver.colatitude)

        nextpoints = self.parsed_mesh.kdtree.query([rotmesh_s, rotmesh_z], k=6)

        # Find the element containing the point of interest.
        mesh = self.parsed_mesh.f.groups["Mesh"]
        for idx in nextpoints[1]:
            fem_mesh = mesh.variables["fem_mesh"]
            corner_point_ids = fem_mesh[idx][:4]
            eltype = mesh.variables["eltype"][idx]

            corner_points = []
            for i in corner_point_ids:
                corner_points.append((
                    mesh.variables["mesh_S"][i],
                    mesh.variables["mesh_Z"][i]
                ))
            corner_points = np.array(corner_points, dtype=np.float64)

            isin, xi, eta = finite_elem_mapping.inside_element(
                rotmesh_s, rotmesh_z, corner_points, eltype,
                tolerance=1E-3)
            if isin:
                id_elem = idx
                break
        else:
            raise ValueError("Element not found")

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
            strain_z = self.__get_strain(self.meshes.pz, gll_point_ids, G, GT,
                                         col_points_xi, col_points_eta,
                                         corner_points, eltype, axis, xi, eta)
        if "N" in components or "E" in components:
            strain_x = self.__get_strain(self.meshes.px, gll_point_ids, G, GT,
                                         col_points_xi, col_points_eta,
                                         corner_points, eltype, axis, xi, eta)

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

        # Convert to an ObsPy Stream object.
        st = Stream()
        dt = self.parsed_mesh.dt
        band_code = self._get_band_code(dt)

        for comp in components:
            
            if remove_source_shift and not reconvolve_stf:
                data[comp] = data[comp][self.parsed_mesh.source_shift_samp:]
            elif reconvolve_stf:
                stf_deconv_f = np.fft.rfft(self.get_sliprate(), n=self.get_ndumps() * 2)

                if abs((source.dt - self.get_dt()) / self.get_dt()) > 1e-7:
                    raise ValueError("dt of the source not compatible")

                stf_conv_f = np.fft.rfft(source.sliprate, n=self.get_ndumps() * 2)

                # TODO: double check wether a taper is needed at the end of the
                #       trace
                dataf = np.fft.rfft(data[comp], n=self.get_ndumps() * 2)

                data[comp] = np.fft.irfft(dataf * stf_conv_f / stf_deconv_f)[:self.get_ndumps()]

            tr = Trace(data=data[comp],
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

    def __get_strain(self, mesh, gll_point_ids, G, GT, col_points_xi,
                     col_points_eta, corner_points, eltype, axis, xi, eta):

        # Single precision in the NetCDF files but the later interpolation
        # routines require double precision. Assignment to this array will
        # force a cast.
        utemp = np.zeros((mesh.ndumps, mesh.npol + 1, mesh.npol + 1, 3),
                         dtype=np.float64, order="F")

        mesh_dict = mesh.f.groups["Snapshots"].variables

        # Load displacement from all GLL points.
        for ipol in xrange(mesh.npol + 1):
            for jpol in xrange(mesh.npol + 1):
                start_chunk = gll_point_ids[ipol, jpol]

                for i, var in enumerate(["disp_s", "disp_p", "disp_z"]):
                    if var not in mesh_dict:
                        continue
                    # Interesting indexing once again...but consistent with
                    # the Fortran output.
                    utemp[:, jpol, ipol, i] = \
                        mesh_dict[var][:, start_chunk]

        strain_fct_map = {
            "monopole": sem_derivatives.strain_monopole_td,
            "dipole": sem_derivatives.strain_dipole_td,
            "quadpole": sem_derivatives.strain_quadpole_td}

        strain = strain_fct_map[mesh.excitation_type](
            utemp, G, GT, col_points_xi, col_points_eta, mesh.npol,
            mesh.ndumps, corner_points, eltype, axis)

        final_strain = np.empty((strain.shape[0], 6))

        for i in xrange(6):
            final_strain[:, i] = spectral_basis.lagrange_interpol_2D_td(
                col_points_xi, col_points_eta, strain[:, :, :, i], xi, eta)
        final_strain[:, 3] *= -1.0
        final_strain[:, 5] *= -1.0

        return final_strain

    def get_dt(self):
        return self.parsed_mesh.dt

    def get_ndumps(self):
        return self.parsed_mesh.ndumps

    def get_background_model(self):
        return self.parsed_mesh.background_model

    def get_sliprate(self):
        return self.parsed_mesh.stf_d_norm

    def get_slip(self):
        return self.parsed_mesh.stf
