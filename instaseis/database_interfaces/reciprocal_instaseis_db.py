#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Python library to extract seismograms from a set of wavefields generated by
AxiSEM.

:copyright:
    Martin van Driel (Martin@vanDriel.de), 2014
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
:license:
    GNU Lesser General Public License, Version 3 [non-commercial/academic use]
    (http://www.gnu.org/copyleft/lgpl.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import collections
import numpy as np

from .base_netcdf_instaseis_db import BaseNetCDFInstaseisDB
from . import mesh
from .. import rotations
from ..source import Source, ForceSource


class ReciprocalInstaseisDB(BaseNetCDFInstaseisDB):
    def __init__(self, db_path, netcdf_files, *args, **kwargs):
        """
        :param db_path: Path to the Instaseis Database containing
            subdirectories PZ and/or PX each containing a
            ``order_output.nc4`` file.
        :type db_path: str
        :param buffer_size_in_mb: Strain and displacement are buffered to
            avoid repeated disc access. Depending on the type of database
            and the number of components of the database, the total buffer
            memory can be up to four times this number. The optimal value is
            highly application and system dependent.
        :type buffer_size_in_mb: int, optional
        :param read_on_demand: Read several global fields on demand (faster
            initialization) or on initialization (slower
            initialization, faster in individual seismogram extraction,
            useful e.g. for finite sources, default).
        :type read_on_demand: bool, optional
        """
        BaseNetCDFInstaseisDB.__init__(self, db_path=db_path, *args, **kwargs)
        self._parse_meshes(netcdf_files)

    def _parse_meshes(self, files):
        if "PX" in files:
            px_file = files["PX"]
            x_exists = True
        else:
            x_exists = False
        if "PZ" in files:
            pz_file = files["PZ"]
            z_exists = True
        else:
            z_exists = False

        # full_parse will force the kd-tree to be built
        if x_exists and z_exists:
            px_m = mesh.Mesh(
                px_file, full_parse=True,
                strain_buffer_size_in_mb=self.buffer_size_in_mb,
                displ_buffer_size_in_mb=self.buffer_size_in_mb,
                read_on_demand=self.read_on_demand)
            pz_m = mesh.Mesh(
                pz_file, full_parse=False,
                strain_buffer_size_in_mb=self.buffer_size_in_mb,
                displ_buffer_size_in_mb=self.buffer_size_in_mb,
                read_on_demand=self.read_on_demand)
            self.parsed_mesh = px_m
        elif x_exists:
            px_m = mesh.Mesh(
                px_file, full_parse=True,
                strain_buffer_size_in_mb=self.buffer_size_in_mb,
                displ_buffer_size_in_mb=self.buffer_size_in_mb,
                read_on_demand=self.read_on_demand)
            pz_m = None
            self.parsed_mesh = px_m
        elif z_exists:
            px_m = None
            pz_m = mesh.Mesh(
                pz_file, full_parse=True,
                strain_buffer_size_in_mb=self.buffer_size_in_mb,
                displ_buffer_size_in_mb=self.buffer_size_in_mb,
                read_on_demand=self.read_on_demand)
            self.parsed_mesh = pz_m
        else:
            # Should not happen.
            raise NotImplementedError

        MeshCollection_bwd = collections.namedtuple(
            "MeshCollection_bwd", ["px", "pz"])
        self.meshes = MeshCollection_bwd(px=px_m, pz=pz_m)

        self._is_reciprocal = True

    def _get_data(self, source, receiver, components, rotmesh_s, rotmesh_phi,
                  rotmesh_z, id_elem, gll_point_ids, xi, eta, corner_points,
                  col_points_xi, col_points_eta, axis, eltype):
        # Collect data arrays and mu in a dictionary.
        data = {}

        mesh = self.parsed_mesh.f["Mesh"]

        # Get mu.
        if not self.read_on_demand:
            mesh_mu = self.parsed_mesh.mesh_mu
        else:
            mesh_mu = mesh["mesh_mu"]
        if self.info.dump_type == "displ_only":
            npol = self.info.spatial_order
            mu = mesh_mu[gll_point_ids[npol // 2, npol // 2]]
        else:
            # XXX: Is this correct?
            mu = mesh_mu[id_elem]
        data["mu"] = mu

        fac_1_map = {"N": np.cos,
                     "E": np.sin}
        fac_2_map = {"N": lambda x: - np.sin(x),
                     "E": np.cos}

        if isinstance(source, Source):
            if self.info.dump_type == 'displ_only':
                if axis:
                    G = self.parsed_mesh.G2
                    GT = self.parsed_mesh.G1T
                else:
                    G = self.parsed_mesh.G2
                    GT = self.parsed_mesh.G2T

            strain_x = None
            strain_z = None

            # Minor optimization: Only read if actually requested.
            if "Z" in components:
                if self.info.dump_type == 'displ_only':
                    strain_z = self._get_strain_interp(
                        self.meshes.pz, id_elem, gll_point_ids, G, GT,
                        col_points_xi, col_points_eta, corner_points,
                        eltype, axis, xi, eta)
                elif (self.info.dump_type == 'fullfields' or
                      self.info.dump_type == 'strain_only'):
                    strain_z = self._get_strain(self.meshes.pz, id_elem)

            if any(comp in components for comp in ['N', 'E', 'R', 'T']):
                if self.info.dump_type == 'displ_only':
                    strain_x = self._get_strain_interp(
                        self.meshes.px, id_elem, gll_point_ids, G, GT,
                        col_points_xi, col_points_eta, corner_points,
                        eltype, axis, xi, eta)
                elif (self.info.dump_type == 'fullfields' or
                      self.info.dump_type == 'strain_only'):
                    strain_x = self._get_strain(self.meshes.px, id_elem)

            mij = rotations \
                .rotate_symm_tensor_voigt_xyz_src_to_xyz_earth(
                    source.tensor_voigt, np.deg2rad(source.longitude),
                    np.deg2rad(source.colatitude))
            mij = rotations \
                .rotate_symm_tensor_voigt_xyz_earth_to_xyz_src(
                    mij, np.deg2rad(receiver.longitude),
                    np.deg2rad(receiver.colatitude))
            mij = rotations.rotate_symm_tensor_voigt_xyz_to_src(
                mij, rotmesh_phi)
            mij /= self.parsed_mesh.amplitude

            if "Z" in components:
                final = np.zeros(strain_z.shape[0], dtype="float64")
                for i in range(3):
                    final += mij[i] * strain_z[:, i]
                final += 2.0 * mij[4] * strain_z[:, 4]
                data["Z"] = final

            if "R" in components:
                final = np.zeros(strain_x.shape[0], dtype="float64")
                final -= strain_x[:, 0] * mij[0] * 1.0
                final -= strain_x[:, 1] * mij[1] * 1.0
                final -= strain_x[:, 2] * mij[2] * 1.0
                final -= strain_x[:, 4] * mij[4] * 2.0
                data["R"] = final

            if "T" in components:
                final = np.zeros(strain_x.shape[0], dtype="float64")
                final += strain_x[:, 3] * mij[3] * 2.0
                final += strain_x[:, 5] * mij[5] * 2.0
                data["T"] = final

            for comp in ["E", "N"]:
                if comp not in components:
                    continue

                fac_1 = fac_1_map[comp](rotmesh_phi)
                fac_2 = fac_2_map[comp](rotmesh_phi)

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

        elif isinstance(source, ForceSource):
            if self.info.dump_type != 'displ_only':
                raise ValueError("Force sources only in displ_only mode")

            if "Z" in components:
                displ_z = self._get_displacement(self.meshes.pz, id_elem,
                                                 gll_point_ids,
                                                 col_points_xi,
                                                 col_points_eta, xi, eta)

            if any(comp in components for comp in ['N', 'E', 'R', 'T']):
                displ_x = self._get_displacement(self.meshes.px, id_elem,
                                                 gll_point_ids,
                                                 col_points_xi,
                                                 col_points_eta, xi, eta)

            force = rotations.rotate_vector_xyz_src_to_xyz_earth(
                source.force_tpr, np.deg2rad(source.longitude),
                np.deg2rad(source.colatitude))
            force = rotations.rotate_vector_xyz_earth_to_xyz_src(
                force, np.deg2rad(receiver.longitude),
                np.deg2rad(receiver.colatitude))
            force = rotations.rotate_vector_xyz_to_src(
                force, rotmesh_phi)
            force /= self.parsed_mesh.amplitude

            if "Z" in components:
                final = np.zeros(displ_z.shape[0], dtype="float64")
                final += displ_z[:, 0] * force[0]
                final += displ_z[:, 2] * force[2]
                data["Z"] = final

            if "R" in components:
                final = np.zeros(displ_x.shape[0], dtype="float64")
                final += displ_x[:, 0] * force[0]
                final += displ_x[:, 2] * force[2]
                data["R"] = final

            if "T" in components:
                final = np.zeros(displ_x.shape[0], dtype="float64")
                final += displ_x[:, 1] * force[1]
                data["T"] = final

            for comp in ["E", "N"]:
                if comp not in components:
                    continue

                fac_1 = fac_1_map[comp](rotmesh_phi)
                fac_2 = fac_2_map[comp](rotmesh_phi)

                final = np.zeros(displ_x.shape[0], dtype="float64")
                final += displ_x[:, 0] * force[0] * fac_1
                final += displ_x[:, 1] * force[1] * fac_2
                final += displ_x[:, 2] * force[2] * fac_1
                if comp == "N":
                    final *= -1.0
                data[comp] = final

        else:
            raise NotImplementedError

        return data
