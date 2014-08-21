import netCDF4
import numpy as np
from scipy.spatial import cKDTree
import os

import finite_elem_mapping
import rotations
import sem_derivatives
from source import Source, Receiver
import spectral_basis


def rotate_frame_rd(x, y, z, phi, theta):
    phi = np.deg2rad(phi)
    theta = np.deg2rad(theta)
    # first rotation (longitude)
    xp_cp = x * np.cos(phi) + y * np.sin(phi)
    yp_cp = -x * np.sin(phi) + y * np.cos(phi)
    zp_cp = z

    # second rotation (colat)
    xp = xp_cp * np.cos(theta) - zp_cp * np.sin(theta)
    yp = yp_cp
    zp = xp_cp * np.sin(theta) + zp_cp * np.cos(theta)

    srd = np.sqrt(xp ** 2 + yp ** 2)
    zrd = zp
    phi_cp = np.arctan2(yp, xp)
    if phi_cp < 0.0:
        phird = 2.0 * np.pi + phi_cp
    else:
        phird = phi_cp
    return srd, phird, zrd


def test_rotate_frame_rd():
    s, phi, z = rotate_frame_rd(
        x=9988.6897343821470, y=0.0, z=6358992.1548998145, phi=74.494,
        theta=47.3609999)
    assert abs(s - 4676105.76848060) < 1E-2
    assert abs(phi - 3.14365101866993) < 1E-5
    assert abs(z - 4309398.5475913) < 1E-2


class Mesh(object):
    def __init__(self, hdf5_obj):
        self.f = hdf5_obj
        self._parse()
        self.displ_varnamelist = ('disp_s', 'disp_p', 'disp_z')

    def _parse(self):
        self.dump_type = \
            getattr(self.f, "dump type (displ_only, displ_velo, fullfields)")
        self.source_type = getattr(self.f, "source type")
        self.excitation_type = getattr(self.f, "excitation type")

        if self.dump_type != "displ_only":
            raise NotImplementedError

        self.npol = self.f.npol
        self.amplitude = getattr(self.f, "scalar source magnitude")
        self.ndumps = getattr(self.f, "number of strain dumps")
        self.npoints = self.f.npoints
        self.chunks = \
            self.f.groups["Snapshots"].variables["disp_s"].chunking()
        self.compression_level = \
            self.f.groups["Snapshots"].variables["disp_s"]\
            .filters()["complevel"]

        self.gll_points = spectral_basis.zelegl(self.npol)
        self.glj_points = spectral_basis.zemngl2(self.npol)
        self.G0, self.G1 = spectral_basis.def_lagrange_derivs_glj(self.npol)
        self.G2 = spectral_basis.def_lagrange_derivs_gll(self.npol)
        self.G1T = np.require(self.G1.transpose(),
                              requirements=["F_CONTIGUOUS"])
        # This is a bit weird but something about C and F memory order causes
        # this confusin...
        self.G2T = np.require(self.G2.transpose(),
                              requirements=["F_CONTIGUOUS"])

        self.s_mp = self.f.groups["Mesh"].variables["mp_mesh_S"]
        self.z_mp = self.f.groups["Mesh"].variables["mp_mesh_Z"]

        self.mesh = np.empty((self.s_mp.shape[0], 2), dtype=self.s_mp.dtype)
        self.mesh[:, 0] = self.s_mp[:]
        self.mesh[:, 1] = self.z_mp[:]

        self.kdtree = cKDTree(data=self.mesh)

    def get_n_closests_points(self, s, z, n=6):
        _, idx = self.kdtree.query([s, z], k=6)
        return self.mesh[idx]


def load_strain_point_interp(mesh, gll_point_ids, xi, eta, model_param,
                             corner_points, eltype, axis, id_elem):
    pass


class AxiSEMDB(object):
    def __init__(self, folder):
        self.folder = folder
        self.__files = {}
        self._meshes = {}
        self._find_and_open_files()

    def _find_and_open_files(self):
        px = os.path.join(self.folder, "PX")
        pz = os.path.join(self.folder, "PZ")
        if not os.path.exists(px) or not os.path.exists(pz):
            raise ValueError(
                "Expecting the 'PX' and 'PZ' subfolders to be present.")
        px_file = os.path.join(px, "Data", "ordered_output.nc4")
        pz_file = os.path.join(pz, "Data", "ordered_output.nc4")
        if not os.path.exists(px_file) or not os.path.exists(pz_file):
            raise ValueError("ordered_output.nc4 files must exist in the "
                             "PZ/Data and PX/Data subfolders")

        self.__files["px"] = netCDF4.Dataset(px_file, "r", format="NETCDF4")
        self.__files["pz"] = netCDF4.Dataset(pz_file, "r", format="NETCDF4")
        self._meshes["px"] = Mesh(self.__files["px"])
        self._meshes["pz"] = Mesh(self.__files["pz"])

    def __del__(self):
        for file_object in self.__files.items():
            try:
                file_object.close()
            except:
                pass

    def get_seismogram(self, source, receiver, component):
        rotmesh_s, rotmesh_phi, rotmesh_z = rotate_frame_rd(
            source.x * 1000.0, source.y * 1000.0, source.z * 1000.0,
            receiver.longitude, receiver.colatitude)

        nextpoints = self._meshes["px"].kdtree.query([rotmesh_s, rotmesh_z],
                                                     k=6)

        mesh = self.__files["px"].groups["Mesh"]
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

        # Get the ids of the GLL points.
        gll_point_ids = mesh.variables["sem_mesh"][id_elem]
        axis = bool(mesh.variables["axis"][id_elem])

        if component == "N":
            mesh = self._meshes["px"]
            if mesh.dump_type.strip() != "displ_only":
                raise NotImplementedError

            if axis:
                G = mesh.G2
                GT = mesh.G1T
                col_points_xi  = mesh.glj_points
                col_points_eta  = mesh.gll_points
            else:
                G = mesh.G2
                GT = mesh.G2T
                col_points_xi  = mesh.gll_points
                col_points_eta  = mesh.gll_points

            # Single precision in the NetCDF file but the later interpolation
            # routines require double precision. Assignment to this array will
            # force a cast.
            utemp = np.zeros((mesh.ndumps, mesh.npol + 1, mesh.npol + 1, 3),
                             dtype=np.float64, order="F")

            mesh_dict = mesh.f.groups["Snapshots"].variables

            # Load displacement from all GLL points.
            for ipol in xrange(mesh.npol + 1):
                for jpol in xrange(mesh.npol + 1):
                    start_chunk = gll_point_ids[ipol, jpol] / \
                        mesh.chunks[1] * mesh.chunks[1]
                    start_chunk = gll_point_ids[ipol, jpol]

                    gll_to_read = min(mesh.chunks[1],
                                      mesh.npoints - start_chunk)

                    for i, var in enumerate(["disp_s", "disp_p", "disp_z"]):
                        if var not in mesh_dict:
                            continue
                        arr = mesh_dict[var]
                        # Interesting indexing once again...but consistent with
                        # the fortran output.
                        utemp[:, jpol, ipol, i] = mesh_dict[var][:, start_chunk]

            strain_fct_map = {
                "monopole": sem_derivatives.strain_monopole_td,
                "dipole": sem_derivatives.strain_dipole_td,
                "quadpole": sem_derivatives.strain_quadpole_td}

            utemp = utemp
            strain = strain_fct_map[mesh.excitation_type](
               utemp, G, GT, col_points_xi, col_points_eta, mesh.npol,
               mesh.ndumps, corner_points, eltype, axis)

            final_strain = np.empty((strain.shape[0], 6))

            for i in xrange(6):
                final_strain[:, i] = spectral_basis.lagrange_interpol_2D_td(
                    col_points_xi, col_points_eta, strain[:, :, :, i], xi, eta)
            final_strain[:, 3] *= -1.0
            final_strain[:, 5] *= -1.0

            mij = rotations.rotate_symm_tensor_voigt_xyz_src_to_xyz_earth_1d(
                source.tensor_voigt, np.deg2rad(source.longitude),
                np.deg2rad(source.colatitude))
            mij = rotations.rotate_symm_tensor_voigt_xyz_earth_to_xyz_src_1d(
                mij, np.deg2rad(receiver.longitude),
                np.deg2rad(receiver.colatitude))
            mij = rotations.rotate_symm_tensor_voigt_xyz_to_src_1d(
                mij, rotmesh_phi)
            mij /= mesh.amplitude

            fac_1 = rotations.azim_factor_bw(
                rotmesh_phi, np.array([0.0, 1.0, 0.0]), 2, 1)
            fac_2 = rotations.azim_factor_bw(
                rotmesh_phi, np.array([0.0, 1.0, 0.0]), 2, 2)

            final = np.zeros(final_strain.shape[0], dtype="float64")
            final += final_strain[:, 0] * mij[0] * 1.0 * fac_1
            final += final_strain[:, 1] * mij[1] * 1.0 * fac_1
            final += final_strain[:, 2] * mij[2] * 1.0 * fac_1
            final += final_strain[:, 3] * mij[3] * 2.0 * fac_2
            final += final_strain[:, 4] * mij[4] * 2.0 * fac_1
            final += final_strain[:, 5] * mij[5] * 2.0 * fac_2
            final *= -1.0

            return final

if __name__ == "__main__":
    axisem_db = AxiSEMDB("../../axisem/SOLVER/prem50s_forces/")
    receiver = Receiver(latitude=42.6390, longitude=74.4940, depth_in_m=0.0)
    source = Source(
        latitude=89.91, longitude=0.0, depth_in_m=12000,
        m_rr=4.710000e+24 / 1E7,
        m_tt=3.810000e+22 / 1E7,
        m_pp=-4.740000e+24 / 1E7,
        m_rt=3.990000e+23 / 1E7,
        m_rp=-8.050000e+23 / 1E7,
        m_tp=-1.230000e+24 / 1E7)
    f = axisem_db.get_seismogram(source=source, receiver=receiver, component="N")

    ################
    # DEBUGGING START
    import sys
    __o_std__ = sys.stdout
    sys.stdout = sys.__stdout__
    from IPython.core.debugger import Tracer
    Tracer(colors="Linux")()
    sys.stdout = __o_std__
    # DEBUGGING END
    ################


