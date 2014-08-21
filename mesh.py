import numpy as np
from scipy.spatial import cKDTree

import spectral_basis


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
        self.dt = getattr(self.f, "strain dump sampling rate in sec")

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
