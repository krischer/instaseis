import ctypes as C
import numpy as np

from helpers import load_lib


lib = load_lib()

lib.azim_factor_bw.restype = C.c_double


def azim_factor_bw(phi, fi, isim, ikind):
    fi = np.require(fi, dtype=np.float64)
    factor = lib.azim_factor_bw(
        C.c_double(phi),
        fi.ctypes.data_as(C.POINTER(C.c_double)),
        C.c_int(isim), C.c_int(ikind))
    return factor


def test_azim_factor_bw():
    factor = azim_factor_bw(3.143651018669930, np.array([0.0, 1.0, 0.0]), 2, 1)
    assert abs(factor - -0.99999788156734637) < 1E-7


def rotate_symm_tensor_voigt_xyz_earth_to_xyz_src_1d(mt, phi, theta):
    mt = np.require(mt, dtype=np.float64)
    out = np.empty(mt.shape, dtype=np.float64)
    lib.rotate_symm_tensor_voigt_xyz_earth_to_xyz_src_1d(
        mt.ctypes.data_as(C.POINTER(C.c_double)),
        C.c_double(phi),
        C.c_double(theta),
        out.ctypes.data_as(C.POINTER(C.c_double)))
    return out


def rotate_symm_tensor_voigt_xyz_src_to_xyz_earth_1d(mt, phi, theta):
    np.require(mt, dtype=np.float64)
    out = np.empty(mt.shape, dtype=np.float64)
    lib.rotate_symm_tensor_voigt_xyz_src_to_xyz_earth_1d(
        mt.ctypes.data_as(C.POINTER(C.c_double)),
        C.c_double(phi),
        C.c_double(theta),
        out.ctypes.data_as(C.POINTER(C.c_double)))
    return out


def rotate_symm_tensor_voigt_xyz_to_src_1d(mt, phi):
    mt = np.require(mt, dtype=np.float64)
    out = np.empty(mt.shape, dtype=np.float64)
    lib.rotate_symm_tensor_voigt_xyz_to_src_1d(
        mt.ctypes.data_as(C.POINTER(C.c_double)),
        C.c_double(phi),
        out.ctypes.data_as(C.POINTER(C.c_double)))
    return out
