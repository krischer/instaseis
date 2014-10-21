#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mostly testing the various wrappers.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
from __future__ import absolute_import

import numpy as np
import os
import inspect


from instaseis import sem_derivatives, InstaSeisDB


# Most generic way to get the data folder path.
DATA = os.path.join(os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe()))), "data")


# def test_strain_monopole():
#     instaseis_bwd = InstaSeisDB(os.path.join(DATA, "100s_db_bwd"))
#
#     G = instaseis_bwd.parsed_mesh.G2
#     GT = instaseis_bwd.parsed_mesh.G2T
#     col_points_xi = instaseis_bwd.parsed_mesh.gll_points
#     col_points_eta = instaseis_bwd.parsed_mesh.gll_points
#     npol = instaseis_bwd.parsed_mesh.npol
#
#     corner_points = np.empty((4, 2), dtype="float64")
#     corner_points[0,:] = [-1,-1]
#     corner_points[1,:] = [1,-1]
#     corner_points[2,:] = [1,1]
#     corner_points[3,:] = [-1,1]
#
#     eltype = 1
#     axis = False
#     nsamp = 1
#
#     corner_points = np.empty((4, 2), dtype="float64")
#
#     s = np.zeros((npol+1, npol+1), dtype="float64")
#     z = np.zeros((npol+1, npol+1), dtype="float64")
#
#     for i in np.arange(npol+1):
#         for j in np.arange(npol+1):
#             s[i,j] = col_points_xi[i]
#             z[i,j] = col_points_eta[j]
#
#     utemp = np.zeros((nsamp, npol+1, npol+1, 3), dtype="float64", order="F")
#     utemp[0,:,:,0] = s**2 * z
#     utemp[0,:,:,1] = 0
#     utemp[0,:,:,2] = s * z**2
#
#     strain_ref = np.zeros((nsamp, npol+1, npol+1, 6), dtype="float64")
#     strain_ref[0,:,:,0] = 2 * s * z
#     strain_ref[0,:,:,1] = s * z
#     strain_ref[0,:,:,2] = 2 * s * z
#     strain_ref[0,:,:,3] = 0
#     strain_ref[0,:,:,4] = (s**2 + z**2) / 2
#     strain_ref[0,:,:,5] = 0
#
#     strain = sem_derivatives.strain_monopole_td(
#         utemp, G, GT, col_points_xi, col_points_eta, npol,
#         nsamp, corner_points, eltype, axis)
#
#     print
#     print s
#     print
#     print z
#     print
#     print utemp
#     print
#     print strain
#     print strain_ref
#
#     #np.testing.assert_allclose(strain, strain_ref,
#     #                           rtol=1E-7, atol=1E-10)
#
# def test_strain_monopole():
#     instaseis_bwd = InstaSeisDB(os.path.join(DATA, "100s_db_bwd"))
#
#     G = np.array([[-.5, -.5], [.5, .5]])
#     GT = G.T
#     col_points_xi = np.array([-1., 1.])
#     col_points_eta = np.array([-1., 1.])
#     npol = 1
#
#     corner_points = np.empty((4, 2), dtype="float64", order='F')
#     corner_points[0,:] = [1,-1]
#     corner_points[1,:] = [3,-1]
#     corner_points[2,:] = [3,1]
#     corner_points[3,:] = [1,1]
#
#     eltype = 1
#     axis = False
#     nsamp = 1
#
#     s = np.zeros((npol+1, npol+1), dtype="float64")
#     z = np.zeros((npol+1, npol+1), dtype="float64")
#
#     for i in np.arange(npol+1):
#         for j in np.arange(npol+1):
#             s[i,j] = col_points_xi[i] + 2
#             z[i,j] = col_points_eta[j]
#
#     utemp = np.zeros((nsamp, npol+1, npol+1, 3), dtype="float64", order="F")
#     utemp[0,:,:,0] = s**2 * z
#     utemp[0,:,:,1] = 0
#     utemp[0,:,:,2] = s * z**2
#
#     strain_ref = np.zeros((nsamp, npol+1, npol+1, 6), dtype="float64")
#     strain_ref[0,:,:,0] = 2 * s * z
#     strain_ref[0,:,:,1] = s * z
#     strain_ref[0,:,:,2] = 2 * s * z
#     strain_ref[0,:,:,3] = 0
#     strain_ref[0,:,:,4] = (s**2 + z**2) / 2
#     strain_ref[0,:,:,5] = 0
#
#     strain = sem_derivatives.strain_monopole_td(
#         utemp, G, GT, col_points_xi, col_points_eta, npol,
#         nsamp, corner_points, eltype, axis)
#
#     print
#     print s
#     print
#     print z
#     print
#     print utemp
#     print
#     print strain
#     print
#     print strain_ref
#
#     #np.testing.assert_allclose(strain, strain_ref,
#     #                           rtol=1E-7, atol=1E-10)
