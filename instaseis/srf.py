#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Function to read 'standard rupture format' (*.srf) files

:copyright:
    Martin van Driel (Martin@vanDriel.de)
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
from __future__ import absolute_import

import numpy as np
from .source import Source

DEFAULT_MU = 32e9


def read_srf(filename, normalize=False):
    """
    Read a 'standard rupture format' (.srf) file and return a list of
    instaseis.Source objects.

    :param filename: path to the .srf file
    :param normalize: normalize the sliprate to 1
    """
    with open(filename, "rt") as fh:
        return _read_srf(fh, normalize=normalize)


def _read_srf(f, normalize=False):
    # go to POINTS block
    line = f.readline()
    while 'POINTS' not in line:
        line = f.readline()

    npoints = int(line.split()[1])

    sources = []

    for _ in np.arange(npoints):
        lon, lat, dep, stk, dip, area, tinit, dt = \
            map(float, f.readline().split())
        rake, slip1, nt1, slip2, nt2, slip3, nt3 = \
            map(float, f.readline().split())

        dep *= 1e3  # km    > m
        area *= 1e-4  # cm^2 > m^2
        slip1 *= 1e-2  # cm   > m
        slip2 *= 1e-2  # cm   > m
        # slip3 *= 1e-2  # cm   > m

        nt1, nt2, nt3 = map(int, (nt1, nt2, nt3))

        if nt1 > 0:
            line = f.readline()
            while len(line.split()) < nt1:
                line = line + f.readline()
            stf = np.array(line.split(), dtype=float)
            if normalize:
                stf /= np.trapz(stf, dx=dt)

            M0 = area * DEFAULT_MU * slip1

            sources.append(Source.from_strike_dip_rake(lat, lon, dep, stk, dip,
                           rake, M0, time_shift=tinit, sliprate=stf, dt=dt))

        if nt2 > 0:
            line = f.readline()
            while len(line.split()) < nt2:
                line = line + f.readline()
            stf = np.array(line.split(), dtype=float)
            if normalize:
                stf /= np.trapz(stf, dx=dt)

            M0 = area * DEFAULT_MU * slip2

            sources.append(Source.from_strike_dip_rake(lat, lon, dep, stk, dip,
                           rake, M0, time_shift=tinit, sliprate=stf, dt=dt))

        if nt3 > 0:
            raise NotImplementedError

    return sources
