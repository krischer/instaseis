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
from source import Source

DEFAULT_MU = 32e9

def read_srf(filename, normalize=False):
    f = open(filename, 'r')

    # go to POINTS block
    line = f.readline()
    while not 'POINTS' in line:
        line = f.readline()

    npoints = int(line.split()[1])

    sources = []

    for i in np.arange(npoints):
        lon, lat, dep, stk, dip, area, tinit, dt = f.readline().split()
        rake, slip1, nt1, slip2, nt2, slip3, nt3 = f.readline().split()

        lon   = float(lon  )
        lat   = float(lat  )
        dep   = float(dep  ) * 1e3 # km    > m
        stk   = float(stk  )
        dip   = float(dip  )
        area  = float(area ) * 1e-4 # cm^2 > m^2
        tinit = float(tinit)
        dt    = float(dt   )
        rake  = float(rake )
        slip1 = float(slip1) * 1e-2 # cm   > m
        slip2 = float(slip2) * 1e-2 # cm   > m
        slip3 = float(slip3) * 1e-2 # cm   > m

        nt1 = int(nt1)
        nt2 = int(nt2)
        nt3 = int(nt3)

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
            line = f.readline()
            while len(line.split()) < nt3:
                line = line + f.readline()
            stf = np.array(line.split(), dtype=float)

    f.close

    return sources
