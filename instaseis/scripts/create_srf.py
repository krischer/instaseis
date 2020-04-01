#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hacky code to generate some finite source in a .srf file

:copyright:
    Martin van Driel (Martin@vanDriel.de), 2020
:license:
    GNU Lesser General Public License, Version 3 [non-commercial/academic use]
    (http://www.gnu.org/copyleft/lgpl.html)
"""
import matplotlib.pyplot as plt
import numpy as np
from obspy.signal.filter import lowpass


def main():
    strike = 90.0
    dip = 90.0
    rake = 0.0
    rupture_velo = 0.9
    rupture_len = 1000
    npoints = 1000
    slip = 10.0
    dep = 50
    vs = 5.0

    dt = 5.0
    nts = 100

    lonstart = 0.0

    area = rupture_len * dep * 2 / npoints * 1e10  # in cm**2

    equator_len = 2 * np.pi * 6371

    lat = np.zeros(npoints)
    lon = np.linspace(
        lonstart, lonstart + rupture_len / equator_len * 360.0, npoints
    )
    tinit = np.linspace(0.0, rupture_len, npoints) / (rupture_velo * vs)

    stf = np.zeros(nts)
    stf[1] = 1.0 / dt
    stf = lowpass(stf, 1.0 / 100.0, 1.0 / dt)

    plt.plot(stf)
    plt.show()

    f = open("strike_slip_eq.srf", "w")
    f.write("POINTS %d\n" % (npoints,))

    for i in np.arange(npoints):
        # lon, lat, dep, stk, dip, area, tinit, dt
        f.write(
            "%11.5f %11.5f %11.5f %11.5f %11.5f %11.5f %11.5f %11.5f\n"
            % (lon[i], lat[i], dep, strike, dip, area, tinit[i], dt)
        )

        # rake, slip1, nt1, slip2, nt2, slip3, nt3
        f.write(
            "%11.5f %11.5f %5d %11.5f %5d %11.5f %5d\n"
            % (rake, slip, nts, 0.0, 0, 0.0, 0)
        )

        # f.write('%11.5f %11.5f %11.5f\n' % (0., 1., 0.))
        count = 0
        for j in np.arange(nts):
            f.write("%11.5f " % (stf[j],))
            count += 1
            if count % 10 == 0:
                f.write("\n")
                count = 0

    f.close()

    # m = Basemap(projection='cyl', lon_0=0, lat_0=0, resolution='c')
    #
    # m.drawcoastlines()
    # m.fillcontinents()
    # m.drawparallels(np.arange(-90., 120., 30.))
    # m.drawmeridians(np.arange(0., 420., 60.))
    # m.drawmapboundary()
    #
    # focmecs = [strike, dip, rake]
    #
    # ax = plt.gca()
    # for i in np.arange(npoints):
    #     x, y = m(lon[i], lat[i])
    #     b = Beach(focmecs, xy=(x, y), width=10, linewidth=1, alpha=0.85)
    #     b.set_zorder(10)
    #     ax.add_collection(b)
    # plt.show()


if __name__ == "__main__":
    main()
