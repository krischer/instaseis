#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Useful script to make sure two databases return exactly the same the
seismograms.

Especially useful to be able to trust the repacking script. It works by
generating random source and receiver locations and compares both. It is an
infinite loop so the user has to manually cancel it after a while.


Usage:

.. code-block:: bash

    $ python -m instaseis.scripts.compare_dbs DB1 DB2 DB3


In this example ``DB2``  and ``DB3`` will both be compared the ``DB1``.


Requires click, Instaseis, and ObsPy.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2016
:license:
    GNU Lesser General Public License, Version 3 [non-commercial/academic use]
    (http://www.gnu.org/copyleft/lgpl.html)
"""
import random

import click
import instaseis
import obspy


@click.command(help="Pass a list of databases to assert that they produce the "
                    "same seismograms. The first one will be treated as the "
                    "reference.")
@click.option("--seed", type=int,
              help="Optionally pass a seed number to make it reproducible.")
@click.argument("databases", type=click.Path(exists=True, file_okay=False,
                                             dir_okay=True), nargs=-1)
def compare_dbs(seed, databases):
    if seed:
        random.seed(seed)
    reference = instaseis.open_db(databases[0])
    others = [instaseis.open_db(_i) for _i in databases[1:]]

    max_depth = (reference.info.max_radius - reference.info.min_radius)

    while True:
        receiver = instaseis.Receiver(
            latitude=random.random() * 180.0 - 90.0,
            longitude=random.random() * 360.0 - 180.0,
            network="AB", station="CED")
        source = instaseis.Source(
            latitude=random.random() * 180.0 - 90.0,
            longitude=random.random() * 360.0 - 180.0,
            depth_in_m=random.random() * max_depth,
            m_rr=4.710000e+24 / 1E7,
            m_tt=3.810000e+22 / 1E7,
            m_pp=-4.740000e+24 / 1E7,
            m_rt=3.990000e+23 / 1E7,
            m_rp=-8.050000e+23 / 1E7,
            m_tp=-1.230000e+24 / 1E7,
            origin_time=obspy.UTCDateTime(2011, 1, 2, 3, 4, 5))

        print('======')

        ref = reference.get_seismograms(source=source, receiver=receiver,
                                        components="ZNERT")

        oth = [_i.get_seismograms(source=source, receiver=receiver,
                                  components="ZNERT") for _i in others]

        for _i, _j in zip(oth, others):
            print(_j.info.directory, ":", ref == _i)
            assert ref == _i, str(source) + "\n" + str(receiver)

if __name__ == "__main__":
    compare_dbs()
