#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Basic integration tests for the AxiSEM database Python interface.

XXX: Right now the path to the database is hardcoded! It is too big to commit
it with the repository so something needs to be figured out. Best way is likely
to just generate a smaller database.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
import numpy as np

from axisem_db import AxiSEMDB
from source import Source, Receiver


def test_basic_output():
    """
    Test against output directly from the axisem DB reader.
    """
    axisem_db = \
        AxiSEMDB("/Users/lion/workspace/code/axisem/SOLVER/prem50s_forces")
    receiver = Receiver(latitude=42.6390, longitude=74.4940, depth_in_m=0.0)
    source = Source(
        latitude=89.91, longitude=0.0, depth_in_m=12000,
        m_rr=4.710000e+24 / 1E7,
        m_tt=3.810000e+22 / 1E7,
        m_pp=-4.740000e+24 / 1E7,
        m_rt=3.990000e+23 / 1E7,
        m_rp=-8.050000e+23 / 1E7,
        m_tp=-1.230000e+24 / 1E7)
    data = axisem_db.get_seismogram(source=source, receiver=receiver,
                                    component="N")
    n_data = np.array([
        -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0,
        -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -7.183695E-44,
        6.201348E-42, 1.478579E-40, -6.104198E-39, -2.939338E-37,
        -4.945937E-37, 1.072107E-34, -3.041621E-34, -7.652161E-33,
        2.214094E-31, 5.601806E-31, 4.240771E-30, 1.006155E-28, -5.399007E-28,
        5.790040E-27, -3.010067E-26, -1.175898E-24, 5.809355E-24, 5.223013E-23,
        -1.453331E-21, -6.294746E-20, -7.540908E-17, -1.381345E-13,
        -3.265489E-11, -1.351891E-09, -3.024001E-09, 6.848123E-08,
        1.205207E-07, 4.819264E-08, 5.518334E-08, 6.583487E-08, 8.150570E-08,
        1.012352E-07, 1.055464E-07, 7.414001E-08, 1.679650E-07, 2.784896E-07,
        2.206829E-07, 1.936204E-07, 2.340856E-07, 1.803208E-07, 1.385476E-07,
        6.785513E-08, 2.899254E-08, 6.294315E-09, -4.599663E-09, -1.743001E-08,
        -2.595952E-08, -1.071743E-09, 4.191247E-08, 1.014835E-07, 1.581918E-07,
        1.939492E-07, 2.346080E-07, 2.382118E-07, 2.137073E-07, 1.263431E-07,
        -4.892889E-08, -4.451496E-07, -1.068539E-06, -5.157217E-07,
        1.545875E-06, 3.819888E-07, -3.938210E-07, 4.684740E-07, -8.194233E-08,
        2.149664E-07, 1.476712E-07, 3.730389E-08, 1.029422E-07, 1.638113E-07,
        1.232860E-07, 1.012607E-07, 4.174790E-08, 1.696431E-07, 6.332481E-08,
        3.907343E-08, -5.263358E-08, -1.933687E-07, -1.378726E-07,
        -1.598703E-07, -1.032468E-07, -1.117853E-07, -5.307041E-08,
        2.657340E-08, 4.544845E-08, -7.143105E-09, -4.218048E-08, 3.107075E-08,
        3.689852E-07, 7.780862E-07, -7.324227E-08, -2.608334E-06,
        -2.285997E-06, 2.627911E-06, 3.922536E-06, 1.285143E-06, 6.946971E-07,
        -3.389947E-07, -8.403630E-07, -8.852953E-07, -8.801109E-07,
        -6.610466E-07, -4.615913E-07, -2.545142E-07, -6.651063E-08,
        6.130591E-08, 1.723345E-07, 2.304882E-07, 2.374456E-07, 2.497180E-07,
        2.150196E-07, 1.797127E-07, 1.443892E-07, 9.477943E-08, 6.186546E-08,
        3.795263E-08, 3.080387E-09, -4.054393E-09, -7.624076E-09,
        -1.720859E-08, -1.281327E-08, -4.164188E-09])
    np.testing.assert_allclose(data, n_data, rtol=1E-4)
