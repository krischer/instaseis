from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import shutil
import tempfile


TEST_DATA = os.path.join(os.path.dirname(__file__), "instaseis", "tests",
                         "data")


def repack_databases():
    """
    Repack databases and create a couple temporary test databases.

    It will generate various repacked databases and use them in the test
    suite - this for one tests the repacking but also that Instaseis can
    work with a number of different database layouts.
    """
    from instaseis.scripts.repack_instaseis_database import transpose_data

    root_folder = tempfile.mkdtemp()

    transposed_bw_db = os.path.join(
        root_folder, "transposed_100s_db_bwd_displ_only")
    transposed_and_back_bw_db = os.path.join(
        root_folder, "100s_db_bwd_displ_only")
    os.makedirs(transposed_bw_db)
    os.makedirs(transposed_and_back_bw_db)

    db = os.path.join(TEST_DATA, "100s_db_bwd_displ_only")
    f = "ordered_output.nc4"
    px = os.path.join(db, "PX", "Data", f)
    pz = os.path.join(db, "PZ", "Data", f)
    px_out = os.path.join(transposed_bw_db, "PX", f)
    pz_out = os.path.join(transposed_bw_db, "PZ", f)

    os.makedirs(os.path.dirname(px_out))
    os.makedirs(os.path.dirname(pz_out))

    transpose_data(input_filename=px, output_filename=px_out, contiguous=True,
                   compression_level=None)
    transpose_data(input_filename=pz, output_filename=pz_out, contiguous=True,
                   compression_level=None)

    return {
        "root_folder": root_folder,
        "databases": {
            # Important is that the name is fairly similar to the original
            # as some tests use the patterns in the name.
            "transposed_100s_db_bwd_displ_only": transposed_bw_db
        }
    }


def pytest_configure(config):
    config.dbs = repack_databases()


def pytest_unconfigure(config):
    if os.path.exists(config.dbs["root_folder"]):
        shutil.rmtree(config.dbs["root_folder"])
