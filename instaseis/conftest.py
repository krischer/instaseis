from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import collections
import pickle
import os
import shutil
import tempfile
import time


TEST_DATA = os.path.join(os.path.dirname(__file__), "tests", "data")


def repack_databases():
    """
    Repack databases and create a couple of temporary test databases.

    It will generate various repacked databases and use them in the test
    suite - this for one tests the repacking but also that Instaseis can
    work with a number of different database layouts.
    """
    try:
        import netCDF4  # NOQA
        import click  # NOQA
    except ImportError:
        print("\nSkipping database repacking tests which require `click` and "
              "`netCDF4` to be installed.\n")
        return {
            "root_folder": None,
            "databases": {}
        }

    import h5py
    from instaseis.scripts.repack_db import merge_files, repack_file

    root_folder = tempfile.mkdtemp()

    # First create a transposed database - make it contiguous.
    transposed_bw_db = os.path.join(
        root_folder, "transposed_100s_db_bwd_displ_only")
    os.makedirs(transposed_bw_db)

    db = os.path.join(TEST_DATA, "100s_db_bwd_displ_only")
    f = "ordered_output.nc4"
    px = os.path.join(db, "PX", "Data", f)
    pz = os.path.join(db, "PZ", "Data", f)

    px_tr = os.path.join(transposed_bw_db, "PX", f)
    pz_tr = os.path.join(transposed_bw_db, "PZ", f)

    os.makedirs(os.path.dirname(px_tr))
    os.makedirs(os.path.dirname(pz_tr))

    print("Creating transposed test database ...")
    repack_file(input_filename=px, output_filename=px_tr, contiguous=True,
                compression_level=None, quiet=True, transpose=True)
    repack_file(input_filename=pz, output_filename=pz_tr, contiguous=True,
                compression_level=None, quiet=True, transpose=True)

    # Now transpose it again which should result in the original layout.
    transposed_and_back_bw_db = os.path.join(
        root_folder, "transposed_and_back_100s_db_bwd_displ_only")
    os.makedirs(transposed_and_back_bw_db)

    px_tr_and_back = os.path.join(transposed_and_back_bw_db, "PX", f)
    pz_tr_and_back = os.path.join(transposed_and_back_bw_db, "PZ", f)
    os.makedirs(os.path.dirname(px_tr_and_back))
    os.makedirs(os.path.dirname(pz_tr_and_back))

    print("Creating compressed re-transposed test database ...")
    repack_file(input_filename=px_tr, output_filename=px_tr_and_back,
                contiguous=False, compression_level=4, quiet=True,
                transpose=True)
    repack_file(input_filename=pz_tr, output_filename=pz_tr_and_back,
                contiguous=False, compression_level=4, quiet=True,
                transpose=True)

    # Now add another simple repacking test - repack the original one and
    # repack the transposed one.
    repacked_bw_db = os.path.join(
        root_folder, "repacked_100s_db_bwd_displ_only")
    os.makedirs(repacked_bw_db)

    px_r = os.path.join(repacked_bw_db, "PX", f)
    pz_r = os.path.join(repacked_bw_db, "PZ", f)

    os.makedirs(os.path.dirname(px_r))
    os.makedirs(os.path.dirname(pz_r))

    print("Creating a simple repacked test database ...")
    repack_file(input_filename=px, output_filename=px_r, contiguous=True,
                compression_level=None, quiet=True, transpose=False)
    repack_file(input_filename=pz, output_filename=pz_r, contiguous=True,
                compression_level=None, quiet=True, transpose=False)

    # Also repack the transposed database.
    repacked_transposed_bw_db = os.path.join(
        root_folder, "repacked_transposed_100s_db_bwd_displ_only")
    os.makedirs(repacked_transposed_bw_db)

    px_r_tr = os.path.join(repacked_transposed_bw_db, "PX", f)
    pz_r_tr = os.path.join(repacked_transposed_bw_db, "PZ", f)

    os.makedirs(os.path.dirname(px_r_tr))
    os.makedirs(os.path.dirname(pz_r_tr))

    print("Creating a simple transposed and repacked test database ...")
    repack_file(input_filename=px_tr, output_filename=px_r_tr, contiguous=True,
                compression_level=None, quiet=True, transpose=False)
    repack_file(input_filename=pz_tr, output_filename=pz_r_tr, contiguous=True,
                compression_level=None, quiet=True, transpose=False)

    # Add a merged database.
    merged_bw_db = os.path.join(
        root_folder, "merged_100s_db_bwd_displ_only")
    os.makedirs(merged_bw_db)
    print("Creating a merged test database ...")
    merge_files(filenames=[px, pz], output_folder=merged_bw_db,
                contiguous=True, compression_level=None, quiet=True)

    # Another merged database but this time originating from a transposed
    # database.
    merged_transposed_bw_db = os.path.join(
        root_folder, "merged_transposed_100s_db_bwd_displ_only")
    os.makedirs(merged_transposed_bw_db)
    print("Creating a merged transposed test database ...")
    merge_files(filenames=[px_tr, pz_tr],
                output_folder=merged_transposed_bw_db,
                contiguous=True, compression_level=None, quiet=True)

    # Make a horizontal only merged database.
    horizontal_only_merged_db = os.path.join(
        root_folder, "horizontal_only_merged_db")
    os.makedirs(horizontal_only_merged_db)
    print("Creating a horizontal only merged test database ...")
    merge_files(filenames=[px_tr],
                output_folder=horizontal_only_merged_db,
                contiguous=False, compression_level=2, quiet=True)

    # Make a vertical only merged database.
    vertical_only_merged_db = os.path.join(
        root_folder, "vertical_only_merged_db")
    os.makedirs(vertical_only_merged_db)
    print("Creating a vertical only merged test database ...")
    merge_files(filenames=[pz_tr],
                output_folder=vertical_only_merged_db,
                contiguous=False, compression_level=2, quiet=True)

    # Create a merged version of the fwd database.
    fwd_db = os.path.join(TEST_DATA, "100s_db_fwd")
    merged_fwd_db = os.path.join(
        root_folder, "merged_100s_db_fwd")
    os.makedirs(merged_fwd_db)

    f = "ordered_output.nc4"
    d1 = os.path.join(fwd_db, "MZZ", "Data", f)
    d2 = os.path.join(fwd_db, "MXX_P_MYY", "Data", f)
    d3 = os.path.join(fwd_db, "MXZ_MYZ", "Data", f)
    d4 = os.path.join(fwd_db, "MXY_MXX_M_MYY", "Data", f)
    assert os.path.exists(d1), d1
    assert os.path.exists(d2), d2
    assert os.path.exists(d3), d3
    assert os.path.exists(d4), d4

    print("Creating a merged forward test database ...")
    merge_files(filenames=[d1, d2, d3, d4],
                output_folder=merged_fwd_db, contiguous=False,
                compression_level=2, quiet=True)

    # Actually test the shapes of the fields to see that something happened.
    with h5py.File(pz, mode="r") as f:
        original_shape = f["Snapshots"]["disp_z"].shape
    with h5py.File(pz_tr, mode="r") as f:
        transposed_shape = f["Snapshots"]["disp_z"].shape
    with h5py.File(pz_tr_and_back, mode="r") as f:
        transposed_and_back_shape = f["Snapshots"]["disp_z"].shape
    with h5py.File(pz_r, mode="r") as f:
        repacked_shape = f["Snapshots"]["disp_z"].shape
    with h5py.File(pz_r_tr, mode="r") as f:
        repacked_transposed_shape = f["Snapshots"]["disp_z"].shape
    with h5py.File(os.path.join(merged_bw_db, "merged_output.nc4"), "r") as f:
        merged_shape = f["MergedSnapshots"].shape
    with h5py.File(os.path.join(merged_transposed_bw_db,
                                "merged_output.nc4"), "r") as f:
        merged_tr_shape = f["MergedSnapshots"].shape
    with h5py.File(os.path.join(horizontal_only_merged_db,
                                "merged_output.nc4"), "r") as f:
        horizontal_only_merged_tr_shape = f["MergedSnapshots"].shape
    with h5py.File(os.path.join(vertical_only_merged_db,
                                "merged_output.nc4"), "r") as f:
        vertical_only_merged_tr_shape = f["MergedSnapshots"].shape
    with h5py.File(os.path.join(merged_fwd_db,
                                "merged_output.nc4"), "r") as f:
        merged_fwd_shape = f["MergedSnapshots"].shape

    assert original_shape == tuple(reversed(transposed_shape))
    assert original_shape == transposed_and_back_shape
    assert original_shape == repacked_shape
    assert original_shape == tuple(reversed(repacked_transposed_shape))
    assert merged_shape == (192, 5, 5, 5, 73), str(merged_shape)
    assert merged_tr_shape == (192, 5, 5, 5, 73), str(merged_tr_shape)
    assert horizontal_only_merged_tr_shape == (192, 3, 5, 5, 73), \
        str(horizontal_only_merged_tr_shape)
    assert vertical_only_merged_tr_shape == (192, 2, 5, 5, 73), \
        str(vertical_only_merged_tr_shape)
    assert merged_fwd_shape == (192, 10, 5, 5, 73), \
        str(merged_fwd_shape)

    dbs = collections.OrderedDict()
    # Important is that the name is fairly similar to the original
    # as some tests use the patterns in the name.
    dbs["transposed_100s_db_bwd_displ_only"] = transposed_bw_db
    dbs["transposed_and_back_100s_db_bwd_displ_only"] = \
        transposed_and_back_bw_db
    dbs["repacked_100s_db_bwd_displ_only"] = repacked_bw_db
    dbs["repacked_transposed_100s_db_bwd_displ_only"] = \
        repacked_transposed_bw_db
    dbs["merged_100s_db_bwd_displ_only"] = merged_bw_db
    dbs["merged_transposed_100s_db_bwd_displ_only"] = merged_transposed_bw_db

    # Special databases.
    dbs["horizontal_only_merged_database"] = horizontal_only_merged_db
    dbs["vertical_only_merged_database"] = vertical_only_merged_db

    # Forward databases.
    dbs["merged_100s_db_fwd"] = merged_fwd_db

    return {
        "root_folder": root_folder,
        "databases": dbs
    }


try:
    import xdist  # NOQA
except ImportError:
    raise Exception("pytest-xdist is required to run the tests. Please "
                    "install with `pip install pytest-xdist`.")


def is_master(config):
    """
    Returns True/False if the current node is the master node.

    Only applies to if run with pytest-xdist.
    """
    # This attribute is only set on slaves.
    if hasattr(config, "slaveinput"):
        return False
    else:
        return True


def pytest_configure(config):
    if is_master(config):
        config.dbs = repack_databases()
    else:
        while True:
            if "dbs" not in config.slaveinput:
                time.sleep(0.01)
                continue
            break
        config.dbs = pickle.loads(config.slaveinput["dbs"])


def pytest_unconfigure(config):
    print("Deleting all test databases ...")
    if is_master(config) and config.dbs["root_folder"]:
        if os.path.exists(config.dbs["root_folder"]):
            shutil.rmtree(config.dbs["root_folder"])


def pytest_configure_node(node):
    """
    This is only called on the master - we use it to send the information to
    all the slaves.

    Only applies to if run with pytest-xdist.
    """
    node.slaveinput["dbs"] = pickle.dumps(node.config.dbs)
