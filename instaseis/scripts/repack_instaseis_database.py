#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Repacking Instaseis databases.

Requires click, h5py, and numpy.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2016
:license:
    GNU Lesser General Public License, Version 3 [non-commercial/academic use]
    (http://www.gnu.org/copyleft/lgpl.html)
"""
import math
import os

import click
import h5py
import numpy as np


def process_file(input_filename, output_filename, method):
    assert os.path.exists(input_filename)
    assert not os.path.exists(output_filename)

    try:
        f_in = h5py.File(input_filename, "r")
        f_out = h5py.File(output_filename, libver="latest")

        # Copy attributes.
        for key, value in f_in.attrs.items():
            f_out.attrs[key] = value

        # And simple groups.
        for group in f_in.keys():
            # Special cased later on.
            if group == "Snapshots":
                continue
            click.echo(click.style("\tCopying group '%s'..." % group,
                                   fg="blue"))
            f_out.copy(f_in[group], group)

        if "Snapshots" not in f_in:
            return

        # Preserve basic default shape, just transpose it.
        if method == "continuous":
            snapshots = f_out.create_group("/Snapshots")
            for _i, group in enumerate(f_in["Snapshots"].keys()):
                click.echo(click.style(
                    "\tCopying 'Snapshots/%s' (%i of %i)..." % (
                        group, _i + 1, len(f_in["Snapshots"])),
                    fg="blue"))

                ds_i = f_in["Snapshots"][group]

                # Transpose and write non-chunked. This creates a continuous
                # array in the file with the time axis being the fast axis.
                ds_o = snapshots.create_dataset(
                        group, shape=(ds_i.shape[1], ds_i.shape[0]),
                        dtype=ds_i.dtype, chunks=None, compression=None)

                # Copy around 8 Megabytes at a time. This seems to be the
                # sweet spot at least on my laptop.
                factor = int((8 * 1024 * 1024 / 4) / ds_i.shape[0])
                s = ds_i.shape[1]
                s = int(math.ceil(ds_i.shape[1] / float(factor)))

                with click.progressbar(range(s), length=s,
                                       label="\t  ") as idx:
                    for _i in idx:
                        ds_o[_i * factor: _i * factor + factor, :] = \
                            ds_i[:, _i * factor: _i * factor + factor].T
        # Rewrite the disp_X arrays to one enormous array. Also store each
        # element with all 25 GLL points. Thus a single read operation will
        # extract whatever is necessary.
        elif method == "unrolled":
            # Create a new array but this time in 5D. The first dimension
            # is the element number, the second and third are the GLL
            # points in both directions, the fourth is the time axis, and the
            # last the displacement axis.
            npts = f_in.attrs["number of strain dumps"][0]
            number_of_elements = f_in.attrs["nelem_kwf_global"][0]
            npol = f_in.attrs["npol"][0]

            # Figure out which variables are available in the correct order.
            available_variables = []
            for i in ["disp_s", "disp_p", "disp_z"]:
                if i not in f_in["Snapshots"]:
                    continue
                available_variables.append(i)

            # Get datasets and the dtype.
            mesh_dict = {}
            dtype = None
            for i in available_variables:
                mesh_dict[i] = f_in["Snapshots"][i]
                dtype = mesh_dict[i].dtype

            ds_o = f_out.create_dataset(
                "unrolled_snapshots",
                shape=(number_of_elements, npts,
                       npol + 1,
                       npol + 1,
                       len(available_variables)),
                dtype=dtype, chunks=None, compression=None)

            utemp = np.zeros(
                (npts, npol + 1, npol + 1, len(available_variables)),
                dtype=dtype, order="F")

            # Now it becomes more interesting and very slow.
            sem_mesh = f_in["Mesh"]["sem_mesh"]
            with click.progressbar(range(number_of_elements),
                                   length=number_of_elements,
                                   label="\t  ") as idx:
                for gll_idx in idx:
                    gll_point_ids = sem_mesh[gll_idx]

                    # Load displacement from all GLL points.
                    for i, var in enumerate(available_variables):
                        # The list of ids we have is unique but not sorted.
                        ids = gll_point_ids.flatten()
                        s_ids = np.sort(ids)
                        temp = mesh_dict[var][:, s_ids]
                        for ipol in range(npol + 1):
                            for jpol in range(npol + 1):
                                idx = ipol * 5 + jpol
                                utemp[:, jpol, ipol, i] = \
                                    temp[:,
                                         np.argwhere(s_ids == ids[idx])[0][0]]
                    ds_o[gll_idx] = utemp
        else:
            raise NotImplementedError

    finally:
        try:
            f_in.close()
        except:
            pass
        try:
            f_out.close()
        except:
            pass

def unroll_and_merge(filenames, output_folder):
    """
    Completely unroll and merge both files.
    """
    # Find PX and PZ files.
    assert len(filenames) == 2
    filenames = [os.path.normpath(_i) for _i in filenames]
    px = [_i for _i in filenames if "PX" in _i]
    pz = [_i for _i in filenames if "PZ" in _i]
    assert len(px) == 1
    assert len(pz) == 1
    px = px[0]
    pz = pz[0]

    output_filename = os.path.join(output_folder, "merged_instaseis_db.nc4")
    assert not os.path.exists(output_filename)

    assert os.path.exists(px)
    assert os.path.exists(pz)
    assert not os.path.exists(output_filename)

    try:
        f_in_x = h5py.File(px, "r")
        f_in_z = h5py.File(pz, "r")
        f_out = h5py.File(output_filename, libver="latest")

        # Copy attributes from the vertical file.
        for key, value in f_in_x.attrs.items():
            f_out.attrs[key] = value

        # Same from simple groups.
        for group in f_in_x.keys():
            # Special cased later on.
            if group == "Snapshots":
                continue
            click.echo(click.style("\tCopying group '%s'..." % group,
                                   fg="blue"))
            f_out.copy(f_in_x[group], group)

        # Create a new array but this time in 5D. The first dimension
        # is the element number, the second and third are the GLL
        # points in both directions, the fourth is the time axis, and the
        # last the displacement axis.
        npts = f_in_x.attrs["number of strain dumps"][0]
        number_of_elements = f_in_x.attrs["nelem_kwf_global"][0]
        npol = f_in_x.attrs["npol"][0]

        # Get datasets and the dtype.
        meshes = [
            f_in_x["Snapshots"]["disp_s"],
            f_in_x["Snapshots"]["disp_p"],
            f_in_x["Snapshots"]["disp_z"],
            f_in_z["Snapshots"]["disp_s"],
            f_in_z["Snapshots"]["disp_z"]]
        dtype = meshes[0].dtype

        ds_o = f_out.create_dataset(
            "merged_snapshots",
            shape=(number_of_elements, npts, npol + 1, npol + 1, 5),
            dtype=dtype, chunks=None, compression=None)

        utemp = np.zeros((npts, npol + 1, npol + 1, 5), dtype=dtype, order="F")

        # Now it becomes more interesting and very slow.
        sem_mesh = f_in_x["Mesh"]["sem_mesh"]
        with click.progressbar(range(number_of_elements),
                               length=number_of_elements,
                               label="\t  ") as idx:
            for gll_idx in idx:
                gll_point_ids = sem_mesh[gll_idx]

                # Load displacement from all GLL points.
                for i, var in enumerate(meshes):
                    # The list of ids we have is unique but not sorted.
                    ids = gll_point_ids.flatten()
                    s_ids = np.sort(ids)
                    temp = var[:, s_ids]
                    for ipol in range(npol + 1):
                        for jpol in range(npol + 1):
                            idx = ipol * 5 + jpol
                            utemp[:, jpol, ipol, i] = \
                                temp[:,
                                np.argwhere(s_ids == ids[idx])[0][0]]
                ds_o[gll_idx] = utemp

    finally:
        try:
            f_in_x.close()
        except:
            pass
        try:
            f_in_z.close()
        except:
            pass
        try:
            f_out.close()
        except:
            pass


@click.command()
@click.argument("input_folder", type=click.Path(exists=True, file_okay=False,
                                                dir_okay=True))
@click.argument("output_folder", type=click.Path(exists=False))
@click.option('--method', type=click.Choice(["continuous", "unrolled",
                                             "unrolled_merge"]),
              help="'continuous' will transpose all arrays and store them "
              "unchunked and uncompressed. 'unrolled' will completely unroll "
              "all arrays into one. It is a slow transformation and also "
              "produces a bigger file but Instaseis can read it faster.")
def repack_database(input_folder, output_folder, method):
    found_filenames = []
    for root, _, filenames in os.walk(input_folder):
        for filename in filenames:
            if filename != "ordered_output.nc4":
                continue
            found_filenames.append(os.path.join(root, filename))

    assert found_filenames, "No files named `ordered_output.nc4` found."

    os.makedirs(output_folder)

    # The unrolled merge completely unrolls everything, dededuplicates the GLL
    # points, and merges both netCDF files into one big file.
    if method == "unrolled_merge":
        unroll_and_merge(filenames=found_filenames,
                         output_folder=output_folder)
    else:
        for _i, filename in enumerate(found_filenames):
            click.echo(click.style(
                "--> Processing file %i of %i: %s" %
                (_i + 1, len(found_filenames), filename), fg="green"))

            output_filename = os.path.join(
                output_folder,
                os.path.relpath(filename, input_folder))

            os.makedirs(os.path.dirname(output_filename))

            process_file(filename, output_filename, method=method)


if __name__ == "__main__":
    repack_database()
