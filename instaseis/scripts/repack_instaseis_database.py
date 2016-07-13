#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Repacking Instaseis databases.

Requires click, netCDF4, and numpy.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2016
:license:
    GNU Lesser General Public License, Version 3 [non-commercial/academic use]
    (http://www.gnu.org/copyleft/lgpl.html)
"""
import math
import os

import click
import netCDF4
import numpy as np


def transpose_data(input_filename, output_filename, contiguous,
                   compression_level):
    """
    Transposes all data in the "/Snapshots" group.

    :param input_filename: The input filename.
    :param output_filename: The output filename.
    """
    assert os.path.exists(input_filename)
    assert not os.path.exists(output_filename)

    with netCDF4.Dataset(input_filename, "r", format="NETCDF4") as f_in, \
            netCDF4.Dataset(output_filename, "w", format="NETCDF4") as f_out:
        recursive_copy(src=f_in, dst=f_out, contiguous=contiguous,
                       compression_level=compression_level)


def recursive_copy(src, dst, contiguous, compression_level):
    """
    Recursively copy the whole file and transpose the all /Snapshots
    variables while at it..
    """
    for attr in src.ncattrs():
        _s = getattr(src, attr)
        if isinstance(_s, str):
            dst.setncattr_string(attr, _s)
        else:
            setattr(dst, attr, _s)

    # We will only transpose the Snapshots group.
    if src.path == "/Snapshots":
        is_snap = True
    else:
        is_snap = False

    # The dimensions will be reversed for the snapshots.
    items = list(src.dimensions.items())
    if is_snap:
        items = list(reversed(items))

    for name, dimension in items:
        dst.createDimension(name, len(
            dimension) if not dimension.isunlimited() else None)

    _j = 0
    for name, variable in src.variables.items():
        _j += 1
        shape = variable.shape

        # Determine chunking - only for the snapshots.
        if is_snap:
            npts = min(shape)
            num_elems = max(shape)
            time_axis = np.argmin(shape)
            # Arbitrary limit.
            _c = int(round(32768 / (npts * 4)))

            if time_axis == 0:
                chunksizes = (_c, npts)
            else:
                chunksizes = (npts, _c)
        else:
            # For non-snapshots, just use the existing chunking.
            chunksizes = variable.chunking()
            # We could infer the chunking here but I'm not sure its worth it.
            if isinstance(chunksizes, str) and chunksizes == "contiguous":
                chunksizes = None

        # For a contiguous output, compression and chunking has to be turned
        # off.
        if contiguous:
            zlib = False
            chunksizes = None
        else:
            zlib = True

        dimensions = variable.dimensions
        if is_snap:
            dimensions = list(reversed(dimensions))

        x = dst.createVariable(name, variable.datatype, dimensions,
                               chunksizes=chunksizes, contiguous=contiguous,
                               zlib=zlib, complevel=compression_level)
        # Non-snapshots variables are just copied in a single go.
        if not is_snap:
            click.echo(click.style("\tCopying group '%s'..." % name,
                                   fg="blue"))
            dst.variables[x.name][:] = src.variables[x.name][:]
        # The snapshots variables are incrementally copied and transposed.
        else:
            click.echo(click.style(
                "\tCopying 'Snapshots/%s' (%i of %i)..." % (
                    name, _j, len(src.variables)),
                fg="blue"))

            # Copy around 8 Megabytes at a time. This seems to be the
            # sweet spot at least on my laptop.
            factor = int((8 * 1024 * 1024 / 4) / npts)
            s = int(math.ceil(num_elems / float(factor)))

            with click.progressbar(range(s), length=s,
                                   label="\t  ") as idx:
                for _i in idx:
                    _s = slice(_i * factor, _i * factor + factor)
                    if time_axis == 0:
                        dst.variables[x.name][_s, :] = \
                            src.variables[x.name][:, _s].T
                    else:
                        dst.variables[x.name][:, _s] = \
                            src.variables[x.name][_s, :].T

    for src_group in src.groups.values():
        dst_group = dst.createGroup(src_group.name)
        recursive_copy(src=src_group, dst=dst_group, contiguous=contiguous,
                       compression_level=compression_level)


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

    import h5py

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

        # Attempt to copy other things.
        sn = f_in_z["Snapshots"]
        f_out.create_group("Snapshots")
        for group in sn.keys():
            if not group.startswith("disp_"):
                f_out.copy(f_in_z["Snapshots"][group], "Snapshots/%s" % group)

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
                                temp[:, np.argwhere(s_ids == ids[idx])[0][0]]
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
@click.option("--contiguous", is_flag=True,
              help="Write a contiguous array - will turn off chunking and "
                   "compression")
@click.option("--compression_level",
              type=click.IntRange(1, 9), default=2,
              help="Compression level from 1 (fast) to 9 (slow).")
@click.option('--method', type=click.Choice(["transposed", "merged"]),
              required=True,
              help="'contiguous' will transpose all arrays and store them "
              "unchunked and uncompressed. 'unrolled' will completely unroll "
              "all arrays into one. It is a slow transformation and also "
              "produces a bigger file but Instaseis can read it faster.")
def repack_database(input_folder, output_folder, contiguous,
                    compression_level, method):
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
    # if method == "unrolled_merge":
    #     unroll_and_merge(filenames=found_filenames,
    #                      output_folder=output_folder)
    if method == "transposed":
        for _i, filename in enumerate(found_filenames):
            click.echo(click.style(
                "--> Processing file %i of %i: %s" %
                (_i + 1, len(found_filenames), filename), fg="green"))

            output_filename = os.path.join(
                output_folder,
                os.path.relpath(filename, input_folder))

            os.makedirs(os.path.dirname(output_filename))

            transpose_data(input_filename=filename,
                           output_filename=output_filename,
                           contiguous=contiguous,
                           compression_level=compression_level)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    repack_database()
