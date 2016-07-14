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
import contextlib
import math
import os

import click
import netCDF4
import numpy as np


@contextlib.contextmanager
def dummy_progressbar(iterator, *args, **kwargs):
    yield iterator


def repack_file(input_filename, output_filename, contiguous,
                compression_level, transpose, quiet=False):
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
                       compression_level=compression_level, quiet=quiet,
                       transpose=transpose)


def recursive_copy(src, dst, contiguous, compression_level, transpose, quiet):
    """
    Recursively copy the whole file and transpose the all /Snapshots
    variables while at it..
    """
    if src.path == "/Seismograms":
        return

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
    if is_snap and transpose:
        items = list(reversed(items))

    for name, dimension in items:
        dst.createDimension(name, len(
            dimension) if not dimension.isunlimited() else None)

    _j = 0
    for name, variable in src.variables.items():
        _j += 1
        shape = variable.shape

        # Determine chunking - only for the snapshots.
        if is_snap and name.startswith("disp_"):
            npts = min(shape)
            num_elems = max(shape)
            time_axis = np.argmin(shape)
            # Arbitrary limit.
            _c = int(round(32768 / (npts * 4)))

            if time_axis == 0 and transpose:
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
        if is_snap and transpose:
            dimensions = list(reversed(dimensions))

        x = dst.createVariable(name, variable.datatype, dimensions,
                               chunksizes=chunksizes, contiguous=contiguous,
                               zlib=zlib, complevel=compression_level)
        # Non-snapshots variables are just copied in a single go.
        if not is_snap or not name.startswith("disp_"):
            if not quiet:
                click.echo(click.style("\tCopying group '%s'..." % name,
                                       fg="blue"))
            dst.variables[x.name][:] = src.variables[x.name][:]
        # The snapshots variables are incrementally copied and transposed.
        else:
            if not quiet:
                click.echo(click.style(
                    "\tCopying 'Snapshots/%s' (%i of %i)..." % (
                        name, _j,
                        len([_i for _i in src.variables
                             if _i.startswith("disp_")])),
                    fg="blue"))

            # Copy around 8 Megabytes at a time. This seems to be the
            # sweet spot at least on my laptop.
            factor = int((8 * 1024 * 1024 / 4) / npts)
            s = int(math.ceil(num_elems / float(factor)))

            if quiet:
                pbar = dummy_progressbar
            else:
                pbar = click.progressbar

            with pbar(range(s), length=s, label="\t  ") as idx:
                for _i in idx:
                    _s = slice(_i * factor, _i * factor + factor)
                    if transpose:
                        if time_axis == 0:
                            dst.variables[x.name][_s, :] = \
                                src.variables[x.name][:, _s].T
                        else:
                            dst.variables[x.name][:, _s] = \
                                src.variables[x.name][_s, :].T
                    else:
                        if time_axis == 0:
                            dst.variables[x.name][_s, :] = \
                                src.variables[x.name][_s, :]
                        else:
                            dst.variables[x.name][:, _s] = \
                                src.variables[x.name][:, _s]

    for src_group in src.groups.values():
        dst_group = dst.createGroup(src_group.name)
        recursive_copy(src=src_group, dst=dst_group, contiguous=contiguous,
                       compression_level=compression_level, quiet=quiet,
                       transpose=transpose)


def recursive_copy_no_snapshots_no_seismograms_no_surface(
        src, dst, quiet, contiguous, compression_level):
    """
    A bit of a copy of the recursive_copy function but it does not copy the
    Snapshots, Seismograms, or Surface group.
    """
    for attr in src.ncattrs():
        _s = getattr(src, attr)
        if isinstance(_s, str):
            dst.setncattr_string(attr, _s)
        else:
            setattr(dst, attr, _s)

    items = list(src.dimensions.items())

    for name, dimension in items:
        dst.createDimension(name, len(
            dimension) if not dimension.isunlimited() else None)

    for name, variable in src.variables.items():
        if name in ["Snapshots", "Seismograms", "Surface"]:
            continue

        # Use the existing chunking.
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

        x = dst.createVariable(name, variable.datatype, dimensions,
                               chunksizes=chunksizes, contiguous=contiguous,
                               zlib=zlib, complevel=compression_level)
        if not quiet:
            click.echo(click.style("\tCopying group '%s'..." % name,
                                   fg="blue"))
        dst.variables[x.name][:] = src.variables[x.name][:]

    for src_group in src.groups.values():
        if src_group.name in ["Snapshots", "Seismograms", "Surface"]:
            continue
        dst_group = dst.createGroup(src_group.name)
        recursive_copy_no_snapshots_no_seismograms_no_surface(
            src=src_group, dst=dst_group, contiguous=contiguous,
            compression_level=compression_level, quiet=quiet)


def merge_files(filenames, output_folder, contiguous, compression_level,
                quiet):
    """
    Completely unroll and merge both files to a single database.
    """
    # Find PX and PZ files.
    assert 1 <= len(filenames) <= 2
    filenames = [os.path.normpath(_i) for _i in filenames]
    px = [_i for _i in filenames if "PX" in _i]
    pz = [_i for _i in filenames if "PZ" in _i]
    assert len(px) <= 1
    assert len(pz) <= 1
    if px:
        px = px[0]
        assert os.path.exists(px)
    else:
        px = None
    if pz:
        pz = pz[0]
        assert os.path.exists(pz)
    else:
        pz = None

    # One must exist.
    assert px or pz

    output = os.path.join(output_folder, "merged_output.nc4")
    assert not os.path.exists(output)

    if px and pz:
        with netCDF4.Dataset(px, "r", format="NETCDF4") as px_in, \
                netCDF4.Dataset(pz, "r", format="NETCDF4") as pz_in, \
                netCDF4.Dataset(output, "w", format="NETCDF4") as out:

            _merge_files(px_in=px_in, pz_in=pz_in, out=out,
                         contiguous=contiguous,
                         compression_level=compression_level, quiet=quiet)
    elif pz and not px:
        with netCDF4.Dataset(pz, "r", format="NETCDF4") as pz_in, \
                netCDF4.Dataset(output, "w", format="NETCDF4") as out:

            _merge_files(px_in=None, pz_in=pz_in, out=out,
                         contiguous=contiguous,
                         compression_level=compression_level, quiet=quiet)
    elif px and not pz:
        with netCDF4.Dataset(px, "r", format="NETCDF4") as px_in, \
                netCDF4.Dataset(output, "w", format="NETCDF4") as out:

            _merge_files(px_in=px_in, pz_in=None, out=out,
                         contiguous=contiguous,
                         compression_level=compression_level, quiet=quiet)
    else:  # pragma: no cover
        raise NotImplementedError


def _merge_files(px_in, pz_in, out, contiguous, compression_level, quiet):
    # First copy everything non-snapshot related.
    if px_in:
        c_db = px_in
    else:
        c_db = pz_in
    recursive_copy_no_snapshots_no_seismograms_no_surface(
        src=c_db, dst=out, quiet=quiet, contiguous=contiguous,
        compression_level=compression_level)

    if contiguous:
        zlib = False
    else:
        zlib = True

    # We need the stf_dump and stf_d_dump datasets. They are either in the
    # "Snapshots" group or in the "Surface" group.
    for g in ("Snapshots", "Surface"):
        if g not in c_db.groups:
            continue
        if "stf_dump" not in c_db[g].variables:
            continue
        break
    else:
        raise Exception("Could not find `stf_dump` array.")

    stf_dump = c_db[g]["stf_dump"]
    stf_d_dump = c_db[g]["stf_d_dump"]

    for data in [stf_dump, stf_d_dump]:
        chunksizes = data.shape
        if contiguous:
            chunksizes = None
        d = out.createVariable(
            varname=data.name,
            dimensions=["snapshots"],
            contiguous=contiguous,
            zlib=zlib,
            chunksizes=chunksizes,
            datatype=data.dtype)
        d[:] = data[:]

    # Get all the snapshots from the other databases.
    if px_in and pz_in:
        meshes = [
            px_in["Snapshots"]["disp_s"],
            px_in["Snapshots"]["disp_p"],
            px_in["Snapshots"]["disp_z"],
            pz_in["Snapshots"]["disp_s"],
            pz_in["Snapshots"]["disp_z"]]
    elif px_in and not pz_in:
        meshes = [
            px_in["Snapshots"]["disp_s"],
            px_in["Snapshots"]["disp_p"],
            px_in["Snapshots"]["disp_z"]]
    elif pz_in and not px_in:
        meshes = [
            pz_in["Snapshots"]["disp_s"],
            pz_in["Snapshots"]["disp_z"]]
    else:  # pragma: no cover
        raise NotImplementedError

    time_axis = np.argmin(meshes[0].shape)

    dtype = meshes[0].dtype

    # Create new dimensions.
    dim_ipol = out.createDimension("ipol", 5)
    dim_jpol = out.createDimension("jpol", 5)
    dim_nvars = out.createDimension("nvars", len(meshes))
    nelem = out.getncattr("nelem_kwf_global")
    dim_elements = out.createDimension("elements", nelem)

    # New dimensions for the 5D Array.
    dims = (dim_elements, dim_nvars, dim_jpol, dim_ipol,
            out.dimensions["snapshots"])
    dimensions = [_i.name for _i in dims]

    if contiguous:
        chunksizes = None
    else:
        chunksizes = [_i.size for _i in dims]

    # We'll called it MergedSnapshots
    x = out.createVariable(
        varname="MergedSnapshots",
        dimensions=dimensions,
        contiguous=contiguous,
        zlib=zlib,
        chunksizes=chunksizes,
        datatype=dtype)

    utemp = np.zeros([_i.size for _i in dims[1:]], dtype=dtype, order="C")

    # Now it becomes more interesting and very slow.
    sem_mesh = c_db["Mesh"]["sem_mesh"]
    with click.progressbar(range(nelem), length=nelem, label="\t  ") as idx:
        for gll_idx in idx:
            gll_point_ids = sem_mesh[gll_idx]

            # Load displacement from all GLL points.
            for i, var in enumerate(meshes):
                # The list of ids we have is unique but not sorted.
                ids = gll_point_ids.flatten()
                s_ids = np.sort(ids)
                if time_axis == 0:
                    temp = var[:, s_ids]
                    for jpol in range(dim_jpol.size):
                        for ipol in range(dim_ipol.size):
                            idx = ipol * 5 + jpol
                            utemp[i, jpol, ipol, :] = \
                                temp[:, np.argwhere(s_ids == ids[idx])[0][0]]
                else:
                    temp = var[s_ids, :]
                    for jpol in range(dim_jpol.size):
                        for ipol in range(dim_ipol.size):
                            idx = ipol * 5 + jpol
                            utemp[i, jpol, ipol, :] = \
                                temp[np.argwhere(s_ids == ids[idx])[0][0], :]
            x[gll_idx] = utemp


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
@click.option('--method', type=click.Choice(["transposed", "repack", "merge"]),
              required=True,
              help="`transposed` will transpose the data arrays which "
                   "oftentimes results in faster extraction times. `repack` "
                   "will just repack the data and solve some compatibility "
                   "issues. `merge` will create a single much larger file "
                   "which is much quicker to read but will take more space.")
def repack_database(input_folder, output_folder, contiguous,
                    compression_level, method):
    found_filenames = []
    for root, _, filenames in os.walk(input_folder):
        for filename in sorted(filenames, reverse=True):
            if filename not in ["ordered_output.nc4", "axisem_output.nc4"]:
                continue
            found_filenames.append(os.path.join(root, filename))
            break

    assert found_filenames, "No files named `ordered_output.nc4` found."

    os.makedirs(output_folder)

    if method in ["transposed", "repack"]:
        for _i, filename in enumerate(found_filenames):
            click.echo(click.style(
                "--> Processing file %i of %i: %s" %
                (_i + 1, len(found_filenames), filename), fg="green"))

            output_filename = os.path.join(
                output_folder,
                os.path.relpath(filename, input_folder))

            output_filename = output_filename.replace(
                "axisem_output.nc4", "ordered_output.nc4")

            os.makedirs(os.path.dirname(output_filename))

            if method == "transposed":
                transpose = True
            else:
                transpose = False

            repack_file(input_filename=filename,
                        output_filename=output_filename,
                        contiguous=contiguous,
                        transpose=transpose,
                        compression_level=compression_level)
    elif method == "merge":
        merge_files(filenames=found_filenames, output_folder=output_folder,
                    contiguous=contiguous, compression_level=compression_level,
                    quiet=False)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    repack_database()
