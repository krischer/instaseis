=============================
Database Layout and Repacking
=============================

Instaseis supports a number of different database layouts, each with its own
advantages and disadvantages as well as repacking functionality to convert
from some layouts to others.

.. contents::
    :local:


Available Layouts
-----------------

From a high-level view Instaseis supports two database layouts which we will
call the *multi file layout* and the *merged layout* in the following. Both
support two components (horizontal + vertical) as well as single component
databases. The multi file layout optionally also support forward databases
which are currently not explained in more detail.

Multi File Layout
^^^^^^^^^^^^^^^^^

This consists of two NetCDF files - one can optionally be neglected -
Instaseis will then only be able to extract horizontal or vertical seismograms.
Its stores the snapshots in time for two (vertical) or three (horizontal)
components per GLL point. GLL points shared between two elements are only
stored once which quite significantly reduces the total files size (only
about 16 GLL points needs to be stored per element instead of 25).

The big downside is that, to extract three components seismograms, it needs to
read all 25 GLL points for a single element from all 3 + 2 displacement
snapshots. Instaseis is being smart about it and batches adjacent reads but
worst case this means that 125 single read accesses across two different files
have to be performed to get data from a single element.

**Expected NetCDF file locations:**

* For the horizontal data file: ``ROOT/.../PX/../ordered_output.nc4`` or
  ``ROOT/../PX/.../axisem_output.nc4``
* For the vertical data file: ``ROOT/.../PZ/../ordered_output.nc4`` or
  ``ROOT/.../PZ/.../axisem_output.nc4``


**File layout** (in a commented representation based on the output from
``ncdump``)::

    dimensions:
        gllpoints_all = 160692 ;
        snapshots = 370 ;

    // required global attributes (e.g. these are used by Instaseis):
    * "dump type (displ_only, displ_velo, fullfields)" (string)
    * "excitation_type" (string)
    * "source type" (string)
    * "background model" (string)
    * "external model name" (string)
    * "git commit hash" (string)
    * "datetime" (string)
    * "compiler brand" (string)
    * "compiler version" (string)
    * "user name" (string)
    * "host name" (string)
    * "time scheme" (string)
    * "source time function" (string)
    * "npol" (int)
    * "file version" (int)
    * "number of strain dumps" (int)
    * "scalar source magnitude" (float/double/real)
    * "strain dump sampling rate in sec" (float/double/real)
    * "source shift factor in sec" (float/double/real)
    * "source shift factor for deltat_coarse" (int)
    * "npoints" (int)
    * "attenuation" (int - 1 is true/0 is false)
    * "planet radius" (float/souble/real)
    * "dominant source period" (float/souble/real)
    * "kernel wavefield rmin" (float/souble/real)
    * "kernel wavefield rmax" (float/souble/real)
    * "kernel wavefield colatmin" (float/souble/real)
    * "kernel wavefield colatmax" (float/souble/real)
    * "source depth in km" (float/souble/real)
    * "nelem_kwf_global" (int)

    group: Snapshots {
      variables:

        # Note that these can exist in both version - this one and the
        # transposed one. If you use this one, make sure the chunking is set
        # up in a way that the snapshots for a single GLL point can be read
        # at once - otherwise performance will be abysmal. The transposed
        # version does not have this problem.
        float disp_s(snapshots, gllpoints_all) ;
        # Not needed for for vertical databases.
        float disp_p(snapshots, gllpoints_all) ;
        float disp_z(snapshots, gllpoints_all) ;

        # These two can (for legacy reasons) also be part of a top level
        # "Surface" group.
        float stf_dump(snapshots) ;
        float stf_d_dump(snapshots) ;
      }

    group: Mesh {
      dimensions:
        elements = 9856 ;
        control_points = 4 ;
        npol = 5 ;
      variables:
        int midpoint_mesh(elements) ;
        int eltype(elements) ;
        int axis(elements) ;
        int fem_mesh(elements, control_points) ;
        int sem_mesh(elements, npol, npol) ;
        float mp_mesh_S(elements) ;
        float mp_mesh_Z(elements) ;
        double G0(npol) ;
        double G1(npol, npol) ;
        double G2(npol, npol) ;
        double gll(npol) ;
        double glj(npol) ;
        float mesh_S(gllpoints_all) ;
        float mesh_Z(gllpoints_all) ;
        float mesh_vp(gllpoints_all) ;
        float mesh_vs(gllpoints_all) ;
        float mesh_rho(gllpoints_all) ;
        float mesh_lambda(gllpoints_all) ;
        float mesh_mu(gllpoints_all) ;
        float mesh_xi(gllpoints_all) ;
        float mesh_phi(gllpoints_all) ;
        float mesh_eta(gllpoints_all) ;
        float mesh_Qmu(gllpoints_all) ;
        float mesh_Qka(gllpoints_all) ;
      }


Merged File Layout
^^^^^^^^^^^^^^^^^^

This, in contrast to the *multi file layout* stores everything in a single
5D array, meaning data from one element can be accessed with a single read
command. The downside is that many GLL points are duplicated which thus
increases the file size. On the other hand this layout can easily increase
the performance by more than an order of magnitude so depending on the use
case this is the way to go. Turning on compression can save quite a lot of
space here but comes at the expense of some speed. Make sure to set the
chunking in a way that each chunk corresponds to all the data from a single
element.

**Expected NetCDF file locations:** ``ROOT/.../merged_output.nc4``

**File layout** (in a commented representation based on the output from
``ncdump``)::

    # Global attributes and mesh the same as above!

    dimensions:
            gllpoints_all = 160692 ;
            snapshots = 370 ;
            ipol = 5 ;
            jpol = 5 ;
            nvars = 5 ;
            elements = 9856 ;
    variables:
            float stf_dump(snapshots) ;
            float stf_d_dump(snapshots) ;
            float MergedSnapshots(elements, nvars, jpol, ipol, snapshots) ;


The second dimension in the ``MergedSnapshots`` variable corresponds to the
displacement in the various directions. In terms of the *multi file layout*,
Instaseis assumes the following order:

**5D => horizontal and vertical database:**

1. ``disp_s horizontal``
2. ``disp_p horizontal``
3. ``disp_z horizontal``
4. ``disp_s vertical``
5. ``disp_z vertical``

**3D => horizontal only database:**

1. ``disp_s horizontal``
2. ``disp_p horizontal``
3. ``disp_z horizontal``

**2D => vertical only database:**

1. ``disp_s vertical``
2. ``disp_z vertical``



Repacking Script
----------------

Instaseis can convert databases from the *multi file layout* (also in the
form that AxiSEM produces directly) to:

* The same layout - (the `repack` method) - this sometimes improves
  compatibility. Additionally compression settings can be changed.
* A transposed version of the same layout - this might improve the
  performance. Running this more than one time will keep transposing the data
  arrays.
* The merged layout. Conversion can take a very long time. Compression is
  also able to save quite a bit of space.


.. code-block:: bash


    $ python -m instaseis.scripts.repack_db --help

    Usage: repack_db.py [OPTIONS] INPUT_FOLDER OUTPUT_FOLDER
    Options:
      --contiguous                    Write a contiguous array - will turn off
                                      chunking and compression
      --compression_level INTEGER RANGE
                                      Compression level from 1 (fast) to 9 (slow).
      --method [transpose|repack|merge]
                                      `transpose` will transpose the data arrays
                                      which oftentimes results in faster
                                      extraction times. `repack` will just repack
                                      the data and solve some compatibility
                                      issues. `merge` will create a single much
                                      larger file which is much quicker to read
                                      but will take more space.  [required]
      --help                          Show this message and exit.


Comparing Databases
-------------------

If you don't trust the repacking script, don't fret - there is another
script that compares two or more databases to make sure they produce the same
waveforms:


.. code-block:: bash

    $ python -m instaseis.scripts.compare_dbs --help

    Usage: compare_dbs.py [OPTIONS] [DATABASES]...

      Pass a list of databases to assert that they produce the same seismograms.
      The first one will be treated as the reference.

    Options:
      --seed INTEGER  Optionally pass a seed number to make it reproducible.
      --help          Show this message and exit.
