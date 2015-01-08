.. only:: not html

    .. image:: http://i.imgur.com/6LNoJD6.png
         :width: 60%

.. only:: html

    .. raw:: html

        <div class="col-lg-4 col-md-5 col-sm-12">
        <img src="http://i.imgur.com/6LNoJD6.png">
        </div>



Instaseis - Instant Global Broadband Seismograms Based on a Waveform Database
=============================================================================

.. only:: html

    .. raw:: html

        <div class="col-md-12">


Instaseis calculates broadband seismograms from Green's function databases
generated with `AxiSEM <http://axisem.info>`_  and allows for near
instantaneous (on the order of milliseconds) extraction of seismograms. Using
the 2.5D axisymmetric spectral element method, the generation of these
databases, based on reciprocity of the Green’s functions, is very efficient and
is approximately half as expensive as a single AxiSEM forward run. Thus this
enables the computation of full databases at half the cost of the computation of
seismograms for a single source in the previous scheme and hence allows to
compute databases at the highest frequencies globally observed. By storing the
basis coefficients of the numerical scheme (Lagrange polynomials), the Green’s
functions are 4th order accurate in space and the spatial discretization
respects discontinuities in the velocity model exactly. On top, AxiSEM allows to
include 2D structure in the source receiver plane and readily includes other
planetes such as Mars.


.. only:: html

    .. raw:: html

        </div>

Installation
------------

Requirements
^^^^^^^^^^^^

Instaseis is implemented as a Python library and has a number of dependencies
listed here. It might well work with other versions but only the versions listed
here are continuously tested and supported. Instaseis currently runs on Linux
and Mac OS X. Adding support for Windows is mainly a question of compiling the
shared Fortran librarys - pull requests are welcome.

* ``gfortran >= 4.7``
* ``Python 2.6, 2.7, 3.3, or 3.4``
* ``NumPy >= 1.7``
* ``ObsPy >= 0.9.2`` *(only the ObsPy master currently supports Python 3)*
* ``netCDF4 >= 4.3`` including Python bindings (``>= 1.1``)
* ``future``
* ``requests``
* ``responses``
* ``tornado``
* ``flake8``
* ``pytest``
* ``mock`` *(only for Python 2.x, otherwise part of the standard library)*

The optional graphical user interface furthermore requires

* ``PyQt4``
* ``pyqtgraph``
* ``matplolitb``
* ``basemap``


Fortran Compiler
~~~~~~~~~~~~~~~~

If you don't have ``gfortran``, please install it (on Linux) with

.. code-block:: bash

    $ sudo apt-get install gfortran

or the equivalent of your distribution. On OSX we recommend to install
`Homebrew <http://brew.sh/>`_ and then use it to install ``gfortran``:

.. code-block:: bash

    $ brew install gcc

Python and Dependencies
~~~~~~~~~~~~~~~~~~~~~~~

If you know what you are doing, just make sure the aforementioned
dependencies are installed. Otherwise do yourself a favor and download the
`Anaconda <https://store.continuum.io/cshop/anaconda/>`_ Python distribution.
It is a free scientific Python distribution bundling almost all necessary
modules with a convenient installer (does not require root access!).
Once installed assert that ``pip`` and ``conda`` point to the Anaconda
installation folder (you may need to open a new terminal after installing
Anaconda).

.. code-block:: bash

    $ conda install netcdf4 future requests tornado flake8 pytest mock basemap pyqt pip
    $ pip install obspy responses pyqtgraph

A possible complication arises if you are running on a server without a
display. In that case please edit (on Linux)
``~/.config/matplotlib/matplotlibrc`` (create if it does not exist) and make
sure the following line is part of it:

.. code-block:: bash

    backend: agg


Installing Instaseis
^^^^^^^^^^^^^^^^^^^^

User Installation
~~~~~~~~~~~~~~~~~

Once released, installation of Instaseis is as easy as

.. code-block:: bash

    $ pip install instaseis

Developer Installation
~~~~~~~~~~~~~~~~~~~~~~

Clone the git repository and install in an editable fashion.

.. code-block:: bash

    $ git clone https://github.com/krischer/instaseis.git
    $ cd instaseis
    $ pip install -v -e .


Testing
^^^^^^^

To assert that your installation is working properly, execute

.. code-block:: bash

    $ python -m instaseis.tests

and make sure all tests pass. Otherwise please contact the developers.

Build the Documentation
^^^^^^^^^^^^^^^^^^^^^^^

The documentation requires ``sphinx`` and the Bootstrap theme. Install both
with

.. code-block:: bash

    $ pip install sphinx sphinx-bootstrap-theme

Build the doc with

.. code-block:: bash

    $ cd doc
    $ make html

Finally open the ``doc/_build/html/index.html`` file with the browser of your
choice.


Tutorial
--------

Learning Python and ObsPy
^^^^^^^^^^^^^^^^^^^^^^^^^

Instaseis is written in `Python <http://www.python.org>`_ and utilizes the
data structures of `ObsPy <http://obspy.org>`_ to allow the construction of
modern and efficient workflows. Python is an easy to learn and powerful
interactive programming language with an exhaustive scientific ecosystem. The
following resources are useful if you are starting out with Python and ObsPy:

* `Good, general Python tutorial <http://learnpythonthehardway.org/book/>`_
* `IPython Notebook in Nature <http://www.nature.com/news/interactive-notebooks-sharing-the-code-1.16261>`_
* `Introduction to the scientific Python ecosystem <https://scipy-lectures.github.io/>`_
* `The ObsPy Documentation <http://docs.obspy.org/master>`_
* `The ObsPy Tutorial <http://docs.obspy.org/master/tutorial/index.html>`_

Using Instaseis
^^^^^^^^^^^^^^^

To use Instaseis you first have to open a connection to an Instaseis database.
Instaseis supports connections to local and remote Green's function databases; a
local database consists of up to four NetCDF files on the filesystem whereas a
remote database requires an Instaseis server answering queries. The usage and
capabilities of both are completely equivalent. This section deals with using
Instaseis to generate seismograms; if you wish to run an Instaseis server,
please see the documentation of the :doc:`server`.

Connecting to either a local or a remote Instaseis database happens with the
:func:`~instaseis.open_db` function. It will return either an
:class:`~instaseis.instaseis_db.InstaseisDB` or a
:class:`~instaseis.remote_instaseis_db.RemoteInstaseisDB` object. Additional
arguments and keyword arguments are passed to the the initialization
function of these objects. Once the database objects are created, usage of
both is identical. Be aware that the initialization of the database objects
is potentially a fairly expensive operation so make sure to do it more often
than necessary (usually once per database).


.. note::

    If opening a local database and the ``ordered_output.nc4`` files are
    located for example in ``/path/to/DB/PZ/Data`` and ``/path/to/DB/PX/Data``,
    please pass ``/path/to/DB`` to the :func:`~instaseis.open_db` function.
    Instaseis will recursively search the child directories for the necessary
    files and open them.


.. code-block:: python

    >>> import instaseis

    >>> # Open connection to a local database by giving the path on disc where
    >>> # the `ordered_output.nc4` files of AxiSEM are stored.
    >>> db = instaseis.open_db("/path/to/10s_PREM")

    >>> # Or open connection to a remote database by giving an HTTP URL.
    >>> db = instaseis.open_db("http://123.456.789.123:1234")

    >>> # Get some basic information by printing the database object.
    >>> print(db)
    RemoteInstaseisDB reciprocal Green's function Database (v7) generated with these parameters:
    components           : vertical and horizontal
    velocity model       : prem_ani
    attenuation          : True
    dominant period      : 10.000 s
    dump type            : displ_only
    excitation type      : dipole
    time step            : 2.436 s
    sampling rate        : 0.411 Hz
    number of samples    : 739
    seismogram length    : 1797.7 s
    source time function : errorf
    source shift         : 17.051 s
    spatial order        : 4
    min/max radius       : 5700.0 - 6371.0 km
    Planet radius        : 6371.0 km
    min/max distance     : 0.0 - 180.0 deg
    time stepping scheme : newmark2
    compiler/user        : gfortran 4.9.1 by lion on Laptop
    directory/url        : http://123.456.789.123:1234
    size of netCDF files : 3.4 GB
    generated by AxiSEM version 60945ec at 2014-10-23T21:32:58.000000Z


The :meth:`~instaseis.base_instaseis_db.BaseInstaseisDB.get_seismograms()`
method is called to generate seismograms as :class:`~obspy.core.stream.Stream`
objects with the waveform data. This enables easy serialization in a large
selection of formats and facilitates post processing by utilizing ObsPy.

To generate seismograms source information must be given either as a
:class:`~instaseis.source.Source` object, a
:class:`~instaseis.source.ForceSource` object, or as a
:class:`~instaseis.source.FiniteSource` object. Receiver information must be
passed in form of a :class:`~instaseis.source.Receiver` object. Please refer
to the documentation of these classes for more details.


.. code-block:: python

    >>> import obspy
    >>> receiver = instaseis.Receiver(
    ...     latitude=42.6390, longitude=74.4940, network="AB", station="CED")
    >>> source = instaseis.Source(
    ...     latitude=89.91, longitude=0.0, depth_in_m=12000,
    ...     m_rr = 4.710000e+24 / 1E7,
    ...     m_tt = 3.810000e+22 / 1E7,
    ...     m_pp =-4.740000e+24 / 1E7,
    ...     m_rt = 3.990000e+23 / 1E7,
    ...     m_rp =-8.050000e+23 / 1E7,
    ...     m_tp =-1.230000e+24 / 1E7,
    ...     origin_time=obspy.UTCDateTime(2011, 1, 2, 3, 4, 5))
    >>> st = db.get_seismograms(source=source, receiver=receiver)
    >>> print(st)
    3 Trace(s) in Stream:
    AB.CED..MXZ | 2011-01-02T03:04:05Z - ... | 0.4 Hz, 732 samples
    AB.CED..MXN | 2011-01-02T03:04:05Z - ... | 0.4 Hz, 732 samples
    AB.CED..MXE | 2011-01-02T03:04:05Z - ... | 0.4 Hz, 732 samples


Source and receiver can also be replaced by ObsPy objects. This can be used
to calculate synthetics based on standard file formats and web services.

.. code-block:: python

    >>> # Read event information from a local QuakeML file.
    >>> cat = obspy.readEvents("quake.xml")
    >>> print(cat)
    1 Event(s) in Catalog:
    2010-03-11T06:22:20.100000Z | -57.460,  -27.580 | 5.6 Mwc

    >>> # Query a web service for station information.
    >>> from obspy.fdsn import Client
    >>> c = Client("IRIS")
    >>> inv = c.get_stations(network="IU", station="ANMO", level="station",
    ...                      starttime=cat[0].origins[0].time)
    >>> print(inv)
    Created by: IRIS WEB SERVICE: fdsnws-station | ...
    Contains:
        Networks (1): IU
        Stations (1): IU.ANMO (Albuquerque, New Mexico, USA)

    >>> st = db.get_seismograms(source=cat, receiver=inv)
    >>> print(st)
    3 Trace(s) in Stream:
    IU.ANMO..MXZ | 2010-03-11T06:22:20Z - ... | 0.4 Hz, 732 samples
    IU.ANMO..MXN | 2010-03-11T06:22:20Z - ... | 0.4 Hz, 732 samples
    IU.ANMO..MXE | 2010-03-11T06:22:20Z - ... | 0.4 Hz, 732 samples

GUI
---

Instaseis contains an optional graphical user interface which is useful to
explore a database and for educational purposes. To launch it just type

.. code-block:: bash

    $ python -m instaseis.gui

.. image:: http://http://i.imgur.com/FfEtlCU.png

Screenshot of the Instaseis graphical user interface (GUI). Aside from quickly
exploring the characteristics of a given Green’s function database it is a
great tool for understanding and teaching many aspect of seismograms. The speed
of Instaseis enables an immediate visual response to changing source and
receiver parameters. The left hand side shows three component seismograms where
theoretical arrival times of various seismic phases are overlaid as vertical
lines. The bar at the top is used to change filter and resampling settings and
the section on the right side is used to modify source and receiver parameters.


Detailed Documentation
----------------------

.. toctree::
   :maxdepth: 2

   instaseis
   mesh
   server
   source
