Instaseis - Instant high frequency seismograms
==============================================

Instaseis calculates high frequency seismograms from Greens function databases
calculated with AxiSEM.

Requirements
------------

Instaseis has the following dependencies. It might well work with other
versions but that has not been tested and we do not officially support it.

* ``gfortran >= 4.7``
* ``Python 2.7``
* ``NumPy >= 1.7``
* ``ObsPy >= 0.9.2``
* ``netCDF4 >= 4.3`` including Python bindings (``>= 1.1``)

The graphical user interface (which is not needed to run Instaseis) furthermore requires:

* ``PyQt4``
* ``pyqtgraph``
* ``matplolitb``
* ``basemap``

Furthermore the tests require

* ``pytest``
* ``flake8``

Fortran Compiler
^^^^^^^^^^^^^^^^

If you don't have ``gfortran``, please install it (on Linux) with

.. code-block:: bash

    $ sudo apt-get install gfortran

or the equivalent of your distribution. On OSX we recommend to install
`Homebrew <http://brew.sh/>`_ and then use it to install ``gfortran``.

.. code-block:: bash

    $ brew install gcc

Python and Dependencies
^^^^^^^^^^^^^^^^^^^^^^^

If you know what you are doing, just make sure the aforementioned
dependencies are installed. Otherwise do yourself a favor and download the
`Anaconda <https://store.continuum.io/cshop/anaconda/>`_ Python distribution.
It is free scientific Python distribution bundling almost all necessary
modules with a convenient installer (does not require root access!). Make sure
to use the distribution for Python 2.7. Once installed assert that ``pip`` and
``conda`` point to the Anaconda installation folder (you need to open a new
terminal after installing Anaconda) and install the missing dependencies with

.. code-block:: bash

    $ pip install obspy
    $ pip install pyqtgraph pytest flake8
    $ conda install basemap


Installing Instaseis
--------------------

Once all dependencies have been installed, you can now install Instaseis.

User Installation
^^^^^^^^^^^^^^^^^

Once released, installation of Instaseis is as easy as

.. code-block::

    $ pip install instaseis

Developer Installation
^^^^^^^^^^^^^^^^^^^^^^

Clone the git repository and install in an editable fashion.

.. code-block:: bash

    $ git clone https://github.com/krischer/instaseis.git
    $ cd instaseis
    $ pip install -v -e .


Testing
-------

Tests are executed with

.. code-block:: bash

    $ python -m instaseis.tests


Build the Documentation
-----------------------

The documentation requires sphinx and the Bootstrap theme. Install both with

.. code-block:: bash

    $ pip install sphinx sphinx-bootstrap-theme

Build the doc with

.. code-block:: bash

    $ cd doc
    $ make html

Finally open the ``doc/_build/html/index.html`` file with the browser of your
choice or host it somewhere.


Usage
-----

Use it by creating an :class:`~instaseis.instaseis.InstaSeisDB` object which needs
to know the path to the output folder containing the netCDF files from an
AxiSEM run. All information is determined from the netCDF files so no other
files have to be present.

Then a source and receiver pair is defined. The moment tensor components are in
``N m``. Lastly the :meth:`~instaseis.instaseis.InstaSeisDB.get_seismograms()`
method is called which by default returns a three component ObsPy
:class:`~obspy.core.stream.Stream` object with the waveform data. This can then
be used process and save the data in any format supported by ObsPy.

The initialization of an :class:`~instaseis.instaseis.InstaSeisDB` object is the
most expensive part so make sure to do it only once if possible.

.. code-block:: python

    In [1]: from instaseis import InstaSeisDB, Source, Receiver

    In [2]: db = InstaSeisDB("./prem50s_forces")

    In [3]: receiver = Receiver(latitude=42.6390, longitude=74.4940)

    In [4]: source = Source(
       ...:     latitude=89.91, longitude=0.0, depth_in_m=12000,
       ...:     m_rr = 4.710000e+24 / 1E7,
       ...:     m_tt = 3.810000e+22 / 1E7,
       ...:     m_pp =-4.740000e+24 / 1E7,
       ...:     m_rt = 3.990000e+23 / 1E7,
       ...:     m_rp =-8.050000e+23 / 1E7,
       ...:     m_tp =-1.230000e+24 / 1E7)

    In [5]: st = db.get_seismograms(source=source, receiver=receiver)

    In [6]: print st
    3 Trace(s) in Stream:
    ...BXZ | 1970-01-01T00:00:00.000000Z - ... | 12.5 s, 144 samples
    ...BXN | 1970-01-01T00:00:00.000000Z - ... | 12.5 s, 144 samples
    ...BXE | 1970-01-01T00:00:00.000000Z - ... | 12.5 s, 144 samples

GUI
---

The GUI is just an experimental feature and will likely crash for some reason. To try it, just execute the correct file

.. code-block:: bash

    $ python instaseis/experimental_gui/instaseis_gui.py

.. image:: http://i.imgur.com/FVmua2X.png


Detailed Documentation
----------------------

.. toctree::
   :maxdepth: 2

   instaseis
   mesh
   source
   srf
