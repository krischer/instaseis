## Python Interface to AxiSEM's netCDF Databases


### Requirements

* gfortran >= 4.7

It has only been tested with Python 2.7 and requires the following Python modules:

* netCDF4
* ObsPy
* numpy

The experimental GUI furthermore requires

* PyQt4
* pyqtgraph
* matplolitb
* basemap


If you are new to Python, first make sure to have a Fortran compiler. On Linux just run

```bash
$ sudo apt-get install gfortran
```

or the equivalent of your distribution. On OSX I recommend to install [Homebrew](http://brew.sh/) and then type

```bash
$ brew install gcc
```

Finally download [Anaconda](https://store.continuum.io/cshop/anaconda/) for Python version 2.7 and finish the installation of the requirements with

```bash
$ pip install obspy
$ pip install pyqtgraph
$ conda install basemap
```


### Installation

Install it by cloning the git repository and installing with an editable installation. The Makefile currently has to be executed to build the shared library until a proper installation routine has been written.

```bash
$ git clone https://github.com/krischer/axisem_db.git
$ cd axisem_db
$ pip install -v -e .
$ cd axisem_db
$ make
```

### Testing

Testing right now is fairly minimal and machine dependent...

To test, make sure pytest is installed

```bash
$ pip install pytest
```

and execute

```bash
$ py.test
```

in the modules source code directory. The command will discover and execute all
defined tests.


### Usage

Use it by creating an `AxiSEMDB` object which needs to know the path to the
output folder containing the netCDF files from an AxiSEM run. All information
is determined from the netCDF files so no other files have to be present.

Then a source and receiver pair is defined. The moment tensor components are in
`N m`. Lastly the `get_seismograms()` method is called which by default returns
a three component ObsPy `Stream` object with the waveform data. This can then
be used process and save the data in any format supported by ObsPy.

The initialization of an `AxiSEMDB` object is the most expensive part so make
sure to do it only once if possible.

```python
In [1]: from axisem_db import AxiSEMDB, Source, Receiver

In [2]: axisem_db = AxiSEMDB("./prem50s_forces")

In [3]: receiver = Receiver(latitude=42.6390, longitude=74.4940)

In [4]: source = Source(
   ...:     latitude=89.91, longitude=0.0, depth_in_m=12000,
   ...:     m_rr = 4.710000e+24 / 1E7,
   ...:     m_tt = 3.810000e+22 / 1E7,
   ...:     m_pp =-4.740000e+24 / 1E7,
   ...:     m_rt = 3.990000e+23 / 1E7,
   ...:     m_rp =-8.050000e+23 / 1E7,
   ...:     m_tp =-1.230000e+24 / 1E7)

In [5]: st = axisem_db.get_seismograms(source=source, receiver=receiver)

In [6]: print st
3 Trace(s) in Stream:
...BXZ | 1970-01-01T00:00:00.000000Z - ... | 12.5 s, 144 samples
...BXN | 1970-01-01T00:00:00.000000Z - ... | 12.5 s, 144 samples
...BXE | 1970-01-01T00:00:00.000000Z - ... | 12.5 s, 144 samples
```

### GUI

The GUI is just an experimental feature and will likely crash for some reason. To try it, just execute the correct file

```bash
$ python axisem_db/experimental_gui/axisem_db_gui.py
```

![Imgur](http://i.imgur.com/FVmua2X.png)
