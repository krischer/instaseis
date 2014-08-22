# Python Interface to AxiSEM's netCDF Databases

## Installation

It requires the following Python modules.

* netCDF4
* ObsPy
* numpy

Install by checking out from git and installing with an editable installation.
The Makefile currently has to executed to build the shared library until a
proper installation routine has been written.

```bash
$ git clone ...
$ cd axisem_db
$ pip install -v - e .
$ cd axisem_db
$ make
```

## Testing

Testing right now is fairly minimal and machine dependent...

To test, make sure pytest is installed

```bash
pip install pytest
```

and execute

```bash
py.test
```

in the modules source code directory. The command will discover and execute all
defined tests.


## Usage

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

In [3]: receiver = Receiver(latitude=42.6390, longitude=74.4940, depth_in_m=0.0)

In [4]: source = Source(
   ...:     latitude=89.91, longitude=0.0, depth_in_m=12000,
   ...:     m_rr=4.710000e+24 / 1E7,
   ...:     m_tt=3.810000e+22 / 1E7,,
   ...:     m_pp=-4.740000e+24 / 1E7,
   ...:     m_rt=3.990000e+23 / 1E7,
   ...:     m_rp=-8.050000e+23 / 1E7,
   ...:     m_tp=-1.230000e+24 / 1E7)

In [5]: st = axisem_db.get_seismograms(source=source, receiver=receiver)

In [6]: print st
3 Trace(s) in Stream:
...BXZ | 1970-01-01T00:00:00.000000Z - 1970-01-01T00:29:41.746126Z | 12.5 s, 144 samples
...BXN | 1970-01-01T00:00:00.000000Z - 1970-01-01T00:29:41.746126Z | 12.5 s, 144 samples
...BXE | 1970-01-01T00:00:00.000000Z - 1970-01-01T00:29:41.746126Z | 12.5 s, 144 samples
```
