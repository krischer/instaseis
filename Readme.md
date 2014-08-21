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
