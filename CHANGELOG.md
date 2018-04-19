# Changelog

## [1.3.0] - 2018-04-19
* Updated to support the latest versions of `ObsPy` and `tornado`.
* Now uses the NMSOP recommended definition of the moment magnitude.
  (see #58)
* Stable finite element mappings also for local/regional scale databases.
  (see #61)
* Source time functions for force sources (see #49).
* Some other small distribution fixes and improved error messages.

## [1.2.0] - 2017-08-07
* Closing sockets after each test case ran (see #48).
* GUI can now display map backgrounds for many planets/moons in the solar
  system. (see #54).
* Forcing installation of `geographiclib` (see #57).
* Setting `CMPINC` and `CMPAZ` headers in SAC files for the server routes.

## [1.1.1] - 2017-04-25
* Compatibility with `numpy` 1.12

## [1.1.0] - 2016-11-07
- `/ttimes` route can now return multiple phases
- `/greensfunction` route now has proper time handling
- net/sta/loc changes for the `/greensfunction` route.

## [1.0.1] - 2016-09-05
* Fixing a sporadically failing test.

## [1.0.0] - 2016-08-22
- Stable and production ready.
