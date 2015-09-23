Advanced Server Configuration
=============================

Certain advanced functionality requires a more extensive server configuration.
This is a bit more work but you can tweak that functionality to best suite your
own needs.

You can find a complete example implementation `here
<https://github.com/krischer/instaseis/tree/master/advanced_server_configuration_example>`_.

It works by implementing up to three callback functions to:

* resolve station coordinates from network and station codes
* resolve event parameters from event ids
* calculate theoretical travel times

You can choose to implement any number of these - appropriate error messages
will be raised if the server requires a callback function this is not
available. The rest will keep on working just fine.


.. note::

    Please raise a ``ValueError`` within in the callback functions if something
    does not go as expected. That will be caught by Instaseis; other exceptions
    will result in a return code 500 (internal server error).


.. contents:: Contents of this Page
    :local:


Station Coordinates Callback
----------------------------

This callback function is used for the ``/coordinates`` route, as well as
``network`` and ``station`` based queries in the ``/seismograms`` route. It
resolves network and station code searches to actual coordinates. How the
coordinates are resolved is up to the users but make sure it is fast. Also make
sure the coordinates are geocentric. Instaseis has a helper function to aid
with that: :func:`instaseis.helpers.elliptic_to_geocentric_latitude`

**Affected routes:**

* :doc:`routes/coordinates`
* :doc:`routes/seismograms`


**Function Signature:**

.. code-block:: python

    def get_station_coordinates(networks, stations):
        ...
        return [...]

**Parameters:**

* ``networks``: [*list of strings*], list of case-insensitive network ids. Wildcards ``*`` and ``?`` are allowed.

* ``stations``: [*list of strings*], list of case-insensitive station ids. Wildcards ``*`` and ``?`` are allowed.

**Return Values:**

* A list with one dictionary per found station if successful. Each dictionary
  has the following keys: ``"network"``, ``"station"``, ``"latitude"``,
  ``"longitude"``.
* An empty list if no station matching the query has been found.
* Raise a ``ValueError`` for other cases.

**Examples:**

.. code-block:: python

    >>> get_station_coordinates(networks=["IU"], stations["ANMO"])
    [{"latitude": 34.94591,
       "longitude": -106.4572,
       "network": "IU",
       "station": "ANMO"}]

    >>> get_station_coordinates(networks=["IU", "B*"], stations["ANT*", "ANM?"])
    [{"latitude": 39.868,
      "longitude": 32.7934,
      "network": "IU",
      "station": "ANTO"
     }, {
      "latitude": 34.94591,
      "longitude": -106.4572,
      "network": "IU",
      "station": "ANMO"}]

    >>> get_station_coordinates(networks=["AA", "BB"], stations["CC", "DD"])
    []


Event Parameters Callback
-------------------------

This callback function has a single purpose: resolve an event identifier to
event parameters. Users could choose to call an external web service within
that function or query a local database. It is used for the ``/event`` route as
well as event identifier based queries in the ``/seismograms`` route. Make sure
the coordinates are geocentric. Instaseis has a helper function to aid with
that: :func:`instaseis.helpers.elliptic_to_geocentric_latitude`

**Affected routes:**

* :doc:`routes/event`
* :doc:`routes/seismograms`

**Function Signature:**

.. code-block:: python

    def get_event(event_id):
        ...
        return {...}

**Parameters:**

* ``event_id``: [*str*], event identifier


**Return Values:**

* A dictionary with the event parameters and the following keys: ``"m_rr"``,
  ``"m_tt"``, ``"m_pp"``, ``"m_rt"``, ``"m_rp"``, ``"m_tp"``, ``"latitude"``,
  ``"longitude"``, ``"depth_in_m"``, ``"origin_time"``. The tensor components
  have to be in *Nm*.
* A ``ValueError`` will be always be interpreted as a not found event.

**Examples:**

.. code-block:: python

    >>> get_event("B071791B")
    {"m_rr": -58000000000000000,
     "m_tt": 78100000000000000,
     "m_pp": -20100000000000000,
     "m_rt": -56500000000000000,
     "m_rp": 108100000000000000,
     "m_tp": 315300000000000000,
     "latitude": -3.8,
     "longitude": -104.21,
     "depth_in_m": 0,
     "origin_time": "1991-07-17T16:41:33.100000Z"}

    >>> get_event("random_things")
    ValueError: Event not found.


Travel Time Callback
--------------------

This callback function is used for the ``/ttimes`` route and for the phase
relative start and end times in the ``/seismograms`` route. It receives source
and receiver coordinates as well as a phase name and is supposed to return the
travel time from source to receiver for that particular phase in seconds. The
coordinates can be assumed to be geocentric and the calculations should happen
in a spherical planet. Make sure to perform the calculations in the same model
that has been used to calculate the databases.

**Affected routes:**

* :doc:`routes/ttimes`
* :doc:`routes/seismograms`


**Function Signature:**

.. code-block:: python

    def get_travel_time(sourcelatitude, sourcelongitude, sourcedepthinmeters,
                        receiverlatitude, receiverlongitude,
                        receiverdepthinmeters, phase_name):
        ...
        return ttime

**Parameters:**

* ``sourcelatitude``: [*float*], geocentric source latitude

* ``sourcelongitude``: [*float*], source longitude

* ``sourcedepthinmeters``: [*float*], source depth in meters

* ``receiverlatitude``: [*float*], geocentric receiver latitude

* ``receiverlongitude``: [*float*], receiver longitude

* ``receiverdepthinmeters``: [*float*], receiver depth in meters

* ``phase_name``: [*str*], case-sensitive phase name


**Return Values:**

* Travel time in seconds if successful.
* ``None`` if phase has no arrival for the given source-receiver geometry.
* Raise a ``ValueError`` for other cases, e.g. unknown phase name, invalid source-receiver geometry, ...

**Examples:**

.. code-block:: python

    >>> get_travel_time(0.0, 50.0, 300000, 0.0, 0.0, 0.0, "P")
    504.357

    >>> get_travel_time(0.0, 50.0, 300000, 0.0, 0.0, 0.0, "Pdiff")
    None

    >>> get_travel_time(0.0, 50.0, 300000, 0.0, 0.0, 0.0, "bogus")
    ValueError: Invalid phase name.


Hooking it up to the Instaseis Server
-------------------------------------

Best have a look at the full example implementation `here
<https://github.com/krischer/instaseis/tree/master/advanced_server_configuration_example>`_.

You will have to create a new file and pass the three callback functions to the
``launch_io_loop()`` function. The following code snippet will give you a
similar command line interface to the default Instaseis server:


.. code-block:: python

    from __future__ import (absolute_import, division, print_function,
                            unicode_literals)
    import argparse
    import os

    from instaseis.server.app import launch_io_loop

    if __name__ == "__main__":
        parser = argparse.ArgumentParser(
            prog="python -m instaseis.server",
            description='Launch an Instaseis server offering seismograms with a '
                        'REST API.')
        parser.add_argument('--port', type=int, required=True,
                            help='Server port.')
        parser.add_argument('--buffer_size_in_mb', type=int,
                            default=0, help='Size of the buffer in MB')
        parser.add_argument('db_path', type=str,
                            help='Database path')
        parser.add_argument(
            '--quiet', action='store_true',
            help="Don't print any output. Overwrites the 'log_level` setting.")
        parser.add_argument(
            '--log-level', type=str, default='INFO',
            choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'],
            help='The log level for all Tornado loggers.')

        args = parser.parse_args()
        db_path = os.path.abspath(args.db_path)

        launch_io_loop(db_path=db_path, port=args.port,
                       buffer_size_in_mb=args.buffer_size_in_mb,
                       quiet=args.quiet, log_level=args.log_level,
                       station_coordinates_callback=get_station_coordinates,
                       event_info_callback=get_event,
                       travel_time_callback=get_travel_time)
