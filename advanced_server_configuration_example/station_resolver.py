#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A very simple and fairly fast way to get coordinates of known seismic stations
based on an in-memory SQLite database including more complex wildcard based
queries.

The data source is the text output of any fdsnws-station service at the
'station' level. You can get one for example from IRIS with

$ wget \
  "http://service.iris.edu/fdsnws/station/1/query?level=station&format=text" \
  -O station_list.txt


Usage
-----

The first step is to create the database. Pass the filename of the previously
downloaded station list. It returns the database connection and an active
cursor.

>>> conn, cursor = parse_station_file("station_list.txt")

Now you can get the coordinates with

>>> coords = get_coordinates(cursor,
...                          networks=["I*", "T?", "GR"],
...                          stations=["A*"])

which will return a list of dictionaries:

>>> print(coords[0])
{'station': 'ATTU1', 'latitude': 52.882099, 'longitude': 173.164307,
 'network': 'IM'}

You can pass any number of network and station constraints (with and without
wildcards; you can also just not have any constraints ...) and it will return
all matching combinations.


The MIT License (MIT)

Copyright (c) 2020 Lion Krischer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import csv
import io
import sqlite3

from instaseis.helpers import elliptic_to_geocentric_latitude


def parse_station_file(filename):
    """
    Parse the station file to an in-memory sqlite database. This is super quick
    so its not necessary to store it on disc.

    Returns the connection and an active cursor.
    """
    conn = sqlite3.connect(":memory:")
    c = conn.cursor()

    # Create table
    c.execute(
        """
    CREATE TABLE coordinates (
        network text COLLATE NOCASE,
        station text COLLATE NOCASE,
        latitude real,
        longitude real
    );
    """
    )
    c.execute("CREATE INDEX network_index ON coordinates(network);")
    c.execute("CREATE INDEX station_index ON coordinates(station);")

    with io.open(filename, mode="rt", newline="") as csvfile:
        stationreader = csv.reader(csvfile, delimiter="|")
        # Skip header.
        next(stationreader, None)
        # Create a dictionary as a cheap way to ensure a unique network +
        # station combination. This also means that the last version of a
        # certain network + station combination will end up in the database
        # which is probably what we want.
        stations = {(i[0], i[1]): i[:4] for i in stationreader}
        # Put into database
        c.executemany(
            """
        INSERT INTO coordinates
            ('network', 'station', 'latitude', 'longitude')
            VALUES (?, ?, ?, ?);
        """,
            (i[:4] for i in stations.values()),
        )
        conn.commit()

    return conn, c


def get_coordinates(cursor, networks=(), stations=(), debug=False):
    # Build query
    query = "SELECT * FROM coordinates"

    network_queries = []
    for network in networks:
        if "*" in network or "?" in network:
            network_queries.append("network GLOB '%s'" % network)
        else:
            network_queries.append("network='%s'" % network)

    station_queries = []
    for station in stations:
        if "*" in station or "?" in station:
            station_queries.append("station GLOB '%s'" % station)
        else:
            station_queries.append("station='%s'" % station)

    if network_queries or station_queries:
        query += " WHERE"

    if network_queries:
        query = query + " ({network_queries})".format(
            network_queries=" OR ".join(network_queries)
        )

    if network_queries and station_queries:
        query += " AND"

    if station_queries:
        query = query + " ({station_queries})".format(
            station_queries=" OR ".join(station_queries)
        )

    if debug:
        print("Constructed query: %s" % query)

    # Convert to geocentric coordinates.
    return [
        {
            "network": i[0],
            "station": i[1],
            "latitude": elliptic_to_geocentric_latitude(i[2]),
            "longitude": i[3],
        }
        for i in cursor.execute(query).fetchall()
    ]


if __name__ == "__main__":
    # Simple example assuming a file `station_list.txt` exists in the current
    # directory.
    FILENAME = "station_list.txt"
    conn, cursor = parse_station_file(FILENAME)

    coords = get_coordinates(
        cursor, networks=["I*", "T?", "GR"], stations=["A*"], debug=True
    )
    conn.close()

    import pprint

    pprint.pprint(coords)
