#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2015
:license:
    GNU Lesser General Public License, Version 3 [non-commercial/academic use]
    (http://www.gnu.org/copyleft/lgpl.html)
"""
import tornado.web

from ..instaseis_request import InstaseisRequestHandler


class CoordinatesHandler(InstaseisRequestHandler):
    def get(self):
        if self.application.station_coordinates_callback is None:
            msg = "Server does not support station coordinates."
            raise tornado.web.HTTPError(
                404, log_message=msg, reason=msg)

        networks = self.get_argument("network", [])
        stations = self.get_argument("station", [])

        # Manually raise to get prettier errors.
        if not networks or not stations:
            msg = "Parameters 'network' and 'station' must be given."
            raise tornado.web.HTTPError(
                400, log_message=msg, reason=msg)

        networks = networks.split(",")
        stations = stations.split(",")

        coordinates = self.application.station_coordinates_callback(
            networks=networks, stations=stations)

        if not coordinates:
            msg = "No coordinates found satisfying the query."
            raise tornado.web.HTTPError(
                404, log_message=msg, reason=msg)

        features = []
        for station in coordinates:
            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [station["longitude"],
                                        station["latitude"]]
                    },
                    "properties": {
                        "network_code": station["network"],
                        "station_code": station["station"]
                    }
                }
            )

        geojson = {"type": "FeatureCollection", "features": features}
        self.write(geojson)
        self.set_header("Content-Type", "application/vnd.geo+json")
