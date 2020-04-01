#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:copyright:
    Lion Krischer (lion.krischer@gmail.com), 2020
:license:
    GNU Lesser General Public License, Version 3 [non-commercial/academic use]
    (http://www.gnu.org/copyleft/lgpl.html)
"""
import collections

import tornado.web

from ..instaseis_request import InstaseisRequestHandler


class TravelTimeHandler(InstaseisRequestHandler):
    def get(self):
        if self.application.travel_time_callback is None:
            msg = "Server does not support travel time calculations."
            raise tornado.web.HTTPError(404, log_message=msg, reason=msg)

        required_parameters = (
            "sourcelatitude",
            "sourcelongitude",
            "sourcedepthinmeters",
            "receiverlatitude",
            "receiverlongitude",
            "receiverdepthinmeters",
            "phases",
        )

        missing_parameters = sorted(
            [
                _i
                for _i in required_parameters
                if _i not in self.request.arguments
            ]
        )
        if missing_parameters:
            msg = "The following required parameters are missing: %s" % (
                ", ".join("'%s'" % _i for _i in missing_parameters)
            )
            raise tornado.web.HTTPError(400, log_message=msg, reason=msg)

        # Preserve order.
        all_phases = collections.OrderedDict()

        for p in self.get_argument("phases").split(","):
            try:
                tt = self.application.travel_time_callback(
                    sourcelatitude=float(self.get_argument("sourcelatitude")),
                    sourcelongitude=float(
                        self.get_argument("sourcelongitude")
                    ),
                    sourcedepthinmeters=float(
                        self.get_argument("sourcedepthinmeters")
                    ),
                    receiverlatitude=float(
                        self.get_argument("receiverlatitude")
                    ),
                    receiverlongitude=float(
                        self.get_argument("receiverlongitude")
                    ),
                    receiverdepthinmeters=float(
                        self.get_argument("receiverdepthinmeters")
                    ),
                    phase_name=p,
                    db_info=self.application.db.info,
                )
                if tt:
                    all_phases[p] = tt
            except ValueError as e:
                err_msg = str(e)
                if err_msg.lower().startswith("invalid phase name"):
                    msg = "Invalid phase name '%s'." % p
                else:
                    msg = "Failed to calculate travel time due to: %s" % str(e)
                raise tornado.web.HTTPError(400, log_message=msg, reason=msg)

        if not all_phases:
            msg = "No ray for the given geometry and any of the phases found."
            raise tornado.web.HTTPError(404, log_message=msg, reason=msg)

        self.write({"travel_times": all_phases})
