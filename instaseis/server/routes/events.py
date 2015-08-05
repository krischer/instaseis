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


class EventHandler(InstaseisRequestHandler):
    def get(self):
        if self.application.event_info_callback is None:
            msg = "Server does not support event information."
            raise tornado.web.HTTPError(
                404, log_message=msg, reason=msg)

        if "id" not in self.request.arguments:
            msg = "'id' parameter is required."
            raise tornado.web.HTTPError(
                400, log_message=msg, reason=msg)

        event_id = self.get_argument("id")

        try:
            event = self.application.event_info_callback(event_id)
        except ValueError:
            msg = "Event not found."
            raise tornado.web.HTTPError(404, log_message=msg, reason=msg)

        # Convert time to a string.
        event["origin_time"] = str(event["origin_time"])

        self.write(event)
