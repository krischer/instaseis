import tornado.web


class CoordinatesHandler(tornado.web.RequestHandler):
    def get(self):
        if self.application.station_coordinates_callback is None:
            msg = "Server does not support station coordinates."
            raise tornado.web.HTTPError(
                404, log_message=msg, reason=msg)

        networks = self.get_argument("network")
        stations = self.get_argument("station")

        if not networks or not stations:
            msg = "Parameters 'network' and 'station' must be given."
            raise tornado.web.HTTPError(
                404, log_message=msg, reason=msg)

        networks = networks.split(",")
        stations = stations.split(",")

        coordinates = self.application.station_coordinates_callback(
            networks=networks, stations=stations)

        if not coordinates:
            msg = "No coordinates found satisfying the query."
            raise tornado.web.HTTPError(
                404, log_message=msg, reason=msg)

        self.write({"count": len(coordinates), "stations": coordinates})
        self.set_header("Access-Control-Allow-Origin", "*")