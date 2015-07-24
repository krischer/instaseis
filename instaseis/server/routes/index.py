import tornado.web

from ... import __version__


class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        response = {
            "type": "Instaseis Remote Server",
            "version": __version__
        }
        self.write(response)
        self.set_header("Access-Control-Allow-Origin", "*")