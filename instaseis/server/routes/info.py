import copy

import tornado.web


class InfoHandler(tornado.web.RequestHandler):
    def get(self):
        info = copy.deepcopy(self.application.db.info)
        # No need to write a custom encoder...
        info["datetime"] = str(info["datetime"])
        info["slip"] = list([float(_i) for _i in info["slip"]])
        info["sliprate"] = list([float(_i) for _i in info["sliprate"]])
        # Clear the directory to avoid leaking any more system information then
        # necessary.
        info["directory"] = ""
        self.write(dict(info))
        self.set_header("Access-Control-Allow-Origin", "*")