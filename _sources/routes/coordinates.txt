GET /coordinates
^^^^^^^^^^^^^^^^

.. note::

    Requires an advanced server setup. Please view the
    :doc:`../advanced_server_configuration` for details.

Description
    Station coordinates if the server has been configured to serve them.

Content-Type
    application/json; charset=UTF-8

Example Response
    .. code-block:: json

        {
            "count": 2,
            "stations": [
                {
                    "latitude": 39.868,
                    "longitude": 32.7934,
                    "network": "IU",
                    "station": "ANTO"
                },
                {
                    "latitude": 34.94591,
                    "longitude": -106.4572,
                    "network": "IU",
                    "station": "ANMO"
                }
            ]
        }

+-------------------------+----------+----------+-----------------------------+----------------------------------------------------------------------+
| Parameter               | Type     | Required | Default Value               | Description                                                          |
+=========================+==========+==========+=============================+======================================================================+
| ``network``             | String   | True     |                             | Wildcarded network codes, e.g. ``I*,B?,AU``.                         |
+-------------------------+----------+----------+-----------------------------+----------------------------------------------------------------------+
| ``station``             | String   | True     |                             | Wildcarded station codes, e.g. ``A*,ANMO``.                          |
+-------------------------+----------+----------+-----------------------------+----------------------------------------------------------------------+
