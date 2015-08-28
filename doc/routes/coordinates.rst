GET /coordinates
^^^^^^^^^^^^^^^^

.. note::

    Requires an advanced server setup. Please view the
    :doc:`../advanced_server_configuration` for details.

Description
    Station coordinates if the server has been configured to serve them.
    Returns a GeoJSON file to simplify further usage.

Content-Type
    application/vnd.geo+json

Example Response
    .. code-block:: json

        {
          "type": "FeatureCollection",
          "features": [
            {
              "geometry": {
                "coordinates": [
                  -106.4572,
                  34.765424215322156
                ],
                "type": "Point"
              },
              "type": "Feature",
              "properties": {
                "station_code": "ANMO",
                "network_code": "IU"
              }
            },
            {
              "geometry": {
                "coordinates": [
                  32.7934,
                  39.678769312230145
                ],
                "type": "Point"
              },
              "type": "Feature",
              "properties": {
                "station_code": "ANTO",
                "network_code": "IU"
              }
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
