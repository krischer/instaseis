GET /ttimes
^^^^^^^^^^^

.. note::

    Requires an advanced server setup. Please view the
    :doc:`../advanced_server_configuration` for details.

Description
    Get theoretical arrival times through a 1D Earth model if the server has been configured with that capability. For each phase name
    it will only return a single time - multiple arrivals of the same phase are thus not visible. The exact returned arrival depends
    on the used callback function - a sensible choice is the first arriving one.

Content-Type
    application/json; charset=UTF-8

Example Response
    .. code-block:: json

        {
            "travel_times": {
                "P": 504.357,
                "PP": 622.559,
                "sPKiKP": 1090.081
            }
        }

+--------------------------+----------+----------+--------------------------------------------------------------------------------------------------------------+
| Parameter                | Type     | Required | Description                                                                                                  |
+==========================+==========+==========+==============================================================================================================+
| ``sourcelatitude``       | Float    | True     | Geocentric latitude of the source.                                                                           |
+--------------------------+----------+----------+--------------------------------------------------------------------------------------------------------------+
| ``sourcelongitude``      | Float    | True     | Longitude of the source.                                                                                     |
+--------------------------+----------+----------+--------------------------------------------------------------------------------------------------------------+
| ``sourcedepthinmeters``  | Float    | True     | Depth of the source in meters.                                                                               |
+--------------------------+----------+----------+--------------------------------------------------------------------------------------------------------------+
| ``receiverlatitude``     | Float    | True     | Geocentric latitude of the receiver.                                                                         |
+--------------------------+----------+----------+--------------------------------------------------------------------------------------------------------------+
| ``receiverlongitude``    | Float    | True     | Longitude of the receiver.                                                                                   |
+--------------------------+----------+----------+--------------------------------------------------------------------------------------------------------------+
| ``receiverdepthinmeters``| Float    | True     | Depth of the receiver in meters. Many implementations will raise an error if this is not zero as they cannot |
|                          |          |          | deal with buried receivers.                                                                                  |
+--------------------------+----------+----------+--------------------------------------------------------------------------------------------------------------+
| ``phases``               | Float    | True     | Comma separated phase names. Depending on the travel time implementation this can be very flexible.          |
+--------------------------+----------+----------+--------------------------------------------------------------------------------------------------------------+
