GET /ttimes
^^^^^^^^^^^

.. note::

    Requires an advanced server setup. Please view the
    :doc:`../advanced_server_configuration` for details.

Description
    Get theoretical arrival times through a 1D Earth model if the server has been configured with that capability.

Content-Type
    application/json; charset=UTF-8

Example Response
    .. code-block:: json

        {
            "travel_time": 570.8677703798731
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
| ``phase``                | Float    | True     | The phase name. Depending on the travel time implementation this can be very flexible.                       |
+--------------------------+----------+----------+--------------------------------------------------------------------------------------------------------------+
