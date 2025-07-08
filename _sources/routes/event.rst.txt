GET /event
^^^^^^^^^^

.. note::

    Requires an advanced server setup. Please view the
    :doc:`../advanced_server_configuration` for details.

Description
    Event information if the server has been configured to serve it.

Content-Type
    application/json; charset=UTF-8

Example Response
    .. code-block:: json

        {
          "m_pp": -20100000000000000,
          "m_rt": -56500000000000000,
          "m_rp": 108100000000000000,
          "longitude": -104.21,
          "m_rr": -58000000000000000,
          "m_tp": 315300000000000000,
          "latitude": -3.8,
          "m_tt": 78100000000000000,
          "origin_time": "1991-07-17T16:41:33.100000Z"
        }



+-------------------------+----------+----------+-----------------------------+----------------------------------------------------------------------+
| Parameter               | Type     | Required | Default Value               | Description                                                          |
+=========================+==========+==========+=============================+======================================================================+
| ``id``                  | String   | True     |                             | The event id.                                                        |
+-------------------------+----------+----------+-----------------------------+----------------------------------------------------------------------+
