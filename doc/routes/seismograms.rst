GET /seismograms
^^^^^^^^^^^^^^^^

.. note::

    Some parts of this route require an advanced server setup. Please view the
    :doc:`../advanced_server_configuration` for details. The parts that don't
    will keep working even with a normal configuration.

Description
    Returns Instaseis seismograms. In addition to the functionality of the
    ``/seismograms_raw`` route this one also offers the possibility to convert
    every seismogram to displacement, velocity, or acceleration. It furthermore
    can shift the start time of the data to the peak of the source time
    function's sliprate and resample the data to an arbitrary (up to 100 Hz)
    sampling rate. This route is suitable to be queried by any program able
    to use HTTP.

Content-Type
    application/octet-stream

    application/zip

Filetype
    Returns MiniSEED files encoded with encoding format 4 (IEEE floating
    point) or a ZIP archive with SAC files.

+-----------------------------+----------+----------+-----------------------------+--------------------------------------------------------------------------------------+
| Parameter                   | Type     | Required | Default Value               | Description                                                                          |
+=============================+==========+==========+=============================+======================================================================================+
| **Output parameters**                                                                                                                                                  |
+-----------------------------+----------+----------+-----------------------------+--------------------------------------------------------------------------------------+
| ``format``                  | String   | False    | saczip                      | Specify output file to be either MiniSEED or a ZIP archive of SAC files, either      |
|                             |          |          |                             | ``miniseed`` or ``saczip``.                                                          |
+-----------------------------+----------+----------+-----------------------------+--------------------------------------------------------------------------------------+
| ``label``                   | String   | False    |                             |  Specify a label to be included in file names and HTTP file name suggestions.        |
+-----------------------------+----------+----------+-----------------------------+--------------------------------------------------------------------------------------+
| ``components``              | String   | False    | ZNE                         | Specify the orientation of the synthetic seismograms as a list of any combination of |
|                             |          |          |                             | ``Z`` (vertical), ``N`` (north), ``E`` (east), ``R`` (radial), ``T`` (transverse).   |
+-----------------------------+----------+----------+-----------------------------+--------------------------------------------------------------------------------------+
| ``units``                   | String   | False    | displacement                | Specify either ``displacement``, ``velocity`` or ``acceleration`` for the synthetics.|
+-----------------------------+----------+----------+-----------------------------+--------------------------------------------------------------------------------------+
| ``dt``                      | Float    | False    |                             | If given, seismograms will be resampled to the desired sample spacing.               |
+-----------------------------+----------+----------+-----------------------------+--------------------------------------------------------------------------------------+
| ``alanczos``                | Integer  | False    | 12                          | Specify the width of the Lanczos kernel used for resampling to requested sample      |
|                             |          |          |                             | interval.                                                                            |
+-----------------------------+----------+----------+-----------------------------+--------------------------------------------------------------------------------------+
| **Time Parameters**                                                                                                                                                    |
+-----------------------------+----------+----------+-----------------------------+--------------------------------------------------------------------------------------+
| ``origintime``              | Datetime | False    | 1970-01-01T00:00:00.000000Z | Specify the source origin time. This must be specified as an                         |
|                             |          |          |                             | absolute date and time. This time coincides with the peak of the                     |
|                             |          |          |                             | source time function.                                                                |
+-----------------------------+----------+----------+-----------------------------+--------------------------------------------------------------------------------------+
| ``starttime``               | Datetime | False    |                             | Start time of the returned seismograms. Might imply zero padding at front.           |
+-----------------------------+----------+----------+-----------------------------+--------------------------------------------------------------------------------------+
| ``endtime``                 | Datetime | False    |                             | End time of the returned seismogram.                                                 |
+-----------------------------+----------+----------+-----------------------------+--------------------------------------------------------------------------------------+
| ``duration``                | Float    | False    |                             | Duration of the returned seismograms in seconds relative to the start time.          |
+-----------------------------+----------+----------+-----------------------------+--------------------------------------------------------------------------------------+
| **Receiver Parameters**                                                                                                                                                |
+-----------------------------+----------+----------+-----------------------------+--------------------------------------------------------------------------------------+
| Directly specify coordinates and network/station codes ...                                                                                                             |
+-----------------------------+----------+----------+-----------------------------+--------------------------------------------------------------------------------------+
| ``receiverlatitude``        | Float    | True     |                             | The geocentric latitude of the receiver.                                             |
+-----------------------------+----------+----------+-----------------------------+--------------------------------------------------------------------------------------+
| ``receiverlongitude``       | Float    | True     |                             | The longitude of the receiver.                                                       |
+-----------------------------+----------+----------+-----------------------------+--------------------------------------------------------------------------------------+
| ``receiverdepthinmeters``   | Float    | False    | 0.0                         | The depth of the receiver in meter.                                                  |
+-----------------------------+----------+----------+-----------------------------+--------------------------------------------------------------------------------------+
| ``networkcode``             | String   | False    | XX                          | Specify the network code of the final seismograms. Maximum of two letters.           |
+-----------------------------+----------+----------+-----------------------------+--------------------------------------------------------------------------------------+
| ``stationcode``             | String   | False    | SYN                         | Specify the station code of the final seismograms. Maximum of five letters.          |
+-----------------------------+----------+----------+-----------------------------+--------------------------------------------------------------------------------------+
| ... or use wildcard searches over network and station codes. Potentially returns multiple stations.                                                                    |
+-----------------------------+----------+----------+-----------------------------+--------------------------------------------------------------------------------------+
| ``network``                 | String   | False    |                             | Wildcarded network codes, e.g. ``I*,B?,AU``.                                         |
+-----------------------------+----------+----------+-----------------------------+--------------------------------------------------------------------------------------+
| ``station``                 | String   | False    |                             | Wildcarded station codes, e.g. ``A*,ANMO``.                                          |
+-----------------------------+----------+----------+-----------------------------+--------------------------------------------------------------------------------------+
| **Receiver Parameters**                                                                                                                                                |
+-----------------------------+----------+----------+-----------------------------+--------------------------------------------------------------------------------------+
| The source can by set by specifying an event id if the server has been set-up for this ...                                                                             |
+-----------------------------+----------+----------+-----------------------------+--------------------------------------------------------------------------------------+
| ``event_id``                | String   | False    |                             | The id of the event to use.                                                          |
+-----------------------------+----------+----------+-----------------------------+--------------------------------------------------------------------------------------+
| ... or by specify the source parameters in a variety of ways.                                                                                                          |
+-----------------------------+----------+----------+-----------------------------+--------------------------------------------------------------------------------------+
| ``sourcelatitude``          | Float    | True     |                             | The geocentric latitude of the source.                                               |
+-----------------------------+----------+----------+-----------------------------+--------------------------------------------------------------------------------------+
| ``sourcelongitude``         | Float    | True     |                             | The longitude of the source.                                                         |
+-----------------------------+----------+----------+-----------------------------+--------------------------------------------------------------------------------------+
| ``sourcedepthinmeters``     | Float    | False    | 0.0                         | The depth of the source in meter.                                                    |
+-----------------------------+----------+----------+-----------------------------+--------------------------------------------------------------------------------------+
| **Source mechanism as a moment tensor**                                                                                                                                |
+-----------------------------+----------+----------+-----------------------------+--------------------------------------------------------------------------------------+
| ``sourcemomenttensor``      | List     | False    |                             | Specify a source in moment tensor components as a list: ``Mrr,Mtt,Mpp,Mrt,Mrp,Mtp``  |
|                             |          |          |                             | with values in Newton meters (Nm).                                                   |
|                             |          |          |                             |                                                                                      |
|                             |          |          |                             | Example: ``1.04e22,-0.039e22,-1e22,0.304e22,-1.52e22,-0.119e22``                     |
+-----------------------------+----------+----------+-----------------------------+--------------------------------------------------------------------------------------+
| **Source mechanism as a double couple**                                                                                                                                |
+-----------------------------+----------+----------+-----------------------------+--------------------------------------------------------------------------------------+
| ``sourcedoublecouple``      | List     | False    |                             | Specify a source as a double couple. The list of values are ``strike,dip,rake[,M0]``,|
|                             |          |          |                             | where strike, dip and rake are in degrees and M0 is the scalar seismic moment in     |
|                             |          |          |                             | Newton meters (Nm). If not specified, a value of *1e19* will be used as the scalar   |
|                             |          |          |                             | moment.                                                                              |
|                             |          |          |                             |                                                                                      |
|                             |          |          |                             | example: ``19,18,116,1e19``                                                          |
+-----------------------------+----------+----------+-----------------------------+--------------------------------------------------------------------------------------+
| **Source mechanism as forces**                                                                                                                                         |
+-----------------------------+----------+----------+-----------------------------+--------------------------------------------------------------------------------------+
| ``sourceforce``             | List     | False    |                             | Specify a force source as a list of ``Fr,Ft,Fp`` in units of Newtons (N).            |
|                             |          |          |                             |                                                                                      |
|                             |          |          |                             | example: ``1e22,1e22,1e22``                                                          |
+-----------------------------+----------+----------+-----------------------------+--------------------------------------------------------------------------------------+
