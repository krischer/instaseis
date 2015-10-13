POST /finite_source
^^^^^^^^^^^^^^^^^^^

.. note::

    Some parts of this route require an advanced server setup. Please view the
    :doc:`../advanced_server_configuration` for details. The parts that don't
    will keep working even with a normal configuration.

    Per default this route allows at most 1000 point sources for a single
    finite source. This limit can be changed when starting the server.

Description
    Returns seismograms for finite sources defined in the USGS param file
    format. Just POST the file and set additional parameters in the URL.

What is Actually Happening
     Some of the calculations performed in this endpoint are not immediatly
     obvious and some choices have to be made so we'll elaborate a bit here.

     1. The sliprate of each point source is defined as an asymmetric cosine
        with a certain rupture, rise, and fall time. We sample it at 10 Hz for
        a thousand seconds - this limits the maximum rise and fall times. Rise
        and fall times smaller than one second will be set to one second to
        make sure it can be accurately sampled.
     2. Each sampled sliprate is zero padded with a number of samples at the
        beginning and the end (the additional time shift is later accounted
        for). This is done to avoid running into boundary issues with the
        following filter.
     3. A fourth order Butterworth filter is applied twice (forwards and
        backwards) resulting in a zero phase filter. The corner frequency is
        the dominant frequency of the database. This makes sure we don't
        introduce frequencies in the convolution that we cannot propagate in
        the numerical simulation.
     4. The seismograms for all point sources are calculated, time-shifted,
        convolved, and stacked.

Content-Type
    * ``application/zip`` (if zipped SAC data is requested)
    * ``application/octet-stream`` (if MiniSEED data is requested)

Filetype
    Returns a ZIP archive with SAC files or MiniSEED files encoded with
    encoding format 4 (IEEE floating point).

+-----------------------------+----------+----------+-----------------------------+--------------------------------------------------------------------------------------+
| Parameter                   | Type     | Required | Default Value               | Description                                                                          |
+=============================+==========+==========+=============================+======================================================================================+
| **Output parameters**                                                                                                                                                  |
+-----------------------------+----------+----------+-----------------------------+--------------------------------------------------------------------------------------+
| ``format``                  | String   | False    | saczip                      | Specify output file to be either MiniSEED or a ZIP archive of SAC files, either      |
|                             |          |          |                             | ``miniseed`` or ``saczip``.                                                          |
+-----------------------------+----------+----------+-----------------------------+--------------------------------------------------------------------------------------+
| ``label``                   | String   | False    |                             | Specify a label to be included in file names and HTTP file name suggestions.         |
+-----------------------------+----------+----------+-----------------------------+--------------------------------------------------------------------------------------+
| ``components``              | String   | False    | ZNE                         | Specify the orientation of the synthetic seismograms as a list of any combination of |
|                             |          |          |                             | ``Z`` (vertical), ``N`` (north), ``E`` (east), ``R`` (radial), ``T`` (transverse).   |
+-----------------------------+----------+----------+-----------------------------+--------------------------------------------------------------------------------------+
| ``units``                   | String   | False    | displacement                | Specify either ``displacement``, ``velocity`` or ``acceleration`` for the synthetics.|
+-----------------------------+----------+----------+-----------------------------+--------------------------------------------------------------------------------------+
| ``dt``                      | Float    | False    |                             | If given, seismograms will be resampled to the desired sample spacing.               |
+-----------------------------+----------+----------+-----------------------------+--------------------------------------------------------------------------------------+
| ``kernelwidth``             | Integer  | False    | 12                          | Specify the width of the sinc kernel used for resampling to requested sample         |
|                             |          |          |                             | interval in terms of the original sampling interval.                                 |
+-----------------------------+----------+----------+-----------------------------+--------------------------------------------------------------------------------------+
| **Time Parameters**                                                                                                                                                    |
+-----------------------------+----------+----------+-----------------------------+--------------------------------------------------------------------------------------+
| ``origintime``              | Datetime | False    | 1900-01-01T00:00:00.000000Z | Specify the source origin time. This must be specified as an                         |
|                             |          |          |                             | absolute date and time. **This time conincides will be the onset time of the**       |
|                             |          |          |                             | **first slipping point source.** USGS param files don't have absolute time values.   |
+-----------------------------+----------+----------+-----------------------------+--------------------------------------------------------------------------------------+
| ``starttime``               | Datetime | False    |                             | Specifies the desired start time for the synthetic trace(s). This may be specified   |
|                             |          |          |                             | as either:                                                                           |
|                             |          |          |                             |                                                                                      |
|                             |          |          |                             | * an absolute date and time                                                          |
|                             |          |          |                             | * a phase-relative offset                                                            |
|                             |          |          |                             | * an offset from origin time in seconds                                              |
|                             |          |          |                             |                                                                                      |
|                             |          |          |                             | If the value is recognized as a date and time, it is interpreted as an absolute time.|
|                             |          |          |                             | If the value is in the form ``phase[+-]offset`` it is interpreted as a               |
|                             |          |          |                             | phase-relative time, for example ``P-10`` (meaning P-wave arrival time minus 10      |
|                             |          |          |                             | seconds). If the value is a numerical value it is interpreted as an offset, in       |
|                             |          |          |                             | seconds, from the origin time. Arrival times are always relative to the origin time. |
+-----------------------------+----------+----------+-----------------------------+--------------------------------------------------------------------------------------+
| ``endtime``                 | Datetime | False    |                             | Specifies the desired end time for the synthetic trace(s). This may be specified     |
|                             |          |          |                             | as either:                                                                           |
|                             |          |          |                             |                                                                                      |
|                             |          |          |                             | * an absolute date and time                                                          |
|                             |          |          |                             | * a phase-relative offset                                                            |
|                             |          |          |                             | * an offset (duration) from start time in seconds                                    |
|                             |          |          |                             |                                                                                      |
|                             |          |          |                             | If the value is recognized as a date and time, it is interpreted as an absolute time.|
|                             |          |          |                             | If the value is in the form ``phase[+-]offset`` it is interpreted as a               |
|                             |          |          |                             | phase-relative time, for example ``P-10`` (meaning P-wave arrival time minus 10      |
|                             |          |          |                             | seconds). If the value is a numerical value it is interpreted as an offset, in       |
|                             |          |          |                             | seconds, from the start time. Arrival times are always relative to the origin time.  |
+-----------------------------+----------+----------+-----------------------------+--------------------------------------------------------------------------------------+
| **Receiver Parameters**                                                                                                                                                |
+-----------------------------+----------+----------+-----------------------------+--------------------------------------------------------------------------------------+
| Directly specify coordinates and network/station codes ...                                                                                                             |
+-----------------------------+----------+----------+-----------------------------+--------------------------------------------------------------------------------------+
| ``receiverlatitude``        | Float    | True     |                             | The geocentric latitude of the receiver.                                             |
+-----------------------------+----------+----------+-----------------------------+--------------------------------------------------------------------------------------+
| ``receiverlongitude``       | Float    | True     |                             | The longitude of the receiver.                                                       |
+-----------------------------+----------+----------+-----------------------------+--------------------------------------------------------------------------------------+
| ``networkcode``             | String   | False    | XX                          | Specify the network code of the final seismograms. Maximum of two letters.           |
+-----------------------------+----------+----------+-----------------------------+--------------------------------------------------------------------------------------+
| ``stationcode``             | String   | False    | SYN                         | Specify the station code of the final seismograms. Maximum of five letters.          |
+-----------------------------+----------+----------+-----------------------------+--------------------------------------------------------------------------------------+
| ``locationcode``            | String   | False    | SE                          | Specify the location code of the final seismograms. Maximum of two letters.          |
+-----------------------------+----------+----------+-----------------------------+--------------------------------------------------------------------------------------+
| ... or use wildcard searches over network and station codes. Potentially returns multiple stations.                                                                    |
+-----------------------------+----------+----------+-----------------------------+--------------------------------------------------------------------------------------+
| ``network``                 | String   | False    |                             | Wildcarded network codes, e.g. ``I*,B?,AU``.                                         |
+-----------------------------+----------+----------+-----------------------------+--------------------------------------------------------------------------------------+
| ``station``                 | String   | False    |                             | Wildcarded station codes, e.g. ``A*,ANMO``.                                          |
+-----------------------------+----------+----------+-----------------------------+--------------------------------------------------------------------------------------+
