GET /greens_function
^^^^^^^^^^^^^^^^^^^^

.. note::

    Some parts of this route require an advanced server setup. Please view the
    :doc:`../advanced_server_configuration` for details. The parts that don't
    will keep working even with a normal configuration.

Description
    Returns Instaseis Greens's function, that is seimograms for contraction with the
    decomposition of the Momenttensor as in Minson & Dreger (2008). Several convenience
    function of the ``/seismograms`` route are replicated here to make it suitable to be
    queried by any program able to use HTTP.

    .. list-table::

        * - | Minson, Sarah E., and Douglas S. Dreger (2008)
            | **Stable Inversions for Complete Moment Tensors.**
            | *Geophysical Journal International* 174 (2): 585–592.
            | http://dx.doi.org/10.1111/j.1365-246X.2008.03797.x

Content-Type
    * ``application/zip`` (if zipped SAC data is requested)
    * ``application/vnd.fdsn.mseed`` (if MiniSEED data is requested)

Filetype
    Returns a ZIP archive with SAC files or MiniSEED files encoded with
    encoding format 4 (IEEE floating point).

    SAC files will have the following user defined variables set:

    * ``KUSER0``: "InstSeis"
    * ``KUSER1``: The first eight letters of the Instaseis version used to generate the waveforms.
    * ``KUSER2``: The first eight letters of the velocity model name.
    * ``USER0``: The scale factor used to generate the waveforms.

+-----------------------------+----------+----------+-----------------------------+--------------------------------------------------------------------------------------+
| Parameter                   | Type     | Required | Default Value               | Description                                                                          |
+=============================+==========+==========+=============================+======================================================================================+
| **Source/Receiver Parameters**                                                                                                                                         |
+-----------------------------+----------+----------+-----------------------------+--------------------------------------------------------------------------------------+
| ``sourcedepthinmeters``     | Float    | True     |                             | The source depth in meters.                                                          |
+-----------------------------+----------+----------+-----------------------------+--------------------------------------------------------------------------------------+
| ``sourcedistanceindegrees`` | Float    | True     |                             | The epicentral disctance of source - receiver in degrees (computed on the surface    |
|                             |          |          |                             | of a sphere).                                                                        |
+-----------------------------+----------+----------+-----------------------------+--------------------------------------------------------------------------------------+
| **Output parameters**                                                                                                                                                  |
+-----------------------------+----------+----------+-----------------------------+--------------------------------------------------------------------------------------+
| ``format``                  | String   | False    | saczip                      | Specify output file to be either MiniSEED or a ZIP archive of SAC files, either      |
|                             |          |          |                             | ``miniseed`` or ``saczip``.                                                          |
+-----------------------------+----------+----------+-----------------------------+--------------------------------------------------------------------------------------+
| ``label``                   | String   | False    |                             | Specify a label to be included in file names and HTTP file name suggestions.         |
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
| ``origintime``              | Datetime | False    | 1970-01-01T00:00:00.000000Z | Specify the source origin time. This must be specified as an                         |
|                             |          |          |                             | absolute date and time. This time coincides with the peak of the                     |
|                             |          |          |                             | source time function.                                                                |
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
|                             |          |          |                             | seconds, from the origin time.                                                       |
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
|                             |          |          |                             | seconds, from the start time.                                                        |
+-----------------------------+----------+----------+-----------------------------+--------------------------------------------------------------------------------------+
