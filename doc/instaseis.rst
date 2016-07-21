======================
Main Instaseis Classes
======================

This page documents the central :func:`~instaseis.open_db` function and the
main Instaseis classes. Please always use the :func:`~instaseis.open_db`
function to open a connection to an Instaseis database.

.. contents::
    :local:

open_db() Function
------------------

.. autofunction:: instaseis.open_db

....

BaseInstaseisDB
---------------

.. autoclass:: instaseis.database_interfaces.base_instaseis_db.BaseInstaseisDB
    :members:

....

BaseNetCDFInstaseisDB
---------------------

.. autoclass:: instaseis.database_interfaces.base_netcdf_instaseis_db.BaseNetCDFInstaseisDB
    :members:

....

ReciprocalInstaseisDB
---------------------

.. autoclass:: instaseis.database_interfaces.reciprocal_instaseis_db.ReciprocalInstaseisDB
    :members:

....

ReciprocalMergedInstaseisDB
---------------------------

.. autoclass:: instaseis.database_interfaces.reciprocal_merged_instaseis_db.ReciprocalMergedInstaseisDB
    :members:

....

ForwardInstaseisDB
------------------

.. autoclass:: instaseis.database_interfaces.forward_instaseis_db.ForwardInstaseisDB
    :members:

....

ForwardMergedInstaseisDB
------------------------

.. autoclass:: instaseis.database_interfaces.forward_merged_instaseis_db.ForwardMergedInstaseisDB
    :members:

....

RemoteInstaseisDB
-----------------

.. autoclass:: instaseis.database_interfaces.remote_instaseis_db.RemoteInstaseisDB
    :members:

....

SyngineInstaseisDB
------------------

.. autoclass:: instaseis.database_interfaces.syngine_instaseis_db.SyngineInstaseisDB
    :members:
