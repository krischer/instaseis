=====================
Sources and Receivers
=====================

Instaseis uses objects to represent seismic receivers and different types of
sources. :class:`~instaseis.source.Source` objects are standard moment tensor
or double couple sources, :class:`~instaseis.source.ForceSource` object are
(as the name implies) force sources, and finite source are represented by the
:class:`~instaseis.source.FiniteSource` class.

.. note::

    Instaseis works with geocentric coordinates. Coordinates in for example
    QuakeML files are defined on the WGS84 ellipsoid. Thus they need to be
    converted. Instaseis performs this conversion if it is used to read from
    any file format - if you create the source objects yourself you have to
    take care to pass geocentric coordinates. The difference in both can be up
    to 20 kilometers. Depending on the application, **this is not an effect you
    can savely ignore**.

    The ruleset within Instaseis is simple:

    1. Source and receiver objects are always in geocentric coordinates.
    2. If parsed from any outside file (QuakeML/SEED/SAC/StationXML/...) the
       coordinates are assumed to be WGS84 and will be converted so the
       source/receiver objects are again in geocentric coordinates.

    The directions of r, theta, and phi are defined according to the standard
    `spherical coordinate system definition <https://en.wikipedia.org/wiki/Spherical_coordinate_system>`_
    used in physics, which is: r positive outward, theta positive downward, and
    phi positive counter-clockwise.

.. contents::
    :local:

Receiver
--------

.. autoclass:: instaseis.source.Receiver
    :members:


....


Source
------

.. autoclass:: instaseis.source.Source
    :members:


....


ForceSource
-----------

.. autoclass:: instaseis.source.ForceSource
    :members:


....


FiniteSource
------------

.. autoclass:: instaseis.source.FiniteSource
    :members:
