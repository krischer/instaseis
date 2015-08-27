#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This is just a polyfill until ObsPy 0.11 has been released.

:copyright:
    Martin van Driel (Martin@vanDriel.de), 2014
    Lion Krischer (Martin@vanDriel.de), 2015
:license:
    GNU Lesser General Public License, Version 3 [non-commercial/academic use]
    (http://www.gnu.org/copyleft/lgpl.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.utils import native_str

import ctypes as C
import math

import obspy
import numpy as np


from .helpers import load_lib

lib = load_lib()


lib.lanczos_resample.argtypes = [
    # y_in
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1,
                           flags=native_str('C_CONTIGUOUS')),
    # y_out
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1,
                           flags=native_str('C_CONTIGUOUS')),
    # dt
    C.c_double,
    # offset
    C.c_double,
    # len_in
    C.c_int,
    # len_out,
    C.c_int,
    # a,
    C.c_int,
    # window
    C.c_int]
lib.lanczos_resample.restype = C.c_void_p


def _validate_parameters(data, old_start, old_dt, new_start, new_dt, new_npts):
    """
    Validates the parameters for various interpolation functions.

    Returns the old and the new end.
    """
    if new_dt <= 0.0:
        raise ValueError("The time step must be positive.")

    # Check for 1D array.
    if data.ndim != 1 or not len(data) or not data.shape[-1]:
        raise ValueError("Not a 1D array.")

    old_end = old_start + old_dt * (len(data) - 1)
    new_end = new_start + new_dt * (new_npts - 1)

    if old_start > new_start or old_end < new_end:
        raise ValueError("The new array must be fully contained in the old "
                         "array. No extrapolation can be performed.")

    return old_end, new_end


# Map corresponding to the enum on the C side of things.
_LANCZOS_KERNEL_MAP = {
    "lanczos": 0,
    "hanning": 1,
    "blackman": 2
}


def interpolate_trace(trace, sampling_rate, a, window="lanczos",
                      starttime=None, npts=None, time_shift=0.0):
    """
    Performs a Lanczos interpolation on a trace. Usage as trace.interpolate()
    in ObsPy.

    Will be removed once ObsPy 0.11 is released.
    """
    dt = float(sampling_rate)
    if dt <= 0.0:
        raise ValueError("The time step must be positive.")
    dt = 1.0 / sampling_rate

    # We just shift the old start time. The interpolation will take care
    # of the rest.
    if time_shift:
        trace.stats.starttime += time_shift

    try:
        func = lanczos_interpolation
        old_start = trace.stats.starttime.timestamp
        old_dt = trace.stats.delta

        if starttime is not None:
            try:
                starttime = starttime.timestamp
            except AttributeError:
                pass
        else:
            starttime = trace.stats.starttime.timestamp
        endtime = trace.stats.endtime.timestamp
        if npts is None:
            npts = int(math.floor((endtime - starttime) / dt)) + 1

        trace.data = np.atleast_1d(func(
            np.require(trace.data, dtype=np.float64), old_start, old_dt,
            starttime, dt, npts, a=a, window=window))
        trace.stats.starttime = obspy.UTCDateTime(starttime)
        trace.stats.delta = dt
    except:
        # Revert the start time change if something went wrong.
        if time_shift:
            trace.stats.starttime -= time_shift
        # re-raise last exception.
        raise

    return trace


def lanczos_interpolation(data, old_start, old_dt, new_start, new_dt, new_npts,
                          a, window="lanczos", *args, **kwargs):
    """
    Function performing Lanczos resampling, see
    http://en.wikipedia.org/wiki/Lanczos_resampling for details. Essentially a
    finite support version of sinc resampling (the ideal reconstruction
    filter). For large values of ``a`` it converges towards sinc resampling. If
    used for downsampling, make sure to apply a proper anti-aliasing lowpass
    filter first.

    .. note::

        In most cases you do not want to call this method directly but invoke
        it via either the :meth:`obspy.core.stream.Stream.interpolate` or the
        :meth:`obspy.core.trace.Trace.interpolate` method. These offer a nicer
        API that naturally integrates with the rest of ObsPy. Use
        ``method="lanczos"`` to use this interpolation method. In that case the
        only additional parameters of interest are ``a`` and ``window``.

    :type data: array_like
    :param data: Array to interpolate.
    :type old_start: float
    :param old_start: The start of the array as a number.
    :type old_start: float
    :param old_dt: The time delta of the current array.
    :type new_start: float
    :param new_start: The start of the interpolated array. Must be greater
        or equal to the current start of the array.
    :type new_dt: float
    :param new_dt: The desired new time delta.
    :type new_npts: int
    :param new_npts: The new number of samples.
    :type a: int
    :param a: The width of the window in samples on either side. Runtimes
        scales linearly with the value of ``a`` but the interpolation also get
        better.
    :type window: str
    :param window: The window used to multiply the sinc function with. One
        of ``"lanczos"``, ``"hanning"``, ``"blackman"``. The window determines
        the trade-off between "sharpness" and the amplitude of the wiggles in
        the pass and stop band. Please use the
        :func:`~obspy.signal.interpolation.plot_lanczos_windows` function to
        judge these for any given application.

    Values of ``a`` >= 20 show good results even for data that has
    energy close to the Nyquist frequency. If your data is way oversampled
    you can get away with much smaller ``a``'s.

    To get an idea of the response of the filter and the effect of the
    different windows, please use the
    :func:`~obspy.signal.interpolation.plot_lanczos_windows` function.

    Also be aware of any boundary effects. All values outside the data
    range are assumed to be zero which matters when calculating interpolated
    values at the boundaries. At each side the area with potential boundary
    effects is ``a`` * ``old_dt``. If you want to avoid any boundary effects
    you will have to remove these values.

    **Mathematical Details:**

    The :math:`sinc` function is defined as

    .. math::

        sinc(t) = \frac{\sin(\pi t)}{\pi t}.

    The Lanczos kernel is then given by a multiplication of the :math:`sinc`
    function with an additional window function resulting in a finite support
    kernel.

    .. math::

        \begin{align}
            L(t) =
            \begin{cases}
                sinc(t)\, \cdot sinc(t/a)
                    & \text{if } t \in [-a, a]
                    \text{ and `window`} = \text{`lanczos`}\\
                sinc(t)\, \cdot \frac{1}{2}
                (1 + \cos(\pi\, t/a))
                    & \text{if } t \in [-a, a]
                    \text{ and `window`} = \text{`hanning`}\\
                sinc(t)\, \cdot \left( \frac{21}{50} + \frac{1}{2}
                \cos(\pi\, t/a) + \frac{2}{25} \cos (2\pi\, t/a) \right)
                    & \text{if } t \in [-a, a] \text{ and `window`} =
                    \text{`blackman`}\\
                0                     & \text{else,}
            \end{cases}
        \end{align}


    Finally interpolation is performed by convolving the discrete signal
    :math:`s_i` with that this kernel and evaluating it at the new timesamples
    :math:`t_j`:

    .. math::

        \begin{align}
            S(t_j) = \sum_{i = \left \lfloor{x}\right \rfloor - a + 1}
                          ^{\left \lfloor{x}\right \rfloor + a}
            s_i L(t_j - i),
        \end{align}

    where :math:`\lfloor \cdot \rfloor` denotes the floor function. For more
    details and justification please see [Burger2009]_ and [vanDriel2015]_.
    """
    _validate_parameters(data, old_start, old_dt, new_start, new_dt, new_npts)
    dt_factor = float(new_dt) / old_dt
    offset = (new_start - old_start) / float(old_dt)
    if offset < 0:
        raise ValueError("Cannot extrapolate.")

    if a < 1:
        raise ValueError("a must be at least 1.")

    return_data = np.zeros(new_npts, dtype="float64")

    lib.lanczos_resample(np.require(data, np.float64), return_data, dt_factor,
                         offset, len(data), len(return_data), int(a), 0)
    return return_data
