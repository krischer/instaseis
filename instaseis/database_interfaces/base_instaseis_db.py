#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Abstract base class for all Instaseis database classes.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
    Martin van Driel (Martin@vanDriel.de), 2014
:license:
    GNU Lesser General Public License, Version 3 [non-commercial/academic use]
    (http://www.gnu.org/copyleft/lgpl.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from future.utils import with_metaclass

from abc import ABCMeta, abstractmethod
import math
import warnings

import numpy as np
from obspy.core import AttribDict, Stream, Trace, UTCDateTime
from obspy.geodetics import locations2degrees
from obspy.signal.interpolation import lanczos_interpolation
from scipy.integrate import cumtrapz
import scipy.signal

from ..source import Source, ForceSource, Receiver
from ..helpers import get_band_code, sizeof_fmt, rfftfreq


DEFAULT_MU = 32e9


KIND_MAP = {
    'displacement': 0,
    'velocity': 1,
    'acceleration': 2}


INV_KIND_MAP = dict((j, i) for i, j in KIND_MAP.items())


STF_MAP = {
    'errorf': 0,
    'quheavi': 0,
    'dirac_0': 1,
    'gauss_0': 1,
    'gauss_1': 2,
    'gauss_2': 3}


def _diff_and_integrate(n_derivative, data, comp, dt_out):
    for _ in np.arange(n_derivative):
        data[comp] = np.gradient(data[comp], [dt_out])

    # Cannot happen currently - maybe with other source time functions?
    for _ in np.arange(-n_derivative):  # pragma: no cover
        # adding a zero at the beginning to avoid phase shift
        data[comp] = cumtrapz(data[comp], dx=dt_out, initial=0.0)


class BaseInstaseisDB(with_metaclass(ABCMeta)):
    """
    Base class for all Instaseis database classes defining the user interface.
    """
    def get_greens_function(self, epicentral_distance_in_degree,
                            source_depth_in_m, origin_time=UTCDateTime(0),
                            kind='displacement', return_obspy_stream=True,
                            dt=None, kernelwidth=12, definition='seiscomp'):
        """
        Extract Green's function from the Green's function database.

        Currently only one definition is implemented: the one assumed by
        Seiscomp, i.e. the components ``TSS``, ``ZSS``, ``RSS``, ``TDS``,
        ``ZDS``, ``RDS``, ``ZDD``, ``RDD``, ``ZEP``, ``REP`` as defined in

        .. list-table::

            * - | Minson, Sarah E., and Douglas S. Dreger (2008)
                | **Stable Inversions for Complete Moment Tensors.**
                | *Geophysical Journal International* 174 (2): 585–592.
                | http://dx.doi.org/10.1111/j.1365-246X.2008.03797.x

        :param epicentral_distance_in_degree: The epicentral distance in
            degree.
        :type epicentral_distance_in_degree: float
        :param source_depth_in_m: The source depth in m below the surface.
        :type source_depth_in_m: float
        :param origin_time: Origin time of the source.
        :type origin_time: :class:`obspy.core.utcdatetime.UTCDateTime`
        :param kind: The desired units of the seismogram:
            ``"displacement"``, ``"velocity"``, or ``"acceleration"``.
        :type kind: str
        :param dt: Desired sampling rate of the Green's functions. Resampling
            is done using a Lanczos kernel.
        :type dt: float
        :param kernelwidth: The width of the sinc kernel used for resampling in
            terms of the original sampling interval. Best choose something
            between 10 and 20.
        :type kernelwidth: int
        :param definition: The desired Green's function definition.
        :type definition: str

        :returns: Multi component seismograms.
        :rtype: A :class:`obspy.core.stream.Stream` object or a dictionary
            with NumPy arrays as values.
        """
        # Currently only the seiscomp definition is implemented. Other
        # implementations will require a refactoring of this method.
        if definition.lower() != "seiscomp":
            raise NotImplementedError

        self._get_greens_seiscomp_sanity_checks(epicentral_distance_in_degree,
                                                source_depth_in_m, kind, dt=dt)

        src_latitude, src_longitude = 90., 0.
        rec_latitude, rec_longitude = 90. - epicentral_distance_in_degree, 0.

        # sources according to https://github.com/krischer/instaseis/issues/8
        # transformed to r, theta, phi
        #
        # Mtt =  Mxx, Mpp = Myy, Mrr =  Mzz
        # Mrp = -Myz, Mrt = Mxz, Mtp = -Mxy
        #
        # Mrr   Mtt   Mpp    Mrt    Mrp    Mtp
        #  0     0     0      0      0     -1.0    m1
        #  0     1.0  -1.0    0      0      0      m2
        #  0     0     0      0     -1.0    0      m3
        #  0     0     0      1.0    0      0      m4
        #  1.0   1.0   1.0    0      0      0      m6
        #  2.0  -1.0  -1.0    0      0      0      cl

        m1 = Source(src_latitude, src_longitude, source_depth_in_m,
                    m_tp=-1.0, origin_time=origin_time)
        m2 = Source(src_latitude, src_longitude, source_depth_in_m,
                    m_tt=1.0, m_pp=-1.0, origin_time=origin_time)
        m3 = Source(src_latitude, src_longitude, source_depth_in_m,
                    m_rp=-1.0, origin_time=origin_time)
        m4 = Source(src_latitude, src_longitude, source_depth_in_m,
                    m_rt=1.0, origin_time=origin_time)
        m6 = Source(src_latitude, src_longitude, source_depth_in_m,
                    m_rr=1.0, m_tt=1.0, m_pp=1.0, origin_time=origin_time)
        cl = Source(src_latitude, src_longitude, source_depth_in_m,
                    m_rr=2.0, m_tt=-1.0, m_pp=-1.0, origin_time=origin_time)

        receiver = Receiver(rec_latitude, rec_longitude)

        # same kwarguments for many callse
        args = {'receiver': receiver,
                'dt': dt,
                'kind': kind,
                'kernelwidth': kernelwidth,
                'return_obspy_stream': False}

        # Collect data arrays a dictionary.
        data = {}
        # on first call extract mu as well
        tmp_dict = self.get_seismograms(
            source=m1, components='T', **args)
        data['mu'] = tmp_dict['mu']
        data['TSS'] = tmp_dict['T']

        data['ZSS'] = self.get_seismograms(
            source=m2, components='Z', **args)['Z']
        data['RSS'] = self.get_seismograms(
            source=m2, components='R', **args)['R']

        data['TDS'] = self.get_seismograms(
            source=m3, components='T', **args)['T']

        data['ZDS'] = self.get_seismograms(
            source=m4, components='Z', **args)['Z']
        data['RDS'] = self.get_seismograms(
            source=m4, components='R', **args)['R']

        data['ZDD'] = self.get_seismograms(
            source=cl, components='Z', **args)['Z']
        data['RDD'] = self.get_seismograms(
            source=cl, components='R', **args)['R']

        data['ZEP'] = self.get_seismograms(
            source=m6, components='Z', **args)['Z']
        data['REP'] = self.get_seismograms(
            source=m6, components='R', **args)['R']

        if return_obspy_stream:
            if dt is None:
                dt_out = self.info.dt
            else:
                dt_out = dt
            components = list(data.keys())
            components.remove('mu')
            return self._convert_to_stream(
                receiver=receiver, components=components,
                data=data, dt_out=dt_out, starttime=UTCDateTime(0),
                add_band_code=False)
        else:
            return data

    def get_seismograms(self, source, receiver, components=None,
                        kind='displacement', remove_source_shift=True,
                        reconvolve_stf=False, return_obspy_stream=True,
                        dt=None, kernelwidth=12):
        """
        Extract seismograms from the Green's function database.

        :param source: The source definition.
        :type source: :class:`instaseis.source.Source` or
            :class:`instaseis.source.ForceSource`
        :param receiver: The seismic receiver.
        :type receiver: :class:`instaseis.source.Receiver`
        :type components: tuple of str, optional
        :param components: Which components to calculate. Must be a tuple
            containing any combination of ``"Z"``, ``"N"``, ``"E"``,
            ``"R"``, and ``"T"``. Defaults to ``["Z", "N", "E"]`` for two
            component databases, to ``["N", "E"]`` for horizontal only
            databases, and to ``["Z"]`` for vertical only databases.
        :type kind: str, optional
        :param kind: The desired units of the seismogram:
            ``"displacement"``, ``"velocity"``, or ``"acceleration"``.
        :type remove_source_shift: bool, optional
        :param remove_source_shift: Cut all samples before the peak of the
            source time function. This has the effect that the first sample
            is the origin time of the source.
        :type reconvolve_stf: bool, optional
        :param reconvolve_stf: Deconvolve the source time function used in
            the AxiSEM run and convolve with the STF attached to the source.
            For this to be stable, the new STF needs to bandlimited.
        :type return_obspy_stream: bool, optional
        :param return_obspy_stream: Return format is either an
            :class:`obspy.core.stream.Stream` object or a dictionary
            containing the raw NumPy arrays.
        :type dt: float, optional
        :param dt: Desired sampling rate of the seismograms. Resampling is done
            using a Lanczos kernel.
        :type kernelwidth: int, optional
        :param kernelwidth: The width of the sinc kernel used for resampling in
            terms of the original sampling interval. Best choose something
            between 10 and 20.

        :returns: Multi component seismograms.
        :rtype: A :class:`obspy.core.stream.Stream` object or a dictionary
            with NumPy arrays as values.
        """
        if components is None:
            components = self.default_components

        source, receiver = self._get_seismograms_sanity_checks(
            source=source, receiver=receiver, components=components,
            kind=kind, dt=dt)

        # Call the _get_seismograms() method of the respective implementation.
        data = self._get_seismograms(source=source, receiver=receiver,
                                     components=components)

        if dt is None:
            dt_out = self.info.dt
        else:
            dt_out = dt

        stf_deconv_map = {
            0: self.info.sliprate,
            1: self.info.slip}

        # Can never be negative with the current logic.
        n_derivative = KIND_MAP[kind] - STF_MAP[self.info.stf]

        if isinstance(source, ForceSource):
            n_derivative += 1

        if reconvolve_stf and remove_source_shift:
            raise ValueError("'remove_source_shift' argument not "
                             "compatible with 'reconvolve_stf'.")

        # Calculate the final time information about the seismograms.
        time_information = _get_seismogram_times(
            info=self.info, origin_time=source.origin_time, dt=dt,
            kernelwidth=kernelwidth, remove_source_shift=remove_source_shift,
            reconvolve_stf=reconvolve_stf)

        for comp in components:
            if reconvolve_stf:
                # We assume here that the sliprate is well-behaved,
                # e.g. zeros at the boundaries and no energy above the mesh
                # resolution.
                if source.dt is None or source.sliprate is None:
                    raise ValueError("source has no source time function")

                if STF_MAP[self.info.stf] not in [0, 1]:
                    raise NotImplementedError(
                        'deconvolution not implemented for stf %s'
                        % (self.info.stf))

                stf_deconv_f = np.fft.rfft(
                    stf_deconv_map[STF_MAP[self.info.stf]],
                    n=self.info.nfft)

                if abs((source.dt - self.info.dt) / self.info.dt) > 1e-7:
                    raise ValueError("dt of the source not compatible")

                stf_conv_f = np.fft.rfft(source.sliprate,
                                         n=self.info.nfft)

                if source.time_shift is not None:
                    stf_conv_f *= \
                        np.exp(- 1j * rfftfreq(self.info.nfft) *
                               2. * np.pi * source.time_shift / self.info.dt)

                # Apply a 5 percent, at least 5 samples taper at the end.
                # The first sample is guaranteed to be zero in any case.
                tlen = max(int(math.ceil(0.05 * len(data[comp]))), 5)
                taper = np.ones_like(data[comp])
                taper[-tlen:] = scipy.signal.hann(tlen * 2)[tlen:]
                dataf = np.fft.rfft(taper * data[comp], n=self.info.nfft)

                # Ensure numerical stability by not dividing with zero.
                f = stf_conv_f
                _l = np.abs(stf_deconv_f)
                _idx = np.where(_l > 0.0)
                f[_idx] /= stf_deconv_f[_idx]
                f[_l == 0] = 0 + 0j

                data[comp] = np.fft.irfft(dataf * f)[:self.info.npts]

            if dt is not None:
                data[comp] = lanczos_interpolation(
                    data=np.require(data[comp], requirements=["C"]),
                    old_start=0, old_dt=self.info.dt,
                    new_start=time_information["time_shift_at_beginning"],
                    new_dt=dt,
                    new_npts=time_information["npts_before_shift_removal"],
                    a=kernelwidth,
                    window="blackman")

            # Integrate/differentiate before removing the source shift in
            # order to reduce boundary effects at the start of the signal.
            #
            # NEVER to this before the resampling! The error can be really big.
            if n_derivative:
                _diff_and_integrate(n_derivative=n_derivative, data=data,
                                    comp=comp, dt_out=dt_out)

            # If desired, remove the samples before the peak of the source
            # time function.
            if remove_source_shift:
                data[comp] = data[comp][time_information["ref_sample"]:]

        if return_obspy_stream:
            return self._convert_to_stream(
                receiver=receiver, components=components, data=data,
                dt_out=dt_out, starttime=time_information["starttime"])
        else:
            return data

    @staticmethod
    def _convert_to_stream(receiver, components, data, dt_out, starttime,
                           add_band_code=True):
        # Convert to an ObsPy Stream object.
        st = Stream()
        band_code = get_band_code(dt_out)
        instaseis_header = AttribDict(mu=data["mu"])

        for comp in components:
            tr = Trace(
                data=data[comp],
                header={"delta": dt_out,
                        "starttime": starttime,
                        "station": receiver.station,
                        "network": receiver.network,
                        "location": receiver.location,
                        "channel": add_band_code * (band_code + 'X') + comp,
                        "instaseis": instaseis_header})
            st += tr
        return st

    @abstractmethod
    def _get_seismograms(self, source, receiver, components=("Z", "N", "E")):
        raise NotImplementedError

    @abstractmethod
    def _get_info(self):
        """
        Must return a dictionary with the following keys:

        ``"is_reciprocal"``, ``"components"``, ``"source_depth"``,
        ``"velocity_model"``, ``"external_model_name"``, ``"attenuation"``,
        ``"period"``, ``"dump_type"``, ``"excitation_type"``, ``"dt"``,
        ``"sampling_rate"``, ``"npts"``, ``"nnft"``, ``"length"``, ``"stf"``,
        ``"slip"``, ``"sliprate"``, ``"src_shift"``, ``"src_shift_samples"``,
        ``"spatial_order"``, ``"min_radius"``, ``"max_radius"``,
        ``"planet_radius"``, ``"min_d"``, ``"max_d"``, ``"time_scheme"``,
        ``"directory"``, ``"filesize"``, ``"compiler"``, ``"user"``,
        ``"format_version"``, ``"axisem_version"``, ``"datetime"``
        """
        raise NotImplementedError

    def get_seismograms_finite_source(self, sources, receiver,
                                      components=None,
                                      kind='displacement', dt=None,
                                      kernelwidth=12, correct_mu=False,
                                      progress_callback=None):
        """
        Extract seismograms for a finite source from an Instaseis database.

        :param sources: A collection of point sources.
        :type sources: :class:`~instaseis.source.FiniteSource` or list of
            :class:`~instaseis.source.Source` objects.
        :param receiver: The seismic receiver.
        :type receiver: :class:`instaseis.source.Receiver`
        :type components: tuple of str, optional
        :param components: Which components to calculate. Must be a tuple
            containing any combination of ``"Z"``, ``"N"``, ``"E"``,
            ``"R"``, and ``"T"``. Defaults to ``["Z", "N", "E"]`` for two
            component databases, to ``["N", "E"]`` for horizontal only
            databases, and to ``["Z"]`` for vertical only databases.
        :type kind: str, optional
        :param kind: The desired units of the seismogram:
            ``"displacement"``, ``"velocity"``, or ``"acceleration"``.
        :type dt: float, optional
        :param dt: Desired sampling rate of the seismograms. Resampling is done
            using a Lanczos kernel.
        :type kernelwidth: int, optional
        :param kernelwidth: The width of the sinc kernel used for resampling in
            terms of the original sampling interval. Best choose something
            between 10 and 20.
        :type correct_mu: bool, optional
        :param correct_mu: Correct the source magnitude for the actual shear
            modulus from the model.
        :type progress_callback: function, optional
        :param progress_callback: Optional callback function that will be
            called with current source number and the number of total
            sources for each calculated source. Useful for integration into
            user interfaces to provide some kind of progress information. If
            the callback returns ``True``, the calculation will be cancelled.

        :returns: Multi component finite source seismogram.
        :rtype: :class:`obspy.core.stream.Stream`
        """
        if components is None:
            components = self.default_components

        if not self.info.is_reciprocal:
            raise NotImplementedError

        data_summed = {}
        count = len(sources)
        for _i, source in enumerate(sources):
            # Don't perform the diff/integration here, but after the
            # resampling later on.
            data = self.get_seismograms(
                source, receiver, components, reconvolve_stf=True,
                # Effectively results in nothing happening.
                kind=INV_KIND_MAP[STF_MAP[self.info.stf]],
                return_obspy_stream=False, remove_source_shift=False)

            if correct_mu:
                corr_fac = data["mu"] / DEFAULT_MU,
            else:
                corr_fac = 1

            for comp in components:
                if comp in data_summed:
                    data_summed[comp] += data[comp] * corr_fac
                else:
                    data_summed[comp] = data[comp] * corr_fac
            # Only used for the GUI.
            if progress_callback:  # pragma: no cover
                cancel = progress_callback(_i + 1, count)
                if cancel:
                    return None

        if dt is not None:
            for comp in components:
                # We don't need to align a sample to the peak of the source
                # time function here.
                new_npts = int(round(
                    (len(data[comp]) - 1) * self.info.dt / dt, 6) + 1)
                data_summed[comp] = lanczos_interpolation(
                    data=np.require(data_summed[comp], requirements=["C"]),
                    old_start=0, old_dt=self.info.dt, new_start=0, new_dt=dt,
                    new_npts=new_npts, a=kernelwidth, window="blackman")

                # The resampling assumes zeros outside the data range. This
                # does not introduce any errors at the beginning as the data is
                # actually zero there but it does affect the end. We will
                # remove all samples that are affected by the boundary
                # conditions here.
                #
                # Also don't cut it for the "identify" interpolation which is
                # important for testing.
                if round(dt / self.info.dt, 6) != 1.0:
                    affected_area = kernelwidth * self.info.dt
                    data_summed[comp] = \
                        data_summed[comp][:-int(np.ceil(affected_area / dt))]

        if dt is None:
            dt_out = self.info.dt
        else:
            dt_out = dt

        # Integrate/differentiate here. No need to do it for every single
        # seismogram and stack the errors.
        n_derivative = KIND_MAP[kind] - STF_MAP[self.info.stf]
        if n_derivative:
            for comp in data_summed.keys():
                _diff_and_integrate(n_derivative=n_derivative,
                                    data=data_summed, comp=comp, dt_out=dt_out)

        # Convert to an ObsPy Stream object.
        st = Stream()
        band_code = get_band_code(dt_out)
        for comp in components:
            tr = Trace(data=data_summed[comp],
                       header={"delta": dt_out,
                               "station": receiver.station,
                               "network": receiver.network,
                               "location": receiver.location,
                               "channel": "%sX%s" % (band_code, comp)})
            st += tr
        return st

    def _get_greens_seiscomp_sanity_checks(self, epicentral_distance_degree,
                                           source_depth_in_m, kind, dt):
        """
        Common sanity checks for the get_greens_seiscomp method.

        :param epicentral_distance_degree: The epicentral distance in degree.
        :type epicentral_distance_degree: float
        :param source_depth_in_m: The source depth in m below the surface.
        :type source_depth_in_m: float
        :param kind: The desired units of the seismogram:
            ``"displacement"``, ``"velocity"``, or ``"acceleration"``.
        :type kind: str
        """
        if dt is not None:
            if dt <= 0.0:
                raise ValueError("dt must be bigger than 0.")
            elif dt > self.info.dt:
                raise ValueError(
                    "The database is sampled with a sample spacing of %.3f "
                    "seconds. You must not pass a 'dt' larger than that as "
                    "that would be a downsampling operation which Instaseis "
                    "does not do." % self.info.dt)

        if kind not in ['displacement', 'velocity', 'acceleration']:
            raise ValueError("unknown kind '%s'." % (kind,))

        if not self.info.is_reciprocal:
            raise ValueError('forward DB cannot be used with '
                             'get_greens_function()')

        if not self.info.components == 'vertical and horizontal':
            raise ValueError('get_greens_function() needs a DB with both '
                             'vertical and horizontal components')

        d = epicentral_distance_degree
        if not self.info.min_d <= d <= self.info.max_d:
            raise ValueError(
                'epicentral_distance_degree should be in [%.1f, %.1f]' % (
                    self.info.min_d, self.info.max_d))

        # Check source depth.
        src_radius = self.info.planet_radius - source_depth_in_m
        if src_radius < self.info.min_radius:
            msg = (
                "Source too deep. Source would be located at a radius of "
                "%.1f meters. The database supports source radii from "
                "%.1f to %.1f meters." % (src_radius, self.info.min_radius,
                                          self.info.max_radius))
            raise ValueError(msg)
        elif src_radius > self.info.max_radius:
            msg = (
                "Source is too shallow. Source would be located at a "
                "radius of %.1f meters. The database supports source "
                "radii from %.1f to %.1f meters." % (
                    src_radius, self.info.min_radius,
                    self.info.max_radius))
            raise ValueError(msg)

    def _get_seismograms_sanity_checks(self, source, receiver, components,
                                       kind, dt):
        """
        Common sanity checks for the get_seismograms method. Also parses
        source and receiver objects if necessary.

        :param source: instaseis.Source or instaseis.ForceSource object
        :type source: :class:`instaseis.source.Source` or
            :class:`instaseis.source.ForceSource`
        :param receiver: instaseis.Receiver object
        :type receiver: :class:`instaseis.source.Receiver`
        :param components: a tuple containing any combination of the
            strings ``"Z"``, ``"N"``, ``"E"``, ``"R"``, and ``"T"``
        :param kind: 'displacement', 'velocity' or 'acceleration'
        """
        if dt is not None:
            if dt <= 0.0:
                raise ValueError("dt must be bigger than 0.")
            elif dt > self.info.dt:
                raise ValueError(
                    "The database is sampled with a sample spacing of %.3f "
                    "seconds. You must not pass a 'dt' larger than that as "
                    "that would be a downsampling operation which Instaseis "
                    "does not do." % self.info.dt)

        # Attempt to parse them if the types are not correct.
        if not isinstance(source, Source) and \
                not isinstance(source, ForceSource):
            source = Source.parse(source)
        if not isinstance(receiver, Receiver):
            # This only works in the special case of one station, otherwise
            # it has to be called more then once.
            rec = Receiver.parse(receiver)
            if len(rec) != 1:
                raise ValueError("Receiver object/file contains multiple "
                                 "stations. Please parse outside the "
                                 "get_seismograms() function and call in a "
                                 "loop.")
            receiver = rec[0]

        if kind not in ['displacement', 'velocity', 'acceleration']:
            raise ValueError("unknown kind '%s'" % (kind,))

        for comp in components:
            if comp not in ["N", "E", "Z", "R", "T"]:
                raise ValueError("Invalid component: %s" % comp)

        if self.info.is_reciprocal:
            if receiver.depth_in_m is not None:
                warnings.warn('Receiver depth cannot be changed when reading '
                              'from reciprocal DB. Using depth from the DB.')

            if any(comp in components for comp in ['N', 'E', 'R', 'T']) and \
                    "horizontal" not in self.info.components:
                raise ValueError("vertical component only DB")

            if 'Z' in components and "vertical" not in self.info.components:
                raise ValueError("horizontal component only DB")

        else:
            if source.depth_in_m is not None:
                warnings.warn('Source depth cannot be changed when reading '
                              'from forward DB. Using depth from the DB.')

        # Make sure that the source is within the domain.
        if self.info.is_reciprocal and source.depth_in_m is not None:
            src_radius = self.info.planet_radius - source.depth_in_m
            if src_radius < self.info.min_radius:
                msg = (
                    "Source too deep. Source would be located at a radius of "
                    "%.1f meters. The database supports source radii from "
                    "%.1f to %.1f meters." % (src_radius, self.info.min_radius,
                                              self.info.max_radius))
                raise ValueError(msg)
            elif src_radius > self.info.max_radius:
                msg = (
                    "Source is too shallow. Source would be located at a "
                    "radius of %.1f meters. The database supports source "
                    "radii from %.1f to %.1f meters." % (
                        src_radius, self.info.min_radius,
                        self.info.max_radius))
                raise ValueError(msg)
        elif not self.info.is_reciprocal and receiver.depth_in_m is not None:
            rec_radius = self.info.planet_radius - receiver.depth_in_m
            if rec_radius < self.info.min_radius:
                msg = (
                    "Receiver too deep. Receiver would be located at a radius "
                    "of %.1f meters. The database supports receiver radii "
                    "from %.1f to %.1f meters." % (
                        rec_radius, self.info.min_radius,
                        self.info.max_radius))
                raise ValueError(msg)
            elif rec_radius > self.info.max_radius:
                msg = (
                    "Receiver is too shallow. Receiver would be located at a "
                    "radius of %.1f meters. The database supports receiver "
                    "radii from %.1f to %.1f meters." % (
                        rec_radius, self.info.min_radius,
                        self.info.max_radius))
                raise ValueError(msg)

        d = locations2degrees(source.latitude, source.longitude,
                              receiver.latitude, receiver.longitude)
        if not self.info.min_d <= d <= self.info.max_d:
            raise ValueError(
                'Epicentral distance is %.1f but should be in [%.1f, '
                '%.1f].' % (d, self.info.min_d, self.info.max_d))

        return source, receiver

    @property
    def info(self):
        try:
            return self.__cached_info
        except:
            pass
        self.__cached_info = AttribDict(self._get_info())
        return self.__cached_info

    def _repr_pretty_(self, p, cycle):  # pragma: no cover
        p.text(str(self))

    def __str__(self):
        info = self.info

        return_str = (
            "{db} {reciprocal} Green's function Database (v{"
            "format_version}) "
            "generated with these parameters:\n"
            "\tcomponents           : {components}\n"
            "{source_depth}"
            "\tvelocity model       : {velocity_model}\n"
            "\tattenuation          : {attenuation}\n"
            "\tdominant period      : {period:.3f} s\n"
            "\tdump type            : {dump_type}\n"
            "\texcitation type      : {excitation_type}\n"
            "\ttime step            : {dt:.3f} s\n"
            "\tsampling rate        : {sampling_rate:.3f} Hz\n"
            "\tnumber of samples    : {npts}\n"
            "\tseismogram length    : {length:.1f} s\n"
            "\tsource time function : {stf}\n"
            "\tsource shift         : {src_shift:.3f} s\n"
            "\tspatial order        : {spatial_order}\n"
            "\tmin/max radius       : {min_radius:.1f} - {max_radius:.1f} km\n"
            "\tPlanet radius        : {planet_radius:.1f} km\n"
            "\tmin/max distance     : {min_d:.1f} - {max_d:.1f} deg\n"
            "\ttime stepping scheme : {time_scheme}\n"
            "\tcompiler/user        : {compiler} by {user}\n"
            "\tdirectory/url        : {directory}\n"
            "\tsize of netCDF files : {filesize}\n"
            "\tgenerated by AxiSEM version {axisem_version} at {datetime}\n"
        ).format(
            db=self.__class__.__name__,
            reciprocal="reciprocal" if info.is_reciprocal else "forward",
            components=info.components,
            source_depth=(
                "\tsource depth         : %.2f km\n" %
                info.source_depth) if info.source_depth is not None else "",
            velocity_model=info.velocity_model,
            attenuation=info.attenuation,
            period=info.period,
            dump_type=info.dump_type,
            excitation_type=info.excitation_type,
            dt=info.dt,
            sampling_rate=info.sampling_rate,
            npts=info.npts,
            length=info.length,
            stf=info.stf,
            src_shift=info.src_shift,
            spatial_order=info.spatial_order,
            min_radius=info.min_radius / 1.0E3,
            max_radius=info.max_radius / 1.0E3,
            planet_radius=info.planet_radius / 1.0E3,
            min_d=info.min_d,
            max_d=info.max_d,
            time_scheme=info.time_scheme,
            directory=info.directory,
            filesize=sizeof_fmt(info.filesize),
            compiler=info.compiler,
            user=info.user,
            format_version=info.format_version,
            axisem_version=info.axisem_version,
            datetime=info.datetime
        )
        return return_str

    @property
    def default_components(self):
        """
        The components returned by default by most of the higher level
        routines.
        """
        c = self.available_components
        if len(c) == 5:
            c = ["Z", "N", "E"]
        elif len(c) == 4:
            c = ["N", "E"]
        elif len(c):
            c = ["Z"]
        else:  # pragma: no cover
            raise NotImplementedError
        return c

    @property
    def available_components(self):
        """
        Returns a list with all available components.
        """
        if self.info.components == "4 elemental moment tensors":
            return ["Z", "N", "E", "R", "T"]
        components = []
        if "vertical" in self.info.components:
            components.append("Z")
        if "horizontal" in self.info.components:
            components.extend(["N", "E", "R", "T"])
        return components


def _get_seismogram_times(info, origin_time, dt, kernelwidth,
                          remove_source_shift, reconvolve_stf=False):
    """
    Helper function to calculate the final times of seismograms.

    It also calculates all the necessary information to determine the final
    times of the seismograms and how to cut the data. This is important to
    make sure the time handling and calculations are consistent across
    Instaseis.

    :param info: The info dictionary of a database.
    :param dt: The desired new sampling rate. None if not set.
    :param kernelwidth: The width of the interpolation kernel.
    :param remove_source_shift: Remove or don't remove the source time shift.
    :param reconvolve_stf: Set to true if reconvolved with a custom STF,
        then the time shift are no longer applied.

    Returned dictionary has the following keys:

    * ``'ref_sample'``: The reference sample, e.g. the sample in the array
        whose time will be the origin time. This is only valid before the
        source shift is removed.
    * ``'starttime'``: The final start time of the seismogram.
    * ``'npts'``: The final number of samples of the seismogram.
    * ``'samples_cut_at_end'``: The number of samples cut at the end.
    * ``'time_shift_at_beginning'``: The time shift at the beginning if
        resampled to a new sampling rate. This is necessary to always make
        sure one sample is exactly at the origin time.
    * ``'endtime'``: The final end time of the seismogram.
    * ``'npts_before_shift_removal'``: The number of samples before the
        source time shift has been removed.
    """
    if reconvolve_stf and remove_source_shift:
        raise ValueError("'remove_source_shift' argument not "
                         "compatible with 'reconvolve_stf'.")

    dt_out = dt or info.dt

    ti = {}
    ti["samples_cut_at_end"] = 0

    if dt is not None:
        if not reconvolve_stf:
            # This is a bit tricky. We want to resample but we also want
            # to make sure that that the peak of the source time
            # function is exactly hit by a sample point.

            # If it cleanly divides within ten microseconds,
            # make integer based calculations.
            if round(info.src_shift / dt, 5) % 1.0 == 0:
                ref_sample = int(round(info.src_shift / dt, 5))
                shift = (info.src_shift_samples * info.dt) - \
                        (ref_sample * dt)
                shift = round(shift, 6)
                duration = (info.npts - 1) * info.dt - shift
                new_npts = int(round(duration / dt, 6)) + 1
            else:
                shift = round(info.src_shift % dt, 8)
                duration = (info.npts - 1) * info.dt - shift
                new_npts = int(round(duration / dt, 6)) + 1
                ref_sample = \
                    int(round((info.src_shift - shift) / dt, 6))

            ti["time_shift_at_beginning"] = shift
            ti["npts_before_shift_removal"] = new_npts
        else:
            ti["time_shift_at_beginning"] = 0
            ti["npts_before_shift_removal"] = \
                int(round((info.npts - 1) * info.dt / dt, 6)) + 1
            ref_sample = 0

        # The resampling assumes zeros outside the data range. This
        # does not introduce any errors at the beginning as the data is
        # actually zero there but it does affect the end. We will
        # remove all samples that are affected by the boundary
        # conditions here.
        #
        # Also don't cut it for the "identify" interpolation which is
        # important for testing.
        if round(dt / info.dt, 6) != 1.0:
            affected_area = kernelwidth * info.dt
            ti["samples_cut_at_end"] = \
                int(np.ceil(affected_area / dt))
            ti["npts_before_shift_removal"] -= ti["samples_cut_at_end"]
    else:
        if not reconvolve_stf:
            ref_sample = info.src_shift_samples
        else:
            ref_sample = 0
        ti["time_shift_at_beginning"] = 0
        ti["npts_before_shift_removal"] = info.npts

    # The reference sample, e.g. the sample at origin time.
    ti["ref_sample"] = ref_sample

    if remove_source_shift:
        ti["starttime"] = origin_time
        ti["npts"] = ti["npts_before_shift_removal"] - ti["ref_sample"]
    else:
        ti["starttime"] = origin_time - ti["ref_sample"] * dt_out
        ti["npts"] = ti["npts_before_shift_removal"]
    ti["endtime"] = ti["starttime"] + (ti["npts"] - 1) * dt_out

    return ti
