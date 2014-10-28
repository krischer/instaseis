#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Benchmarks for instaseis.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
from __future__ import print_function

from abc import ABCMeta, abstractmethod, abstractproperty
import argparse
import numpy as np
import os
import random
import sys
import timeit

from instaseis import InstaSeisDB, Source, Receiver

# Write interval.
WRITE_INTERVAL = 0.05


class InstaSeisBenchmark(object):
    __metaclass__ = ABCMeta

    def __init__(self, path, time_per_benchmark):
        self.path = path
        self.time_per_benchmark = time_per_benchmark

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def iterate(self):
        pass

    @abstractproperty
    def description(self):
        pass

    def run(self):
        a = timeit.default_timer()
        self.setup()
        b = timeit.default_timer()
        print("\tTime for initialization: %s sec" % (b - a))

        starttime = timeit.default_timer()
        endtime = starttime + self.time_per_benchmark
        all_times = []

        last_write_time = starttime
        latest_times = []

        print("\tStarting...", end="\r")
        t = starttime
        while (t < endtime):
            s = timeit.default_timer()
            self.iterate()
            t = timeit.default_timer()
            all_times.append(t - s)

            # Pretty and immediate output.
            latest_times.append(t - s)

            if t >= (last_write_time + WRITE_INTERVAL):
                cumtime = sum(latest_times)
                speed = len(latest_times) / cumtime
                print("\tseismograms/sec: {0:>8.2f}, remaining time: "
                      "{1:>2.1f} sec".format(speed, endtime - t),
                      end="\r")
                sys.stdout.flush()
                latest_times = []
                last_write_time = t
        print(80 * " ", end="\r")

        all_times = np.array(all_times, dtype="float64")
        cumtime = sum(all_times)
        count = len(all_times)
        print("\t%i seismograms in %.2f sec" % (count, cumtime))
        print("\t%g sec/seismogram" % all_times.mean())
        print("\t%g seismograms/sec" % (count / cumtime))
        for p in [0, 10, 25, 50, 75, 90, 100]:
            print("\t {0:>3}th percentile: {1} sec".format(
                p, np.percentile(all_times, p)))


class BufferedFixedSrcRecRoDOffSeismogramGeneration(InstaSeisBenchmark):
    def setup(self):
        self.db = InstaSeisDB(self.path, read_on_demand=False,
                              buffer_size_in_mb=250)

    def iterate(self):
        src = Source(latitude=10, longitude=10)
        rec = Receiver(latitude=20, longitude=20)
        self.db.get_seismograms(source=src, receiver=rec)

    @property
    def description(self):
        return "Buffered, fixed source and receiver, " \
               "read_on_demand=False"


class BufferedFixedSrcRecRoDOffSeismogramGenerationNoObsPy(InstaSeisBenchmark):
    def setup(self):
        self.db = InstaSeisDB(self.path, read_on_demand=False,
                              buffer_size_in_mb=250)

    def iterate(self):
        src = Source(latitude=10, longitude=10)
        rec = Receiver(latitude=20, longitude=20)
        self.db.get_seismograms(source=src, receiver=rec,
                                return_obspy_stream=False)

    @property
    def description(self):
        return "Buffered, fixed source and receiver, " \
               "read_on_demand=False, no ObsPy output"


class BufferedFixedSrcRecRoDOnSeismogramGeneration(InstaSeisBenchmark):
    def setup(self):
        self.db = InstaSeisDB(self.path, read_on_demand=True,
                              buffer_size_in_mb=250)

    def iterate(self):
        src = Source(latitude=10, longitude=10)
        rec = Receiver(latitude=20, longitude=20)
        self.db.get_seismograms(source=src, receiver=rec)

    @property
    def description(self):
        return "Buffered, fixed source and receiver, " \
               "read_on_demand=True"


class Buffered2DegreeLatLngDepthScatter(InstaSeisBenchmark):
    def setup(self):
        self.db = InstaSeisDB(self.path, read_on_demand=False,
                              buffer_size_in_mb=250)

    def iterate(self):
        rec = Receiver(latitude=20, longitude=20)
        lat = random.random() * 2
        lat += 44
        lng = random.random() * 2
        lng += 44
        depth_in_m = random.random() * 200000
        src = Source(latitude=lat, longitude=lng, depth_in_m=depth_in_m)
        self.db.get_seismograms(source=src, receiver=rec)

    @property
    def description(self):
        return "Buffered, 2 Degree/200 km source position scatter"


class BufferedHalfDegreeLatLngDepthScatter(InstaSeisBenchmark):
    def setup(self):
        self.db = InstaSeisDB(self.path, read_on_demand=False,
                              buffer_size_in_mb=250)

    def iterate(self):
        rec = Receiver(latitude=20, longitude=20)
        lat = random.random() * 0.5
        lat += 44
        lng = random.random() * 0.5
        lng += 44
        depth_in_m = random.random() * 100000
        src = Source(latitude=lat, longitude=lng, depth_in_m=depth_in_m)
        self.db.get_seismograms(source=src, receiver=rec)

    @property
    def description(self):
        return "Buffered, 0.5 Degree/100 km (depth) source position scatter"


class BufferedFullyRandom(InstaSeisBenchmark):
    def setup(self):
        self.db = InstaSeisDB(self.path, read_on_demand=False,
                              buffer_size_in_mb=250)

    def iterate(self):
        # Random points on a sphere.
        lat = np.rad2deg(np.arcsin(2 * random.random() - 1))
        lng = random.random() * 360.0 - 180.0
        rec = Receiver(latitude=lat, longitude=lng)

        lat = np.rad2deg(np.arcsin(2 * random.random() - 1))
        lng = random.random() * 360.0 - 180.0
        depth_in_m = random.random() * 300000
        src = Source(latitude=lat, longitude=lng, depth_in_m=depth_in_m)

        self.db.get_seismograms(source=src, receiver=rec)

    @property
    def description(self):
        return "Buffered, random src and receiver"


class UnbufferedFullyRandom(InstaSeisBenchmark):
    def setup(self):
        self.db = InstaSeisDB(self.path, read_on_demand=False,
                              buffer_size_in_mb=0)

    def iterate(self):
        # Random points on a sphere.
        lat = np.rad2deg(np.arcsin(2 * random.random() - 1))
        lng = random.random() * 360.0 - 180.0
        rec = Receiver(latitude=lat, longitude=lng)

        lat = np.rad2deg(np.arcsin(2 * random.random() - 1))
        lng = random.random() * 360.0 - 180.0
        depth_in_m = random.random() * 300000
        src = Source(latitude=lat, longitude=lng, depth_in_m=depth_in_m)

        self.db.get_seismograms(source=src, receiver=rec)

    @property
    def description(self):
        return "Unbuffered, random src and receiver"


class UnbufferedFullyRandomReadOnDemandTrue(InstaSeisBenchmark):
    def setup(self):
        self.db = InstaSeisDB(self.path, read_on_demand=True,
                              buffer_size_in_mb=0)

    def iterate(self):
        # Random points on a sphere.
        lat = np.rad2deg(np.arcsin(2 * random.random() - 1))
        lng = random.random() * 360.0 - 180.0
        rec = Receiver(latitude=lat, longitude=lng)

        lat = np.rad2deg(np.arcsin(2 * random.random() - 1))
        lng = random.random() * 360.0 - 180.0
        depth_in_m = random.random() * 300000
        src = Source(latitude=lat, longitude=lng, depth_in_m=depth_in_m)

        self.db.get_seismograms(source=src, receiver=rec)

    @property
    def description(self):
        return "Unbuffered, random src and receiver, read_on_demand=True, " \
               "worst case!"

parser = argparse.ArgumentParser(
    prog="python -m instaseis.benchmark",
    description='Benchmark InstaSeis.')
parser.add_argument('folder', type=str,
                    help="path to AxiSEM Green's function database")
parser.add_argument('--time', type=float, default=10.0,
                    help='time spent per benchmark in seconds')
args = parser.parse_args()
path = os.path.abspath(args.folder)

db = InstaSeisDB(path, read_on_demand=True, buffer_size_in_mb=0)
if not db.reciprocal:
    print("Benchmark currently only works with a reciprocal database.")
    sys.exit(1)
print(db)


# Recursively get all subclasses of the benchmark class.
def get_subclasses(cls):
    subclasses = []

    sub = cls.__subclasses__()
    subclasses.extend(sub)

    for subclass in sub:
        subclasses.extend(get_subclasses(subclass))

    return subclasses

benchmarks = [i(path, args.time) for i in get_subclasses(InstaSeisBenchmark)]
benchmarks.sort(key=lambda x: x.description)

print(80 * "=")
print(80 * "=")
print("\nFound %i benchmark(s)\n" % len(benchmarks))

for benchmark in benchmarks:
    print(80 * "=")
    print(benchmark.description)
    benchmark.run()
