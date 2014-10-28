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
import sys
import timeit

from instaseis import InstaSeisDB, Source, Receiver

# Time spent per benchmark in seconds.
TIME_PER_BENCHMARK = 5.0

# Write interval.
WRITE_INTERVAL = 0.05


class InstaSeisBenchmark(object):
    __metaclass__ = ABCMeta

    def __init__(self, path):
        self.path = path

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
        endtime = starttime + TIME_PER_BENCHMARK
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


class FullyBufferedFixedSrcRecRoDOffSeismogramGeneration(InstaSeisBenchmark):
    def setup(self):
        self.db = InstaSeisDB(self.path, read_on_demand=False,
                              buffer_size_in_mb=250)

    def iterate(self):
        src = Source(latitude=10, longitude=10)
        rec = Receiver(latitude=20, longitude=20)
        self.db.get_seismograms(source=src, receiver=rec)

    @property
    def description(self):
        return "Fully buffered, fixed source and receiver, " \
               "read_on_demand=False"


class FullyBufferedFixedSrcRecRoDOnSeismogramGeneration(InstaSeisBenchmark):
    def setup(self):
        self.db = InstaSeisDB(self.path, read_on_demand=True,
                              buffer_size_in_mb=250)

    def iterate(self):
        src = Source(latitude=10, longitude=10)
        rec = Receiver(latitude=20, longitude=20)
        self.db.get_seismograms(source=src, receiver=rec)

    @property
    def description(self):
        return "Fully buffered, fixed source and receiver, " \
               "read_on_demand=True"


parser = argparse.ArgumentParser(
    prog="python -m instaseis.benchmark",
    description='Benchmark InstaSeis.')
parser.add_argument('folder', type=str,
                    help="path to AxiSEM Green's function database")
args = parser.parse_args()
path = os.path.abspath(args.folder)

db = InstaSeisDB(path, read_on_demand=True, buffer_size_in_mb=0)
print(db)


# Recursively get all subclasses of the benchmark class.
def get_subclasses(cls):
    subclasses = []

    sub = cls.__subclasses__()
    subclasses.extend(sub)

    for subclass in sub:
        subclasses.extend(get_subclasses(subclass))

    return subclasses

benchmarks = [i(path) for i in get_subclasses(InstaSeisBenchmark)]
benchmarks.sort(key=lambda x: x.description)

print(80 * "=")
print(80 * "=")
print("\nFound %i benchmark(s)\n" % len(benchmarks))

for benchmark in benchmarks:
    print(80 * "=")
    print(benchmark.description)
    benchmark.run()