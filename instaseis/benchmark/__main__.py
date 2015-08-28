#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Benchmarks for Instaseis.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
:license:
    GNU Lesser General Public License, Version 3 [non-commercial/academic use]
    (http://www.gnu.org/copyleft/lgpl.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from future.utils import with_metaclass

from abc import ABCMeta, abstractmethod, abstractproperty
import argparse
import colorama
import fnmatch
import numpy as np
import obspy
import os
import random
import subprocess
import sys
import time
import timeit

from instaseis import open_db, Source, Receiver

# Write interval.
WRITE_INTERVAL = 0.05


def plot_gnuplot(times):
    try:
        gnuplot = subprocess.Popen(["gnuplot"],
                                   stdin=subprocess.PIPE)
        gnuplot.stdin.write("set term dumb 79 15\n".encode())
        gnuplot.stdin.write("set xlabel 'Seismogram Number'\n".encode())
        gnuplot.stdin.write("plot '-' using 1:2 title 'time per sm' with "
                            "linespoints \n".encode())
        for i, j in zip(np.arange(len(times)), times):
            gnuplot.stdin.write(("%f %f\n" % (i, j)).encode())
        gnuplot.stdin.write("e\n".encode())
        gnuplot.stdin.flush()
        sys.stdout.flush()
    except OSError:
        print("Could not plot graph. No gnuplot installed?")


class InstaseisBenchmark(with_metaclass(ABCMeta)):

    def __init__(self, path, time_per_benchmark, save_output=False,
                 seed=None, count=None):
        self.path = path
        self.time_per_benchmark = time_per_benchmark
        self.save_output = save_output
        self.seed = seed
        self.count = count

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
        # Set seeds to be able to reproduce results.
        if self.seed is not None:
            print("\tSetting random seed to %i" % self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)
        a = timeit.default_timer()
        self.setup()
        b = timeit.default_timer()
        print("\tTime for initialization: %s sec" % (b - a))

        starttime = timeit.default_timer()
        endtime = starttime + self.time_per_benchmark
        all_times = []

        last_write_time = starttime
        latest_times = []
        count = 0

        print("\tStarting...", end="\r")
        t = starttime
        while ((self.count is not None and count < self.count) or
                (self.count is None and t < endtime)):
            count += 1
            s = timeit.default_timer()
            self.iterate()
            t = timeit.default_timer()
            all_times.append(t - s)

            # Pretty and immediate output.
            latest_times.append(t - s)

            if t >= (last_write_time + WRITE_INTERVAL):
                cumtime = sum(latest_times)
                speed = len(latest_times) / cumtime
                if self.count is None:
                    print("\tseismograms/sec: {0:>8.2f}, remaining time: "
                          "{1:>2.1f} sec".format(speed, endtime - t),
                          end="\r")
                else:
                    print("\tseismograms/sec: {0:>8.2f}, remaining runs: "
                          "{1:>6d}".format(speed, self.count - count),
                          end="\r")
                sys.stdout.flush()
                latest_times = []
                last_write_time = t
        print(79 * " ", end="\r")

        all_times = np.array(all_times, dtype="float64")
        cumtime = sum(all_times)
        count = len(all_times)
        print("\t%i seismograms in %.2f sec" % (count, cumtime))
        print("\t%g sec/seismogram" % all_times.mean())
        print("\t%g seismograms/sec" % (count / cumtime))
        for p in [0, 10, 25, 50, 75, 90, 100]:
            print("\t {0:>3}th percentile: {1} sec".format(
                p, np.percentile(all_times, p)))
        sys.stdout.flush()
        plot_gnuplot(all_times)
        time.sleep(0.1)
        if self.save_output:
            folder = "benchmark_results"
            if not os.path.exists(folder):
                os.makedirs(folder)
            _i = 0
            while True:
                _i += 1
                filename = os.path.join(folder, "%s_%04i.txt" % (
                    self.__class__.__name__, _i))
                if not os.path.exists(filename):
                    break
            np.savetxt(filename, all_times,
                       header="time per seismogram [run at %s]" % (
                           obspy.UTCDateTime()))


class BufferedFixedSrcRecRoDOffSeismogramGeneration(InstaseisBenchmark):
    def setup(self):
        self.db = open_db(self.path, read_on_demand=False,
                          buffer_size_in_mb=250)

    def iterate(self):
        src = Source(latitude=10, longitude=10)
        rec = Receiver(latitude=20, longitude=20)
        self.db.get_seismograms(source=src, receiver=rec)

    @property
    def description(self):
        return "Buffered, fixed source and receiver, " \
               "read_on_demand=False"


class BufferedFixedSrcRecRoDOffSeismogramGenerationNoObsPy(InstaseisBenchmark):
    def setup(self):
        self.db = open_db(self.path, read_on_demand=False,
                          buffer_size_in_mb=250)

    def iterate(self):
        src = Source(latitude=10, longitude=10)
        rec = Receiver(latitude=20, longitude=20)
        self.db.get_seismograms(source=src, receiver=rec,
                                return_obspy_stream=False)

    @property
    def description(self):
        return "Buffered, fixed source and receiver, " \
               "read_on_demand=False, no ObsPy output, best case!"


class UnbufferedFixedSrcRecRoDOffSeismogramGenerationNoObsPy(
        InstaseisBenchmark):
    def setup(self):
        self.db = open_db(self.path, read_on_demand=False,
                          buffer_size_in_mb=0)

    def iterate(self):
        src = Source(latitude=10, longitude=10)
        rec = Receiver(latitude=20, longitude=20)
        self.db.get_seismograms(source=src, receiver=rec)

    @property
    def description(self):
        return "Unbuffered, fixed source and receiver, " \
               "read_on_demand=False"


class BufferedFixedSrcRecRoDOnSeismogramGeneration(InstaseisBenchmark):
    def setup(self):
        self.db = open_db(self.path, read_on_demand=True,
                          buffer_size_in_mb=250)

    def iterate(self):
        src = Source(latitude=10, longitude=10)
        rec = Receiver(latitude=20, longitude=20)
        self.db.get_seismograms(source=src, receiver=rec)

    @property
    def description(self):
        return "Buffered, fixed source and receiver, " \
               "read_on_demand=True"


class Buffered2DegreeLatLngDepthScatter(InstaseisBenchmark):
    def setup(self):
        self.db = open_db(self.path, read_on_demand=False,
                          buffer_size_in_mb=250)
        self.max_depth = self.db.info.max_radius - self.db.info.min_radius

    def iterate(self):
        rec = Receiver(latitude=20, longitude=20)
        lat = random.random() * 2
        lat += 44
        lng = random.random() * 2
        lng += 44
        depth_in_m = random.random() * min(200000, self.max_depth)
        src = Source(latitude=lat, longitude=lng, depth_in_m=depth_in_m)
        self.db.get_seismograms(source=src, receiver=rec)

    @property
    def description(self):
        return "Buffered, 2 Degree/200 km source position scatter"


class BufferedHalfDegreeLatLngDepthScatter(InstaseisBenchmark):
    def setup(self):
        self.db = open_db(self.path, read_on_demand=False,
                          buffer_size_in_mb=250)

    def iterate(self):
        rec = Receiver(latitude=20, longitude=20)
        lat = random.random() * 0.5
        lat += 44
        lng = random.random() * 0.5
        lng += 44
        depth_in_m = random.random() * 50000
        src = Source(latitude=lat, longitude=lng, depth_in_m=depth_in_m)
        self.db.get_seismograms(source=src, receiver=rec)

    @property
    def description(self):
        return "Buffered, 0.5 Degree/50 km (depth) source position scatter"


class BufferedFullyRandom(InstaseisBenchmark):
    def setup(self):
        self.db = open_db(self.path, read_on_demand=False,
                          buffer_size_in_mb=250)
        self.max_depth = self.db.info.max_radius - self.db.info.min_radius

    def iterate(self):
        # Random points on a sphere.
        lat = np.rad2deg(np.arcsin(2 * random.random() - 1))
        lng = random.random() * 360.0 - 180.0
        rec = Receiver(latitude=lat, longitude=lng)

        lat = np.rad2deg(np.arcsin(2 * random.random() - 1))
        lng = random.random() * 360.0 - 180.0
        depth_in_m = random.random() * self.max_depth
        src = Source(latitude=lat, longitude=lng, depth_in_m=depth_in_m)

        self.db.get_seismograms(source=src, receiver=rec)

    @property
    def description(self):
        return "Buffered, random src and receiver"


class UnbufferedFullyRandom(InstaseisBenchmark):
    def setup(self):
        self.db = open_db(self.path, read_on_demand=False,
                          buffer_size_in_mb=0)
        self.max_depth = self.db.info.max_radius - self.db.info.min_radius

    def iterate(self):
        # Random points on a sphere.
        lat = np.rad2deg(np.arcsin(2 * random.random() - 1))
        lng = random.random() * 360.0 - 180.0
        rec = Receiver(latitude=lat, longitude=lng)

        lat = np.rad2deg(np.arcsin(2 * random.random() - 1))
        lng = random.random() * 360.0 - 180.0
        depth_in_m = random.random() * self.max_depth
        src = Source(latitude=lat, longitude=lng, depth_in_m=depth_in_m)

        self.db.get_seismograms(source=src, receiver=rec)

    @property
    def description(self):
        return "Unbuffered, random src and receiver"


class UnbufferedAndRandomReadOnDemandTrue(InstaseisBenchmark):
    def setup(self):
        self.db = open_db(self.path, read_on_demand=True,
                          buffer_size_in_mb=0)
        self.max_depth = self.db.info.max_radius - self.db.info.min_radius

    def iterate(self):
        # Random points on a sphere.
        lat = np.rad2deg(np.arcsin(2 * random.random() - 1))
        lng = random.random() * 360.0 - 180.0
        rec = Receiver(latitude=lat, longitude=lng)

        lat = np.rad2deg(np.arcsin(2 * random.random() - 1))
        lng = random.random() * 360.0 - 180.0
        depth_in_m = random.random() * self.max_depth
        src = Source(latitude=lat, longitude=lng, depth_in_m=depth_in_m)

        self.db.get_seismograms(source=src, receiver=rec)

    @property
    def description(self):
        return "Unbuffered, random src and receiver, read_on_demand=True, " \
               "worst case!"


class FiniteSourceEmulation(InstaseisBenchmark):
    def setup(self):
        self.db = open_db(self.path, read_on_demand=False,
                          buffer_size_in_mb=250)
        self.counter = 0
        self.current_depth_counter = 0
        # Depth increases in 1 km steps up to a depth of 25 km.
        self.depths = range(0, 25001, 1000)
        self.depth_count = len(self.depths)

        # Fix Receiver
        self.rec = Receiver(latitude=45.0, longitude=45.0)

    def iterate(self):
        if self.current_depth_counter >= self.depth_count:
            self.counter += 1
            self.current_depth_counter = 0

        # Fix latitude.
        lat = 0.0
        # Longitude values increase in 1 km steps.
        lng = self.counter * 0.01
        src = Source(latitude=lat, longitude=lng,
                     depth_in_m=self.depths[self.current_depth_counter])

        self.db.get_seismograms(source=src, receiver=self.rec)
        self.current_depth_counter += 1

    @property
    def description(self):
        return "Finite source emulation."


parser = argparse.ArgumentParser(
    prog="python -m instaseis.benchmark",
    description='Benchmark Instaseis.')
parser.add_argument('folder', type=str,
                    help="path to AxiSEM Green's function database")
parser.add_argument('--time', type=float, default=10.0,
                    help='time spent per benchmark in seconds')
parser.add_argument('--pattern', type=str,
                    help='UNIX style patterns to only run certain benchmarks')
parser.add_argument('--seed', type=int,
                    help='Seed used for the random number generation')
parser.add_argument('--count', type=int,
                    help='Number of seismograms to be calculated for each '
                         'benchmark. Overwrites any time limitations if '
                         'given.')
parser.add_argument('--save', action="store_true",
                    help='save output to txt file')
args = parser.parse_args()
path = os.path.abspath(args.folder) if "://" not in args.folder \
    else args.folder

print(colorama.Fore.GREEN + 79 * "=" + "\nInstaseis Benchmark Suite\n")
print("It enables to gauge the speed of Instaseis for a certain DB.")
print(79 * "=" + colorama.Fore.RESET)
print(colorama.Fore.RED + "\nIt does not deal with OS level caches! So "
      "interpret the results accordingly!\n" + colorama.Fore.RESET)

db = open_db(path, read_on_demand=True, buffer_size_in_mb=0)
if not db.info.is_reciprocal:
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

benchmarks = [i(path, args.time, args.save, args.seed, args.count) for i in
              get_subclasses(InstaseisBenchmark)]
benchmarks.sort(key=lambda x: x.description)

print(79 * "=")
print("Discovered %i benchmark(s)" % len(benchmarks))

if args.pattern is not None:
    pattern = "*%s*" % args.pattern.lower()
    benchmarks = [_i for _i in benchmarks if
                  fnmatch.fnmatch(_i.__class__.__name__.lower(), pattern)]
    print("Pattern matching retained %i benchmark(s)" % len(benchmarks))

print(79 * "=")

for benchmark in benchmarks:
    print("\n")
    print(colorama.Fore.YELLOW + 79 * "=")
    print(79 * "=" + colorama.Fore.RESET)
    print("\n")
    print(colorama.Fore.BLUE +
          benchmark.__class__.__name__ + ": " + benchmark.description +
          colorama.Fore.RESET, end="\n\n")
    benchmark.run()
