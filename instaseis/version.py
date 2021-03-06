# -*- coding: utf-8 -*-
# Author: Douglas Creager <dcreager@dcreager.net>
# This file is placed into the public domain.
#
# Modified by the ObsPy project and instaseis.

# Calculates the current version number.  If possible, this is the
# output of “git describe”, modified to conform to the versioning
# scheme that setuptools uses.  If “git describe” returns an error
# (most likely because we're in an unpacked copy of a release tarball,
# rather than in a git working copy), then we fall back on reading the
# contents of the RELEASE-VERSION file.
#
# To use this script, simply import it your setup.py file, and use the
# results of get_git_version() as your package version:
#
# from version import *
#
# setup(
#     version=get_git_version(),
#     .
#     .
#     .
# )
#
# This will automatically update the RELEASE-VERSION file, if
# necessary.  Note that the RELEASE-VERSION file should *not* be
# checked into git; please add it to your top-level .gitignore file.
#
# You'll probably want to distribute the RELEASE-VERSION file in your
# sdist tarballs; to do this, just create a MANIFEST.in file that
# contains the following line:
#
#   include RELEASE-VERSION
# NO IMPORTS FROM INSTASEIS OR FUTURE IN THIS FILE! (file gets used at
# installation time)


import io
import os
import inspect
from subprocess import Popen, PIPE

__all__ = "get_git_version"


script_dir = os.path.abspath(
    os.path.dirname(inspect.getfile(inspect.currentframe()))
)
INSTASEIS_ROOT = os.path.abspath(os.path.join(script_dir, os.pardir))
VERSION_FILE = os.path.join(script_dir, "RELEASE-VERSION")


def call_git_describe(abbrev=4):  # pragma: no cover
    try:
        p = Popen(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=INSTASEIS_ROOT,
            stdout=PIPE,
            stderr=PIPE,
        )
        p.stderr.close()
        path = p.stdout.readline().decode().strip()
        p.stdout.close()
    except Exception:
        return None
    if os.path.normpath(path) != INSTASEIS_ROOT:
        return None
    try:
        p = Popen(
            [
                "git",
                "describe",
                "--dirty",
                "--abbrev=%d" % abbrev,
                "--always",
                "--tags",
            ],
            cwd=INSTASEIS_ROOT,
            stdout=PIPE,
            stderr=PIPE,
        )

        p.stderr.close()
        line = p.stdout.readline().decode()
        p.stdout.close()

        if "-" not in line and "." not in line:
            line = "0.0.0-g%s" % line
        return line.strip()
    except Exception:
        return None


def read_release_version():  # pragma: no cover
    try:
        with io.open(VERSION_FILE, "rt") as fh:
            version = fh.readline()
        return version.strip()
    except Exception:
        return None


def write_release_version(version):  # pragma: no cover
    with io.open(VERSION_FILE, "wb") as fh:
        fh.write(("%s\n" % version).encode("ascii", "strict"))


def get_git_version(abbrev=4):  # pragma: no cover
    # Read in the version that's currently in RELEASE-VERSION.
    release_version = read_release_version()

    # First try to get the current version using “git describe”.
    version = call_git_describe(abbrev)

    # If that doesn't work, fall back on the value that's in
    # RELEASE-VERSION.
    if version is None:
        version = release_version

    # If we still don't have anything, that's an error.
    if version is None:
        return "0.0.0-tar/zipball"

    # If the current version is different from what's in the
    # RELEASE-VERSION file, update the file to be current.
    if version != release_version:
        write_release_version(version)

    # Finally, return the current version.
    return version


if __name__ == "__main__":
    print(get_git_version())
