#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The MIT License (MIT)

Copyright (c) 2015 Lion Krischer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import json
import obspy
import os


CACHE = {}


class EventNotFoundError(Exception):
    pass


def create_event_json_file(output_filename):
    """
    Download the GCMT catalog up to 2013 and store everything in a JSON file
    which can be used for the later stage.

    If the file already exist this function is a no-op.
    """
    if os.path.exists(output_filename):
        print("File '%s' already exists. It will not be recreated." %
              output_filename)
        return
    print("Downloading GCMT catalog up to 2013...")
    cat = obspy.readEvents("http://www.ldeo.columbia.edu/~gcmt/projects/CMT/"
                           "catalog/jan76_dec13.ndk.gz")
    events = {}
    for event in cat:
        # This only works due to the way ObsPy parses NDK files.
        gcmt_id = [_i.text for _i in event.event_descriptions
                   if _i.type == 'earthquake name'][0]

        # Get the centroid origin.
        origin = [_i for _i in event.origins
                  if _i.origin_type == "centroid"][0]
        mt = event.focal_mechanisms[0].moment_tensor.tensor

        events[gcmt_id] = {
            "latitude": float(origin.latitude),
            "longitude": float(origin.longitude),
            "origin_time": str(origin.time),
            "m_rr": mt.m_rr,
            "m_tt": mt.m_tt,
            "m_pp": mt.m_pp,
            "m_rt": mt.m_rt,
            "m_rp": mt.m_rp,
            "m_tp": mt.m_tp}

    with open(output_filename, "wt") as fh:
        json.dump(events, fh)


def get_event_information(filename, event_id):
    """
    Return the event information for the given event id as a dictionary.
    """
    filename = os.path.abspath(filename)
    if filename not in CACHE:
        with open(filename, "rt") as fh:
            CACHE[filename] = json.load(fh)
    events = CACHE[filename]

    if event_id not in events:
        raise EventNotFoundError

    event = events[event_id]
    event["origin_time"] = obspy.UTCDateTime(event["origin_time"])
    return event


if __name__ == "__main__":
    FILENAME = "event_db.json"
    create_event_json_file(FILENAME)

    import pprint
    pprint.pprint(get_event_information(FILENAME, "C201308081707A"))
    pprint.pprint(get_event_information(FILENAME, "B071791B"))
