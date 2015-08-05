# Advanced Instaseis Server Configuration Example

This folder contains an example for an advanced Instaseis server 
configuration. Feel free to built upon it to suit your own purposes.

This example needs one more thing before it can work:

* A file ``station_list.txt`` in this folder which has to be the text output of
  any fdsnws station service at the station level. This is the source for the 
  station coordinates. If a certain network, station combination is 
  encountered multiple times, the last one in the file will be used.
  
  
The event information will be downloaded on the first run from the GCMT web 
page.


To start the server, simply change into this directory and

```bash
$ python launch_server.py --port=8123 --log-level=DEBUG /path/to/db
```
