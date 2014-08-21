import ctypes as C
import inspect
import os


LIB_DIR = os.path.join(os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe()))), "lib")

cache = []


def load_lib():
    if cache:
        return cache[0]
    else:
        filename = os.path.join(LIB_DIR, "axisem_helpers.so")
        lib = C.CDLL(filename)
        cache.append(lib)
        return lib
