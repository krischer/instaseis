#!/usr/bin/env python

if __name__ == "__main__":
    import inspect
    import os
    import pytest
    import sys
    PATH = os.path.dirname(os.path.dirname(os.path.abspath(
        inspect.getfile(inspect.currentframe()))))
    sys.exit(pytest.main(PATH))
