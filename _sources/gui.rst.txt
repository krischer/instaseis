=============
Instaseis GUI
=============


Recompile UI file
=================

To recompile the UI file, you can use the `pyside6-uic` command line tool. This
tool converts `.ui` files created with Qt Designer into Python code that can be
used in your application.

.. code-block:: bash

    $ pyside6-uic src/instaseis/gui/qt_window.ui -o src/instaseis/gui/qt_window.py
