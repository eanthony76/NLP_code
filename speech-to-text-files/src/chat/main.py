"""
This module provides functionalities to execute two external Python scripts
(`listener.py` and `watcher.py`) concurrently using the `multiprocessing` module.

Functions:
-----------
run_script1():
    Execute `listener.py` using the system's Python interpreter.

run_script2():
    Execute `watcher.py` using the system's Python interpreter.

Usage:
-------
To run both scripts simultaneously:
    python <name_of_this_module>.py

Note:
-----
It's important that both `listener.py` and `watcher.py` are present in the same
directory as this script or in the system's PATH. If they aren't, execution
will fail. Also, the scripts will run indefinitely until terminated manually
or until they finish their respective executions.

Requirements:
-------------
- Python (with `multiprocessing` and `os` modules available)
- `listener.py` and `watcher.py` scripts
"""

import multiprocessing
import os


def run_script1():
    """Execute the `listener.py` script using the system's Python interpreter."""
    os.system("python listener.py")


def run_script2():
    """Execute the `watcher.py` script using the system's Python interpreter."""
    os.system("python watcher.py")


if __name__ == "__main__":
    # Start both scripts simultaneously
    p1 = multiprocessing.Process(target=run_script1)
    p2 = multiprocessing.Process(target=run_script2)

    p1.start()
    p2.start()

    # Allow processes to run indefinitely
    p1.join()
    p2.join()
