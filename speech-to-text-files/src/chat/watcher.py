"""
This module provides utilities to monitor a directory
for the creation of new HTML files and
automatically open them in a web browser when detected.

The Watcher class is responsible for setting up and
managing an Observer from the watchdog library
to monitor the desired directory. It utilizes the
Handler class to define the actions to be taken
on the detected filesystem events.

Classes:
    Watcher: A directory monitor that checks
    for new files and reacts based on the file type.
    Handler: A custom event handler to define
    responses to detected filesystem events.

Usage:
    To use this utility, simply run the script.
    By default, it monitors the 'jsons/maps' directory.
    You can change the `DIRECTORY_TO_WATCH` attribute of
    the `Watcher` class to the desired directory path.
"""

import time
import os
from subprocess import call

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# pylint: disable=too-few-public-methods
class Watcher:
    """Watches a directory for new file creations
    and triggers actions based on the file type.

    Attributes:
        DIRECTORY_TO_WATCH (str): The directory path to be monitored.

    Methods:
        run: Initiates the monitoring of the specified directory.

    Note:
        To change the directory to be monitored, modify
        the `DIRECTORY_TO_WATCH` attribute.
    """

    DIRECTORY_TO_WATCH = "processed"  # Change this to your folder path

    def __init__(self):
        self.observer = Observer()

    def run(self):
        """Starts the directory monitoring process.

        This method sets up an event handler for the directory,
        starts the observer, and keeps it running.
        """
        event_handler = Handler()
        self.observer.schedule(event_handler, self.DIRECTORY_TO_WATCH, recursive=True)
        self.observer.start()
        try:
            while True:
                time.sleep(5)
        except KeyboardInterrupt:
            self.observer.stop()
            print("Observer stopped")

        self.observer.join()


class Handler(FileSystemEventHandler):
    """Handles filesystem events detected by the Watcher."""

    @staticmethod
    # pylint: disable=arguments-differ
    def on_any_event(event):
        """Opens newly created HTML files in a web browser.

        Args:
            event: The detected file system event.
        """
        if event.is_directory:
            return None

        if event.event_type == "created":
            # Take action if a new file is created
            if event.src_path.endswith("test.wav"):
                call(["aplay", f"{os.path.realpath(event.src_path)}"])
                # webbrowser.open("file://" + os.path.realpath(event.src_path))
                return "Assistant response:"

        return None


if __name__ == "__main__":
    w = Watcher()
    w.run()
