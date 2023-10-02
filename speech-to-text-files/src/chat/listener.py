"""Monitor and Process Audio Files."""
import json
import os
import shutil
import time

from speaker import speak
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
from datasets import Audio, Dataset
from funcs import remove_punctuation
from transformers import pipeline
from modeler import chat_model
from hit_speak_api import speak_api

def load_configs():
    """Load phonetic alphabet and nine line configurations."""
    with open("config.json", "r", encoding="utf-8") as file_in:
        _config = json.load(file_in)
    return _config


class FileHandler(FileSystemEventHandler):
    """Handles file creation events and processes the created files."""

    def __init__(
        self,
        _source_dir,
        _dest_dir,
        _pipe,
        _pipe2,
        _owd,
    ):
        self.source_dir = _source_dir
        self.dest_dir = _dest_dir
        self.pipe = _pipe
        self.pipe2 = _pipe2
        self.owd = _owd

    def on_created(self, event):
        """When a new file is created in the monitored directory,
        process and move it."""
        if event.is_directory:
            return
        self.process_and_move(event.src_path)

    def process_and_move(self, file_path):
        """Process a file and then move it to the processed directory."""
        self.process_file(file_path)
        destination_path = os.path.join(self.dest_dir, os.path.basename(file_path))
        shutil.move(file_path, destination_path)
        print(f"Moved {file_path} to {destination_path}")

    # pylint: disable=too-many-locals
    def process_file(self, file_path):
        """Extracts information from the audio file"""
        dataset = Dataset.from_dict({"audio": os.listdir(self.source_dir)}).cast_column(
            "audio", Audio(sampling_rate=16_000)
        )
        os.chdir(self.source_dir)
        for samples in dataset:
            if samples["audio"]["path"].endswith("test.wav"):
                print(f'Skipping {samples["audio"]["path"]}')
                continue
            try:
                prediction = self.pipe(samples["audio"].copy(), batch_size=8)["text"]
                print(prediction)
                prediction = remove_punctuation(prediction)
                # text = self.pipe2(prediction)
                # text = remove_punctuation(str(text)).replace("generatedtext", "")
                text = chat_model(prediction)
                print(text)
                print("Moving to speech model")
                start_time = time.time()
                #speak(text)
                speak_api('158.101.118.0:5000', text)
                end_time = time.time()
                print(f"Elapsed time to speak: {end_time - start_time}")
                print("response created")
            except IsADirectoryError as exception:
                print(f"Error while processing: {exception}")

        os.chdir(self.owd)
        print(f"Processed file: {file_path}")


def monitor_directory(
    _source_directory,
    _destination_directory,
    _pipe,
    _pipe2,
    _owd,
):
    """Monitors the source directory for any new files and processes them."""
    event_handler = FileHandler(
        _source_directory,
        _destination_directory,
        _pipe,
        _pipe2,
        _owd,
    )
    observer = Observer()
    observer.schedule(event_handler, path=_source_directory, recursive=False)
    print(f"Starting to monitor {_source_directory}...")
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    SOURCE_DIRECTORY = "audio_files"
    DESTINATION_DIRECTORY = os.path.join(os.getcwd(), "processed")
    owd = os.getcwd()
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small",
        chunk_length_s=30,
        device=0,
    )

    pipe2 = pipeline("text2text-generation", model="google/flan-t5-small", device=0)

    monitor_directory(
        SOURCE_DIRECTORY,
        DESTINATION_DIRECTORY,
        pipe,
        pipe2,
        owd,
    )
