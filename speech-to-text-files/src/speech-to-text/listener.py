"""Monitor and Process Audio Files."""

import json
import os
import shutil
import time
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from datasets import Audio, Dataset
from funcs import (
    collect_info,
    plot_mgrs_on_map,
    remove_punctuation,
    remove_spaces_from_index,
    replace_value,
)
from mgrs import core
from transformers import pipeline


def load_configs():
    """Load phonetic alphabet and nine line configurations."""
    with open("phonetic_alphabet.json", "r", encoding="utf-8") as alphabet_f:
        _phonetic_alphabet = json.load(alphabet_f)
    with open("nine_line_dict.json", "r", encoding="utf-8") as nine_line_f:
        _nine_line_config = json.load(nine_line_f)

    return _phonetic_alphabet, _nine_line_config


def convert_phrases(_nine_line_config):
    """Convert nine line dictionary to a list of converted phrases."""
    phrases = [
        value["phrases"]
        for key, value in nine_line_config.items()
        if "phrases" in value
    ]
    return [tuple(item.split()) for sub_list in phrases for item in sub_list]


class FileHandler(FileSystemEventHandler):
    """Handles file creation events and processes the created files."""

    def __init__(
        self,
        _source_dir,
        _dest_dir,
        _pipe,
        _reverse_alphabet,
        _converted_phrases,
        _nine_line_config,
        _owd,
    ):
        self.source_dir = _source_dir
        self.dest_dir = _dest_dir
        self.pipe = _pipe
        self.reverse_alphabet = _reverse_alphabet
        self.converted_phrases = _converted_phrases
        self.nine_line_config = _nine_line_config
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
        """Extracts information from the audio file and
        saves a JSON and possibly a map."""
        name = str(os.listdir(self.source_dir)).replace(".wav", "")
        dataset = Dataset.from_dict({"audio": os.listdir(self.source_dir)}).cast_column(
            "audio", Audio(sampling_rate=16_000)
        )
        os.chdir(self.source_dir)
        for samples in dataset:
            try:
                converted_results = []
                prediction = self.pipe(samples["audio"].copy(), batch_size=8)["text"]
                prediction = remove_punctuation(prediction)
                print(prediction)
                info = collect_info(prediction, self.converted_phrases)
                for inf in info:
                    converted_info = {}
                    for key, values in inf.items():
                        converted_values = [
                            self.reverse_alphabet.get(val, val) for val in values
                        ]
                        converted_info[key] = converted_values
                        converted_info = {
                            " ".join(
                                remove_punctuation(item) for item in key
                            ): " ".join(remove_punctuation(value) for value in val_list)
                            for key, val_list in converted_info.items()
                        }
                    converted_results.append(converted_info)
                to_json = []
                for result in converted_results:
                    new_dict = {}
                    for key, value in result.items():
                        new_dict[key] = replace_value(key, value, self.nine_line_config)
                    to_json.append(new_dict)
                for idx in range(0, 2):
                    to_json = remove_spaces_from_index(to_json, idx)
                with open(
                    f"{self.owd}/jsons/{name}.json", "w", encoding="utf-8"
                ) as file:
                    json.dump(to_json, file, indent=4)
                    print(f"Saved JSON {name}")
                for key, value in to_json[0].items():
                    try:
                        map_obj = plot_mgrs_on_map(value)
                        map_obj.save(f"{self.owd}/jsons/maps/map_{name}.html")
                        print("Saved map")
                    except core.MGRSError as mgrs_error:
                        print(
                            f"Not a valid MGRS coordinate. Unable to save map {name}: {mgrs_error}"
                        )
            except IsADirectoryError as exception:
                print(f"Error while processing: {exception}")

        os.chdir(self.owd)
        print(f"Processed file: {file_path}")


def monitor_directory(
    _source_directory,
    _destination_directory,
    _pipe,
    _reverse_alphabet,
    _converted_phrases_list,
    _nine_line_config,
    _owd,
):
    """Monitors the source directory for any new files and processes them."""
    event_handler = FileHandler(
        _source_directory,
        _destination_directory,
        _pipe,
        _reverse_alphabet,
        _converted_phrases_list,
        _nine_line_config,
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
    phonetic_alphabet, nine_line_config = load_configs()
    converted_phrases_list = convert_phrases(nine_line_config)
    reverse_alphabet = {value.lower(): key for key, value in phonetic_alphabet.items()}

    monitor_directory(
        SOURCE_DIRECTORY,
        DESTINATION_DIRECTORY,
        pipe,
        reverse_alphabet,
        converted_phrases_list,
        nine_line_config,
        owd,
    )
