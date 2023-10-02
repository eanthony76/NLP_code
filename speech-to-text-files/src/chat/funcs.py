"""Utility module for string and MGRS handling."""
import os
import string
from typing import List, Tuple, Dict, Union, Any
import mgrs
import folium


def word_tuples(sentence: str) -> List[Tuple[str, str]]:
    """Convert a sentence into consecutive word tuples.

    Args:
        sentence (str): Input sentence.

    Returns:
        List[Tuple[str, str]]: List of consecutive word tuples.
    """
    words = sentence.lower().split()
    return [(words[i], words[i + 1]) for i in range(len(words) - 1)]


def collect_info(
    sentence: str, phrases: List[Tuple[str, str]]
) -> List[Dict[Tuple[str, str], List[str]]]:
    """Collect information based on provided phrases.

    Args:
        sentence (str): Input sentence.
        phrases (List[Tuple[str, str]]): List of word pairs.

    Returns:
        List[Dict[Tuple[str, str], List[str]]]: Information collected.
    """
    words = sentence.lower().split()
    collected = []
    i = 0
    while i < len(words) - 1:
        current_phrase = (words[i], words[i + 1])
        if current_phrase in phrases:
            temp_collected = []
            j = i + 2
            while j < len(words) - 1:
                next_phrase = (words[j], words[j + 1])
                if next_phrase in phrases:
                    break
                temp_collected.append(words[j])
                j += 1
            collected.append({current_phrase: temp_collected})
            i = j
        else:
            i += 1
    return collected


def remove_punctuation(input_string: str) -> str:
    """Remove punctuation from the input string.

    Args:
        input_string (str): Input string.

    Returns:
        str: String without punctuation.
    """
    translator = str.maketrans("", "", string.punctuation)
    return input_string.translate(translator)


def find_key_by_phrase(dict_key: str, json_data: Dict[str, Any]) -> Union[str, None]:
    """Find key in JSON data by phrase.

    Args:
        dict_key (str): Dictionary key.
        json_data (Dict[str, Any]): JSON data.

    Returns:
        Union[str, None]: Key if found, None otherwise.
    """
    cleaned_key = dict_key.replace("break", "line")
    for key, value in json_data.items():
        if any(phrase in cleaned_key for phrase in value["phrases"]):
            return key
    return None


def replace_value(dict_key: str, dict_value: str, json_data: Dict[str, Any]) -> str:
    """Replace value in dictionary based on JSON data.

    Args:
        dict_key (str): Dictionary key.
        dict_value (str): Dictionary value.
        json_data (Dict[str, Any]): JSON data.

    Returns:
        str: Replaced value.
    """
    key = find_key_by_phrase(dict_key, json_data)
    if not key:
        return dict_value
    new_values = [
        json_data[key]["options"][val]
        if val in json_data[key].get("options", {})
        else val
        for val in dict_value.split()
        if val != "break"
    ]
    return " ".join(new_values)


def remove_spaces_from_index(
    data: List[Dict[str, str]], index: int
) -> List[Dict[str, str]]:
    """Remove spaces from values of a specified index in data.

    Args:
        data (List[Dict[str, str]]): Data list.
        index (int): Index to update.

    Returns:
        List[Dict[str, str]]: Updated data.
    """
    if index < len(data):
        for key, value in data[index].items():
            data[index][key] = value.replace(" ", "")
    return data


def plot_mgrs_on_map(mgrs_str: str) -> folium.Map:
    """Plot MGRS coordinate on a map.

    Args:
        mgrs_str (str): MGRS string.

    Returns:
        folium.Map: Folium map object with the MGRS marker.
    """
    mgrs_ = mgrs.MGRS()
    lat, lon = mgrs_.toLatLon(mgrs_str)
    map_object = folium.Map(location=[lat, lon], zoom_start=15)
    folium.Marker([lat, lon], tooltip="MGRS Coordinate").add_to(map_object)
    return map_object


def delete_files_in_directory(directory_path: str):
    """Delete all files in a directory.

    Args:
        directory_path (str): directory path

    Returns:
        Removes all files from given directory.
    """
    try:
        files = os.listdir(directory_path)
        for file in files:
            file_path = os.path.join(directory_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print("All files deleted successfully.")
    except OSError:
        print("Error occurred while deleting files.")
