import os

import nltk
from IPython.display import Audio
import numpy as np
from bark.generation import (
    preload_models,
)
from bark import generate_audio, SAMPLE_RATE

os.environ["SUNO_OFFLOAD_CPU"] = "True"
os.environ["SUNO_USE_SMALL_MODELS"] = "True"

preload_models()

# pylint: disable=no-member, invalid-name
def speak(text_input):
    """
    Converts the given text input to speech and saves the audio output as a WAV file.

    This function uses the `suno/bark` model from HuggingFace's Transformers library
    to generate the speech. The function then processes the input text and generates speech
    values. The resultant audio is saved as "test.wav" in the current working directory.

    Parameters:
    - text_input (str): The text that needs to be converted to speech.

    Outputs:
    - A WAV file named "test.wav" containing the generated speech.

    Notes:
    - Ensure that the necessary libraries and models are available before calling
      this function.
    - The default voice preset used is "v2/en_speaker_6".
    - The sampling rate of the generated audio is 24000 Hz.

    """
    sentences = nltk.sent_tokenize(text_input)
    SPEAKER = "v2/en_speaker_6"
    silence = np.zeros(int(0.25 * SAMPLE_RATE))

    pieces = []
    for sentence in sentences:
        audio_array = generate_audio(sentence, history_prompt=SPEAKER)
        pieces += [audio_array, silence.copy()]

    audio_obj = Audio(np.concatenate(pieces).squeeze(), rate=SAMPLE_RATE)
    with open("test.wav", "wb") as file_out:
        file_out.write(audio_obj.data)
