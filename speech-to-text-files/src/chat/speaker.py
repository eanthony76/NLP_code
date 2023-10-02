from transformers import AutoProcessor, AutoModel
import torch
from IPython.display import Audio

# pylint: disable=no-member
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained("suno/bark")
    speech_model = AutoModel.from_pretrained("suno/bark", torch_dtype=torch.float16).to(
        device
    )
    speech_model.enable_cpu_offload()
    speech_model = speech_model.to_bettertransformer()
    inputs = processor(
        text=text_input,
        voice_preset="v2/en_speaker_6",
    ).to(device)

    speech_values = speech_model.generate(**inputs)
    sampling_rate = 24000

    audio_obj = Audio(speech_values.cpu().numpy().squeeze(), rate=sampling_rate)
    with open("test.wav", "wb") as file_out:
        file_out.write(audio_obj.data)
