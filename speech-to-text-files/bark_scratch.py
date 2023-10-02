# coding: utf-8
from transformers import AutoProcessor, AutoModel
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained("suno/bark-small")
model = AutoModel.from_pretrained("suno/bark-small", torch_dtype=torch.float16).to(
    device
)
inputs = processor(
    text="Consider gifting her a unique, beautiful plant or flowering bulb for her garden. Would you like more suggestions?",
    voice_preset="v2/en_speaker_6",
)

speech_values = model.generate(**inputs.cuda())
import scipy

sampling_rate = 24000
scipy.io.wavfile.write(
    "bark_out.wav", rate=sampling_rate, data=speech_values.cpu().numpy().squeeze()
)
from transformers import AutoProcessor, AutoModel
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained("suno/bark-small")
model = AutoModel.from_pretrained("suno/bark-small", torch_dtype=torch.float16).to(
    device
)
inputs = processor(
    text="Consider gifting her a unique, beautiful plant or flowering bulb for her garden. Would you like more suggestions?",
    voice_preset="v2/en_speaker_6",
).to(device)

speech_values = model.generate(**inputs)
import scipy

sampling_rate = 24000
scipy.io.wavfile.write(
    "bark_out.wav", rate=sampling_rate, data=speech_values.cpu().numpy().squeeze()
)
