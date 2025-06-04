"""
So this code replaces the sound of a speaker with another speaker's sound.
"""
import torch 
from TTS.api import TTS

#@ Setting up the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#@ TTS init 
tts = TTS(model_name="voice_conversion_models/multilingual/vctk/freevc24", progress_bar=False).to(device)

tts.voice_conversion_to_file(source_wav = "XTTS/sound_original/UZie_cucck.wav", target_wav = "XTTS/sound_original/original_sound_one.wav", file_path="XTTS/outputs/voice_converted_uzzie.wav")