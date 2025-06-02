import torch
from TTS.api import TTS

#@ Device config
device = "cuda" if torch.cuda.is_available() else "cpu"

#! Show all available models
print(TTS().list_models())

#@ Init TTS 
tts = TTS("tts_models/en/ljspeech/tacotron2-DDC").to(device)

#@ Runing the TTS model
sound = tts.tts("Hey! this is how you can use TTS API in Python")

#@ Save the output to the file 
tts.tts_to_file(text="Hey! This is how you can use TTS API in Python", file_path="./outputs/api_call_day_02.wav")