## XTTS: By [Coqui.ai](https://docs.coqui.ai/en/latest/models/xtts.html) 🐸:

#### 🎯 Targets:

| Task                  | Status   |
|-----------------------|:--------:|
| Normal Speech Synthesis | 🏊‍♂️ |
| Learn Voice Cloning     | ⌛ |
| Multi Speaker training  | ⏳ |
| Finetuning the TTS Model| ⏳ |

---
#### Day 01: **Synthesizing Speech**:

- xtss has its own local demo server. Just run `tts-server` in the powershell after installing TTS.
- It also has some inbuilt models. Get the list of them using `tts --list_models`
- Also, if we dont want to go through the local server we can just create synthesize the speech using:
```pwsh
tts --text "The text to be synthesized" \
    --model_name "tts_models/en/multi-dataset/tortoise-v2" \
    --out_path XTTS/outputs/first_day_output.wav
```

Output: [Click here](./outputs/first_day_output.wav) to listen.

> [!Note]
> In docs they talk about **vocoder**. 
> Well vocoder is a signal processing module that converts intermediate audio representations *(spectrograms)* into raw audio waveforms.
>
> And to use that, simply add another line inside the above script as:
> `--vocoder_name "vocoder_models/en/ljspeech/hifigan_v2"`

- Also we can run multi-speaker TTS model using:
```pwsh 
tts --text "Text for TTS."\
    --out_path XTTS/outputs/first_multispeaker_output.wav \
    --model_name "tts_models/multilingual/multi-dataset/xtts_v2"  
    --speaker_idx "<speaker_id>"
```

> [!Important]
> But before that please run: \
> `tts --model_name "tts_models/<language>/<dataset>/<model_name>"  --list_speaker_idxs` 

---
#### Day 02: **Using TTS API in a code**:

[Click Here](./TTS_api_using_day_02.py) to redirect to the code.