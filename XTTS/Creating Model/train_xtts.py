import os
import numpy as np
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.tts.datasets import load_tts_samples
from TTS.utils.audio import AudioProcessor
from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer

# Custom formatter for two-column metadata.csv with absolute wav paths
def custom_nepali_formatter(root_path, meta_file, ignored_speakers=None):
    """Process metadata.csv with format C:/.../wavs/file_name.wav|transcription_text"""
    items = []
    meta_file_path = meta_file if os.path.isabs(meta_file) else os.path.join(root_path, meta_file)
    with open(meta_file_path, "r", encoding="utf-8") as f:
        for line in f:
            cols = line.strip().split("|")
            if len(cols) != 2:
                print(f"Skipping invalid line: {line}")
                continue
            wav_file, text = cols
            # Normalize separators for comparison
            wav_file = wav_file.replace('/', '\\')
            root_path_normalized = root_path.replace('/', '\\')
            # If wav_file is absolute, extract relative path
            if os.path.isabs(wav_file):
                if wav_file.startswith(root_path_normalized):
                    wav_file = wav_file[len(root_path_normalized) + 1:]  # +1 for path separator
                else:
                    print(f"Skipping wav with unexpected path: {wav_file}")
                    continue
            full_wav_path = os.path.join(root_path, wav_file)
            if not os.path.exists(full_wav_path):
                print(f"Skipping missing wav: {full_wav_path}")
                continue
            items.append({
                "audio_file": full_wav_path,
                "text": text,
                "speaker_name": "nepali_speaker",  # Single speaker
                "root_path": root_path
            })
    if not items:
        raise ValueError("No valid samples found in metadata.csv")
    print(f"Loaded {len(items)} valid samples from metadata.csv")
    return items

# Base path for Creating Model
base_path = r"C:\Users\firoj\OneDrive\Documents\GitHub\TTS_learning_journey\XTTS\Creating Model"

# Debug: Print working directory and directory contents
print(f"Current working directory: {os.getcwd()}")
print(f"Contents of Creating Model: {os.listdir(base_path)}")

# Dataset configuration
dataset_config = {
    "name": "nepali_xtts",
    "dataset_name": "nepali_xtts",
    "path": base_path,
    "meta_file_train": os.path.join(base_path, "metadata.csv"),
    "meta_file_val": os.path.join(base_path, "metadata.csv"),
    "formatter": "custom",
    "ignored_speakers": [],
    "language": "ne",
    "meta_file_attn_mask": None  # Explicitly set to None
}

# Model configuration for XTTS
model_config = XttsConfig(
    model="xtts",
    run_name="nepali_xtts",
    output_path=os.path.join(base_path, "output_model"),
    audio={
        "sample_rate": 22050,
        "output_sample_rate": 24000,
        "win_length": 1024,
        "hop_length": 256,
        "num_mels": 80,
        "mel_fmin": 0,
        "mel_fmax": 8000,
    },
    languages=["ne"],
    use_phonemes=False,
    add_blank=True,
    enable_eos_bos_chars=False,
    characters={
        "characters_class": "TTS.tts.utils.text.characters.Graphemes",
        "pad": "<PAD>",
        "eos": "<EOS>",
        "bos": "<BOS>",
        "blank": "<BLNK>",
        "characters": None,
        "punctuations": "।!?,;:'\"()",
        "phonemes": None
    },
    model_args={
        "gpt_number_text_tokens": 255,
        "gpt_start_text_token": 255,
        "gpt_max_text_tokens": 402,
        "gpt_max_audio_tokens": 605,
        "gpt_layers": 30,
        "gpt_n_model_channels": 1024,
        "gpt_n_heads": 16,
        "gpt_code_stride_len": 1024,
        "num_chars": 255,
        "input_sample_rate": 22050,
        "output_sample_rate": 24000,
        "output_hop_length": 256,
        "decoder_input_dim": 1024,
        "d_vector_dim": 512
    },
    temperature=0.85,
    top_k=50,
    top_p=0.85,
    gpt_cond_len=12,
    gpt_cond_chunk_len=4,
    max_ref_len=10,
    sound_norm_refs=False
)

# Initialize audio processor
ap = AudioProcessor(**model_config.audio)

# Initialize tokenizer
tokenizer = VoiceBpeTokenizer()

# Load dataset with custom formatter
train_samples, eval_samples = load_tts_samples(dataset_config, formatter=custom_nepali_formatter)

# Initialize model
model = Xtts.init_from_config(model_config)

# Training arguments
train_args = TrainerArgs(
    batch_size=4,
    eval_batch_size=4,
    num_epochs=50,
    mixed_precision=True,
    restore_path=None
)

# Initialize trainer
trainer = Trainer(
    train_args,
    model_config,
    train_samples,
    eval_samples,
    audio_processor=ap,
    model=model,
    tokenizer=tokenizer
)

# Start training
trainer.fit()