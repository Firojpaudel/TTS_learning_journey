import csv
import os
from pydub import AudioSegment

# === PATHS ===
base_dir = r"C:\Users\firoj\Downloads\cv-corpus-21.0-2025-03-14-ne-NP\cv-corpus-21.0-2025-03-14\ne-NP"
clips_dir = os.path.join(base_dir, "clips")
output_wav_dir = os.path.join(base_dir, "wavs")
os.makedirs(output_wav_dir, exist_ok=True)

tsv_path = os.path.join(base_dir, "validated.tsv")
metadata_path = os.path.join(base_dir, "metadata.csv")

# === CONVERT & GENERATE METADATA ===
with open(tsv_path, "r", encoding="utf-8") as tsv_file, open(metadata_path, "w", encoding="utf-8") as out_file:
    reader = csv.DictReader(tsv_file, delimiter="\t")
    for row in reader:
        mp3_name = row["path"]
        transcript = row["sentence"]
        if mp3_name and transcript:
            mp3_path = os.path.join(clips_dir, mp3_name)
            wav_name = mp3_name.replace(".mp3", ".wav")
            wav_path = os.path.join(output_wav_dir, wav_name)

            # Convert mp3 → wav (mono, 22050 Hz)
            sound = AudioSegment.from_mp3(mp3_path)
            sound = sound.set_channels(1).set_frame_rate(22050)
            sound.export(wav_path, format="wav")

            # Write metadata
            out_file.write(f"{wav_path}|{transcript.strip()}\n")

print(f"✅ Done! Created metadata and converted WAVs.")
