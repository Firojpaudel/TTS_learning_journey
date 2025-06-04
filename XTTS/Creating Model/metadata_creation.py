import os 
import csv 

ouput_metadata = []
dataset_path = "/content/drive/MyDrive/multilingual_dataset"

"""
Here I'm expecting to download the datasets on my colab instance and then
run this script to create the metadata file for the XTTS model.
"""
# Helper function to process files
def add_to_metadata(folder, lang, metadata_file, separator='|'):
    with open(f'{dataset_path}/{folder}/{metadata_file}', 'r') as f:
        for line in f:
            parts = line.strip().split(separator)
            wav_file = parts[0] if folder in ['en'] else f"{folder}/{parts[0]}"
            text = parts[2] if folder == 'en' else parts[1]
            output_metadata.append([f"{folder}_subset/{parts[0]}.wav", lang, text])

# German (CSS10)
add_to_metadata('de', 'de', 'transcript.txt', '|')

# English (LJSpeech)
add_to_metadata('en', 'en', 'metadata.csv', '|')

# Spanish (CSS10)
add_to_metadata('es', 'es', 'transcript.txt', '|')

# Japanese (JSUT)
with open(f'{dataset_path}/ja/transcript.txt', 'r') as f:
    for line in f:
        parts = line.strip().split(':')
        output_metadata.append([f"ja_subset/{parts[0]}.wav", "ja", parts[1]])

# Portuguese (TTSPortuguese)
add_to_metadata('pt', 'pt', 'metadata.txt', '|')

# Turkish (Common Voice)
with open(f'{dataset_path}/tr/validated.tsv', 'r') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        wav_file = row['path'].replace('.mp3', '.wav')
        if os.path.exists(f'{dataset_path}/tr_subset/{wav_file}'):
            output_metadata.append([f"tr_subset/{wav_file}", "tr", row['sentence']])

# Mandarin (AISHELL-3)
with open(f'{dataset_path}/zh-cn/transcript.txt', 'r') as f:
    for line in f:
        parts = line.strip().split(' ')
        output_metadata.append([f"zh-cn_subset/{parts[0]}.wav", "zh-cn", ' '.join(parts[1:])])

# Nepali (OpenSLR)
with open(f'{dataset_path}/ne/transcript.txt', 'r') as f:
    for line in f:
        parts = line.strip().split(' ')
        output_metadata.append([f"ne_subset/{parts[0]}.wav", "ne", ' '.join(parts[1:])])

# Hindi (Common Voice)
with open(f'{dataset_path}/in/validated.tsv', 'r') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        wav_file = row['path'].replace('.mp3', '.wav')
        if os.path.exists(f'{dataset_path}/in_subset/{wav_file}'):
            output_metadata.append([f"in_subset/{wav_file}", "in", row['sentence']])

# Save metadata
with open(f'{dataset_path}/metadata.csv', 'w') as f:
    writer = csv.writer(f, delimiter='|')
    writer.writerows(output_metadata)