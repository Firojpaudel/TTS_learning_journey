import csv
with open(r"C:\Users\firoj\OneDrive\Documents\GitHub\TTS_learning_journey\XTTS\Creating Model\metadata.csv", "r", encoding="utf-8") as f:
    reader = csv.reader(f, delimiter="|")
    for i, row in enumerate(reader, 1):
        print(f"Line {i}: {row}")
        if i >= 10:
            break