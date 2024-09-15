import json
import os
from datasets import load_dataset

import soundfile as sf
from pydub import AudioSegment
import numpy as np
import os

# Load the Common Voice dataset (replace 'en' with the desired language)
dataset = load_dataset("mozilla-foundation/common_voice_11_0", "ar", split="validation")

# Directory where audio files will be saved
audio_dir = "./common_voice_audio"
# os.makedirs(audio_dir, exist_ok=True)

# List to store metadata
metadata_list = []

# Iterate through the dataset and save metadata
for idx, data in enumerate(dataset):
    # Save the audio file path
    # import pdb; pdb.set_trace()
    # audio_path = os.path.join(audio_dir, f"{data['client_id']}_{idx}.mp3")
    
    # audio_data = np.array(data["audio"]["array"])
    # sample_rate = data["audio"]["sampling_rate"]

    # wav_path = audio_path.replace(".mp3", ".wav")
    # sf.write(wav_path, audio_data, sample_rate)

    # audio = AudioSegment.from_wav(wav_path)
    # audio.export(audio_path, format="mp3")
    
    # os.remove(wav_path)

    # Save the audio file locally
    # with open(audio_path, "wb") as f:
        # f.write(data["audio"]["array"].tobytes())  # Saving raw audio array
    
    # Collect metadata for each file
    metadata = {
        "client_id": data["client_id"],
        "path": data['path'],
        "sentence": data["sentence"],
        "up_votes": data.get("up_votes", 0),
        "down_votes": data.get("down_votes", 0),
        "age": data.get("age", "unknown"),
        "gender": data.get("gender", "unknown"),
        "accent": data.get("accent", "unknown"),
    }
    metadata_list.append(metadata)

# Save the metadata list to a JSON file with UTF-8 encoding
metadata_json_path = "common_voice_metadata.json"
with open(metadata_json_path, "w", encoding="utf-8") as json_file:
    json.dump(metadata_list, json_file, ensure_ascii=False, indent=4)

print(f"Metadata saved to {metadata_json_path}")
