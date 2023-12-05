import os
from pydub import AudioSegment

def convert_and_delete_wav(folder_path):
    for subdir, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".wav"):
                wav_path = os.path.join(subdir, file)
                mp3_path = os.path.splitext(wav_path)[0] + ".mp3"

                # Convert wav to mp3
                audio = AudioSegment.from_wav(wav_path)
                audio.export(mp3_path, format="mp3")
                print(f"Converted '{wav_path}' to '{mp3_path}'")

                # Delete the original wav file
                os.remove(wav_path)
                print(f"Deleted '{wav_path}'")

# Folder path
folder_path = "./cats_dogs_copy"
convert_and_delete_wav(folder_path)

