import os
import random
from pydub import AudioSegment

def shorten_mp3_files(folder_path):
    for subdir, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".mp3"):
                mp3_path = os.path.join(subdir, file)

                # Load the mp3 file
                audio = AudioSegment.from_mp3(mp3_path)

                # Ensure the audio is longer than 1 second
                if len(audio) > 1000:
                    # Pick a random start time
                    start_time = random.randint(0, len(audio) - 1000)
                    end_time = start_time + 1000

                    # Extract one second of audio
                    one_sec_audio = audio[start_time:end_time]

                    # Replace the original file with the shortened version
                    one_sec_audio.export(mp3_path, format="mp3")
                    print(f"Shortened '{mp3_path}' to 1 second")
                else:
                    print(f"File '{mp3_path}' is already less than or equal to 1 second, skipping.")

# Folder path
folder_path = "./cats_dogs_copy"
shorten_mp3_files(folder_path)

