import pandas as pd
import os

audio_dir = 'C:/Users/raada/Documents/Pray&Hope/data/audio'
# print(f'audio_dir: {audio_dir}')

# Define the mapping of emotion codes to emotion names
emotion_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised',
}

# Initialize an empty list to store the data
data = []

# Traverse the audio directory and collect file paths and emotion
for root, dirs, files in os.walk(audio_dir):
    for file in files:
        if file.endswith('.wav'):
            parts = file.split('-')
            emotion_code = parts [2]
            emotion = emotion_map.get(emotion_code, 'unknown')
            #Store the  relative path from data/audio
            relative_path = os.path.relpath(os.path.join(root, file), audio_dir)
            data.append({'filename': relative_path, 'emotion': emotion})

df = pd.DataFrame(data)


df.to_csv(os.path.join(audio_dir, 'labels.csv'), index=False)

print("file banaise")

