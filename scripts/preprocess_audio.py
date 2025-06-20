import os
import pandas as pd
import librosa
import numpy as np

# path to the audio directory and labels file
audio_dir = 'C:/Users/raada/Documents/Pray&Hope/data/audio'
labels_df = pd.read_csv(os.path.join(audio_dir, 'labels.csv'))

#initialize lists to hold features and labels
features = []
labels = []

# loop through each row in the labels DataFrame
for index, row in labels_df.iterrows():
    audio_file = os.path.join(audio_dir, row['filename'])
    emotion = row['emotion']

    #load the audio file and standardize it
    y, sr = librosa.load(audio_file, sr=16000) #resample to 16KHz
    y = librosa.util.normalize(y)  # normalize the audio signal

    # extract MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)

    features.append(mfccs_mean)
    labels.append(emotion)

#conver to numpy arrays
X = np.array(features)
y =np.array(labels)

np.save(os.path.join(audio_dir, 'x.npy'), X)
np.save(os.path.join(audio_dir, 'y.npy'), y)