import librosa
import numpy as np
import joblib


model = joblib.load('C:/Users/raada/Documents/Pray&Hope/models/audio_emotion_model.pkl')
#path to the new audio file
audio_file =  'C:/Users/raada/Documents/Pray&Hope/data/predict_test/03-01-06-01-02-01-01.wav'

y, sr =librosa.load(audio_file, sr=16000)  # load and resample the audio file
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
mfccs_mean = np.mean(mfccs, axis=1)
X_new = mfccs_mean.reshape(1, -1)  # reshape for prediction
emotion = model.predict(X_new)

print(f'Predicted emotion: {emotion[0]}')