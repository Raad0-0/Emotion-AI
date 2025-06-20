import joblib
import librosa
import numpy as np
from scripts.preprocess_text import preprocess_text

# Load the pre-trained models and vectorizer
audio_model = joblib.load('C:/Users/raada/Documents/Pray&Hope/models/audio_emotion_model.pkl')
text_model = joblib.load('C:/Users/raada/Documents/Pray&Hope/models/text_emotion_model.pkl')
vectorizer = joblib.load('C:/Users/raada/Documents/Pray&Hope/models/text_vectorizer.pkl')


def detect_text_emotion(text):
    cleanned_text = preprocess_text(text)
    X_new = vectorizer.transform([cleanned_text])
    prediction = text_model.predict(X_new)
    return prediction[0]


def detect_audio_emotion(audio_file):
    y, sr =librosa.load(audio_file, sr=16000)  # load and resample the audio file
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    X_new = mfccs_mean.reshape(1, -1)  # reshape for prediction
    prediction = audio_model.predict(X_new)
    return prediction[0]

