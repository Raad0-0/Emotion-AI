import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

audio_dir = 'C:/Users/raada/Documents/Pray&Hope/data/audio'
X = np.load(os.path.join(audio_dir, 'x.npy'))
y = np.load(os.path.join(audio_dir, 'y.npy'))

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#train a model( Random Forest Classifier)
model = RandomForestClassifier()
model.fit(X_train, y_train)

#evaluate the model
y_pred =model.predict(X_test)
print(classification_report(y_test, y_pred))

dump_dir = 'C:/Users/raada/Documents/Pray&Hope/models'
joblib.dump(model, os.path.join(dump_dir, 'emotion_recognition_model.pkl'))
print("Model trained and saved successfully.")