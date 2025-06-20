import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Download NLTK data (if not already downloaded)
# nltk.download('punkt_tab')
# nltk.download('stopwords')
# nltk.download('punkt')

# Load the training data
file_path ='C:/Users/raada/Documents/Pray&Hope/data/text/training.csv' # Adjust path if needed
df = pd.read_csv(file_path)
print("kaam korse")
# print(df.head())


# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Apply preprocessing
df['cleaned_text'] = df['text'].apply(preprocess_text)
# print(df[['text', 'cleaned_text']].head())

# Save preprocessed data for use in other scripts
df.to_csv('C:/Users/raada/Documents/Pray&Hope/data/text/cleaned_training.csv', index=False)
# print("file banaise")