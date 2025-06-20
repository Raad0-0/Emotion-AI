import pandas as pd

# Load the training data
file_path ='C:/Users/raada/Documents/Pray&Hope/data/text/training.csv'  # Adjust path if needed
df = pd.read_csv(file_path)  # Adjust path if needed

# Show the first 5 rows
print(df.head())

# # Check the number of rows and columns
# print(df.shape)

# # View column names
# print(df.columns)

# # See a summary of the data
# print(df.info())

# # Check the distribution of emotion labels
# print(df['emotion'].value_counts())
