# preprocess.py

import pandas as pd
import re
import os

# Load data
fake = pd.read_csv('data/Fake.csv')
true = pd.read_csv('data/True.csv')

# Add labels: 0 for fake, 1 for real
fake['label'] = 0
true['label'] = 1

# Combine datasets
data = pd.concat([fake, true], ignore_index=True)

# Shuffle the data
data = data.sample(frac=1).reset_index(drop=True)

# Clean the text (lowercase, remove punctuation and numbers)
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)      # Remove numbers
    return text

# Combine title and text, then clean
data['text'] = data['title'].fillna('') + ' ' + data['text'].fillna('')
data['text'] = data['text'].apply(clean_text)

# Keep only needed columns
data = data[['text', 'label']]

# Save cleaned dataset
cleaned_path = 'data/cleaned_data.csv'
data.to_csv(cleaned_path, index=False)
print(f"✅ Data preprocessing complete. Saved as '{cleaned_path}'.")
