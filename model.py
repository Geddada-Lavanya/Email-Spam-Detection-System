import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# ---------------------- Load Dataset ----------------------
data = pd.read_csv("spam_sms.csv", encoding="latin-1")

# Keep only required columns
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# print("First 5 rows of dataset:")
# print(data.head())


# ---------------------- Text Cleaning Function ----------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)   # Remove special characters
    text = re.sub(r'\s+', ' ', text) # Remove extra spaces
    return text

data['message'] = data['message'].apply(clean_text)


# ---------------------- Encode Labels ---------------------
encoder = LabelEncoder()
data['label'] = encoder.fit_transform(data['label'])  # spam=1, ham=0


# ---------------------- Text Vectorization ----------------------
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['message']).toarray()
y = data['label']


# ---------------------- Train-Test Split ----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ---------------------- Build Neural Network Model ----------------------
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])


# ---------------------- Compile Model ----------------------
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)


# ---------------------- Train Model ----------------------
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2
)


# ---------------------- Evaluate Model ----------------------
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)


# ---------------------- Prediction Function ----------------------
def predict_spam(message):
    message = clean_text(message)
    message_vector = vectorizer.transform([message]).toarray()
    prediction = model.predict(message_vector)
    return "Spam" if prediction > 0.5 else "Not Spam"


# # ---------------------- Test Examples ----------------------
# print(predict_spam("Congratulations! You have won a free gift card. Click now!"))
# print(predict_spam("Hi, are we meeting tomorrow for class?"))