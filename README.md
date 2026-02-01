📩 Spam Detection System

This project is a machine learning-based web application that classifies messages as Spam or Not Spam. It uses TF-IDF for text feature extraction and a neural network model for prediction. The backend is built with Flask, and the frontend is developed using HTML, CSS, and JavaScript.


🚀 Features

Real-time spam prediction

Clean and modern user interface

Dataset preview option

Displays model accuracy


📊 Dataset

The dataset contains labeled text messages with two columns:
Label (Spam / Ham) and Message (text content).

It is used to train and test the spam classification model.

Users can view a preview of the dataset directly in the web app.


🎯 Model Accuracy

The trained neural network achieves approximately 97–98% accuracy on the test dataset.


🛠️ Technologies

Python, Pandas, Scikit-learn, TensorFlow, Flask, HTML, CSS, JavaScript


▶️ How to Run

Install dependencies:

  pip install flask pandas scikit-learn tensorflow
  

Run the app:

  python app.py
  

Open in browser:

  http://127.0.0.1:5000
