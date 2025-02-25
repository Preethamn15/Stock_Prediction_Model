﻿# Stock_Prediction_Model
A machine learning project for predicting stock closing prices using a Random Forest model. The project leverages historical stock data and allows users to input real-time stock values to predict the next closing price through a web interface built using Flask.

⚙️ Project Features:
Machine Learning Model:

Utilizes a Random Forest model (rf_model.pkl) to predict stock closing prices.
Trained on key stock features such as Open, High, Low, Volume, and Previous Day Close.
Web Interface (Flask):

A user-friendly web app to input stock values and get real-time predictions.
The web app uses HTML templates for a clean UI.
Files Included:

app.py: Flask application code to handle the web interface and predictions.
rf_model.pkl: Pre-trained Random Forest model.
requirements.txt: Project dependencies for easy environment setup.
templates/index.html: Web page for the app's UI.
EW-MAX.csv: Sample dataset used for training.
