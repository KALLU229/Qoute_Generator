# AI Quote Generator
An AI-powered Quote Generator built using LSTM (Long Short-Term Memory) and TensorFlow.
The application generates motivational and inspirational quotes based on user-provided categories such as success, love, life, and more.

## Project Overview
This project demonstrates the use of Natural Language Processing (NLP) and Deep Learning to generate human-like text.
A trained LSTM model predicts the next word in a sequence and generates meaningful quotes dynamically. The model was trained on categorized inspirational quotes and integrated into a Flask web application for user interaction.

## Features
Generate quotes based on category input
LSTM-based deep learning model
Temperature-based sampling for creative output
Flask web interface
Tokenizer saved using Pickle
Trained model saved in .h5 format

## Technologies Used
Python 3.11
Flask
TensorFlow / Keras
NumPy
Pickle

## Installation
1️⃣ Clone the Repository
git clone https://github.com/KALLU229/Qoute_Generator.git
cd Qoute_Generator
2️⃣ Create a Virtual Environment (Recommended)
For Windows:
python -m venv venv
venv\Scripts\activate
For Mac/Linux:
python3 -m venv venv
source venv/bin/activate
3️⃣ Install Dependencies
pip install -r requirements.txt
▶️ Run the Application
python app.py

Open your browser and go to:
http://127.0.0.1:5000

## How It Works
User enters a category.
The input text is tokenized.
The LSTM model predicts the next word.
The system generates a sequence of words to form a quote.
The generated quote is displayed on the webpage.

## Future Improvements
Improve grammar using Transformer-based models
Expand dataset for better generation quality
Deploy using Docker or cloud platforms
Add API support

## Author
Kanishka Kumar Shaw
B.Tech CSE Student
Deep Learning & AI Enthusiast
