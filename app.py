from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

app = Flask(__name__)

# Load model
model = load_model("quote_model.h5")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_seq_len = 25  # same value used during training

# Reverse word index
index_to_word = {index: word for word, index in tokenizer.word_index.items()}

# Remove OOV token index if exists
oov_index = tokenizer.word_index.get("<OOV>")


def sample_with_temperature(preds, temperature=0.7):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return np.random.choice(len(preds), p=preds)


def generate_quote(seed_text, next_words=25):
    seed_text = seed_text.lower()

    for _ in range(next_words):

        token_list = tokenizer.texts_to_sequences([seed_text])[0]

        token_list = pad_sequences(
            [token_list],
            maxlen=max_seq_len - 1,
            padding="pre"
        )

        prediction = model.predict(token_list, verbose=0)[0]

        predicted_index = sample_with_temperature(prediction)

        # 🔥 FIX: Skip OOV predictions
        if predicted_index == oov_index:
            continue

        next_word = index_to_word.get(predicted_index)

        if not next_word:
            break

        seed_text += " " + next_word

    return seed_text.capitalize()


@app.route("/", methods=["GET", "POST"])
def home():
    quote = ""
    if request.method == "POST":
        category = request.form.get("category")
        quote = generate_quote(category)

    return render_template("index.html", quote=quote)


import os
if __name__ == "__main__":
    app.run(debug = True)