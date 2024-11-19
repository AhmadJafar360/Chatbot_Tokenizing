from flask import Flask, request, jsonify
from flask_cors import CORS 
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

nltk.download('punkt')

app = Flask(__name__)
CORS(app, methods=['POST'], headers=['Content-Type', 'Accept']) 

with open('data.json') as json_file:
    data = json.load(json_file)

patterns = []
tags = []

patterns = []
tags = []

for intent in data["intens"]:
    for pattern in intent["patterns"]:
        patterns.append(pattern)
        tags.append(intent["tags"])

tokenizer = Tokenizer()
tokenizer.fit_on_texts(patterns)
vocab_size = len(tokenizer.word_index) + 1

X = tokenizer.texts_to_sequences(patterns)
X_padded = pad_sequences(X)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(tags)

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=50, input_length=X_padded.shape[1]))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(len(set(y_encoded)), activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_padded, y_encoded, epochs=25, batch_size=1, verbose=2)

def get_response(user_input):
    user_input_sequence = tokenizer.texts_to_sequences([user_input])
    user_input_padded = pad_sequences(user_input_sequence, maxlen=X_padded.shape[1])
    predicted_probabilities = model.predict(user_input_padded)
    predicted_class = np.argmax(predicted_probabilities, axis=-1)
    predicted_tag = label_encoder.inverse_transform(predicted_class)

    for intent in data["intens"]:
        if intent["tags"] == predicted_tag:
            return intent["responses"]

@app.route('/', methods=['OPTIONS'])
def handle_options():
    return jsonify({'status': 'success'}), 200

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data['message']

    bot_response = bot_response(user_input)

    return jsonify({'botResponse': bot_response})

if __name__ == '__main__':
    app.run(debug=True)