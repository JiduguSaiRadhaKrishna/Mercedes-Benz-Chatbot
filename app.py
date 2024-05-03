from flask import Flask, render_template, request, jsonify
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
import numpy as np
import bs4 as bs
import urllib.request
import re

app = Flask(__name__)

# Global variables to store the trained model and vectorizer
model = None
vectorizer = None
corpus = None

def get_wikipedia_data(url):
    """
    Fetches data from the given Wikipedia URL and returns the parsed text.
    """
    get_link = urllib.request.urlopen(url)
    get_link = get_link.read()
    data = bs.BeautifulSoup(get_link, 'lxml')
    data_paragraphs = data.find_all('p')
    data_text = ''
    for para in data_paragraphs:
        data_text += para.text
    data_text = data_text.lower()
    data_text = re.sub(r'\[[0-9]*\]', '', data_text)
    return data_text

def preprocess_text(text):
    """
    Tokenizes, removes stop words and punctuation, and lemmatizes the text.
    """
    tokens = word_tokenize(text)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    words = [word.lower() for word in tokens if word.isalnum() and word.lower() not in stop_words]
    lemmatizer = WordNetLemmatizer()
    cleaned_data = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(cleaned_data)

def train_chatbot_model(corpus):
    """
    Trains a chatbot model using TF-IDF vectorization and a neural network.
    """
    global model, vectorizer
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    y = np.eye(X.shape[0])  # Dummy y values for training, assuming all questions have unique answers
    model = Sequential([
        Dense(512, activation='relu', input_shape=(X.shape[1],)),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(X.shape[0], activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X.toarray(), y, epochs=50, batch_size=32)
    return model

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    global model, vectorizer, corpus
    user_input = request.form['user_input']
    user_input = preprocess_text(user_input)

    # Use trained model to get response
    question_vec = vectorizer.transform([user_input])
    predictions = model.predict(question_vec)
    max_index = np.argmax(predictions)  # Finding the index of the most probable answer
    answer = corpus[max_index]

    return jsonify({'response': answer})

if __name__ == '__main__':
    # Load Wikipedia data and train chatbot model
    url = "https://en.wikipedia.org/wiki/Mercedes-Benz"
    data_text = get_wikipedia_data(url)
    data_sentences = sent_tokenize(data_text)
    corpus = [preprocess_text(sent) for sent in data_sentences]
    model = train_chatbot_model(corpus)

    app.run(debug=True)
