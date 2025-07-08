from flask import Flask
import nltk
from nltk.tokenize import word_tokenize

app = Flask(__name__)

@app.route('/')
def index():
        nltk.download('punkt')
        return str(word_tokenize("Ini adalah contoh kalimat."))

if __name__ == '__main__':
        app.run()