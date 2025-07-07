from flask import Flask, request, render_template, send_file
import pickle
import random
from wordcloud import WordCloud
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import pandas as pd
import nltk
from nltk.corpus import stopwords
from collections import Counter
import seaborn as sns

# Load vectorizer dan model
with open('tfidf_vectorizer (2).pkl', 'rb') as f:
    vectorizer = pickle.load(f)
with open('svm_sentiment_model (2).pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

# Data statis untuk word cloud
KOMENTAR_SAMPLE = [
    "Tempatnya sangat nyaman dan bersih.",
    "Tempatnya kotor dan sampah berserakan",
    "alun alun wonosobo ada di tengah kota",
    "Pemandangan indah, cocok untuk keluarga.",
    "Fasilitas kurang memadai dan parkir sempit.",
    "Lokasi strategis tapi terlalu ramai di akhir pekan."
]

@app.route('/', methods=['GET', 'POST'])
def index():
    hasil = None
    confidence = None
    keywords = []
    error = None
    if request.method == 'POST':
        teks = request.form['teks'].strip()
        if not teks:
            error = 'Komentar tidak boleh kosong.'
        else:
            fitur = vectorizer.transform([teks])
            prediksi = model.predict(fitur)[0]
            hasil = str(prediksi).capitalize()
            # Dummy confidence dan keywords (bisa diganti jika model support)
            confidence = random.randint(80, 99)
            # Ambil 2-3 kata unik dari input sebagai kata kunci (atau dummy)
            keywords = list({w.lower() for w in teks.split() if len(w) > 4})[:3]
            if not keywords:
                keywords = ['alun-alun', 'wonosobo']
    return render_template('index.html', hasil=hasil, confidence=confidence, keywords=keywords, error=error)

@app.route('/wordcloud')
def wordcloud():
    # Pastikan stopwords sudah di-download
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    # Load data
    df = pd.read_csv('data.csv')
    text = ' '.join(df['steming_data'].astype(str).tolist())
    nltk_stopwords = set(stopwords.words('indonesian'))
    wc = WordCloud(
        background_color='white',
        max_words=500,
        stopwords=nltk_stopwords,
        width=800,
        height=400,
        colormap='viridis'
    ).generate(text)
    img = io.BytesIO()
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    return send_file(img, mimetype='image/png')

@app.route('/frekuensi')
def frekuensi():
    # Pastikan stopwords sudah di-download
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    df = pd.read_csv('data.csv')
    text = " ".join(df["steming_data"].astype(str))
    stop_words = set(stopwords.words('indonesian'))
    tokens = [word for word in text.split() if word not in stop_words]
    word_counts = Counter(tokens)
    top_words = word_counts.most_common(10)
    if not top_words:
        top_words = [('', 0)]
    word, count = zip(*top_words)
    colors = plt.cm.Pastel1(range(len(word)))
    img = io.BytesIO()
    plt.figure(figsize=(12, 6))
    bars = plt.bar(word, count, color=colors)
    plt.xlabel("Kata-Kata yang sering muncul", fontsize=12, fontweight="bold")
    plt.ylabel("Jumlah Kata", fontsize=12, fontweight="bold")
    plt.title("Frekuensi Kata", fontsize=18, fontweight="bold")
    plt.xticks(rotation=45)
    for bar, num in zip(bars, count):
        plt.text(bar.get_x() + bar.get_width() / 2, num + 1, str(num), ha='center', color='black', fontsize=10)
    plt.tight_layout()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    return send_file(img, mimetype='image/png')

@app.route('/labeling')
def labeling():
    df = pd.read_csv('sentiment.csv')
    sentiment_count = df['Sentiment'].value_counts().reset_index()
    sentiment_count.columns = ['Sentiment', 'Count']
    sns.set_style('whitegrid')
    fig, ax = plt.subplots(figsize=(6, 4))
    ax = sns.barplot(
        data=sentiment_count,
        x='Sentiment',
        y='Count',
        hue='Sentiment',
        palette='pastel',
        legend=False
    )
    plt.title('Jumlah Sentiment\nMetode inSet Lexicon Based\n', fontsize=14, pad=20)
    plt.xlabel('\nClass Sentiment', fontsize=12)
    plt.ylabel('Jumlah Tweet', fontsize=12)
    total = sentiment_count['Count'].sum()
    for i, row in sentiment_count.iterrows():
        percentage = f'{100 * row["Count"] / total:.2f}%'
        ax.text(i, row["Count"] + 0.10, f'{row["Count"]}\n({percentage})', ha='center', va='bottom')
    img = io.BytesIO()
    plt.tight_layout()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    return send_file(img, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
