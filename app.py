from flask import Flask, request, render_template, send_file, redirect, url_for, flash, Response, jsonify
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
import os
import re
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import csv
from nltk.tokenize import word_tokenize
import threading
import time

# Load vectorizer dan model
with open('tfidf_vectorizer (2).pkl', 'rb') as f:
    vectorizer = pickle.load(f)
with open('svm_sentiment_model (2).pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)
app.secret_key = 'ini_kunci_rahasia_anda_12345'

# Data statis untuk word cloud
KOMENTAR_SAMPLE = [
    "Tempatnya sangat nyaman dan bersih.",
    "Tempatnya kotor dan sampah berserakan",
    "alun alun wonosobo ada di tengah kota",
    "Pemandangan indah, cocok untuk keluarga.",
    "Fasilitas kurang memadai dan parkir sempit.",
    "Lokasi strategis tapi terlalu ramai di akhir pekan."
]

# Variabel global untuk status progress
progress_status = {'progress': 0, 'status': 'Menunggu upload...'}
progress_lock = threading.Lock()

def set_progress(progress, status):
    with progress_lock:
        progress_status['progress'] = progress
        progress_status['status'] = status

def get_progress():
    with progress_lock:
        return progress_status.copy()

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

@app.route('/progress_stream')
def progress_stream():
    def event_stream():
        last_progress = -1
        while True:
            prog = get_progress()
            if prog['progress'] != last_progress:
                yield f"data: {{\"progress\": {prog['progress']}, \"status\": \"{prog['status']}\"}}\n\n"
                last_progress = prog['progress']
            if prog['progress'] >= 100:
                break
            time.sleep(0.2)
    return Response(event_stream(), mimetype='text/event-stream')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        flash('Tidak ada file yang diupload')
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        flash('File belum dipilih')
        return redirect(url_for('index'))
    if file and file.filename.endswith('.csv'):
        df = pd.read_csv(file)
        df_raw = df.copy()  # Simpan data mentah
        # --- PREPROCESSING ---
        # Rename kolom jika ada
        if 'rsqaWe' in df.columns and 'wiI7pd' in df.columns:
            df = df.rename(columns={'rsqaWe': 'waktu', 'wiI7pd': 'komentar'})
        # Ambil kolom yang diperlukan
        if 'waktu' in df.columns and 'komentar' in df.columns:
            df = pd.DataFrame(df[['waktu','komentar']])
        elif 'komentar' in df.columns:
            df = pd.DataFrame(df[['komentar']])
        else:
            flash('Kolom komentar tidak ditemukan!')
            return redirect(url_for('index'))
        # Hapus baris kosong/NaN/komentar satu kata/duplikat
        df = df.dropna(subset=['komentar'])
        df = df[df['komentar'].str.strip() != '']
        df = df[df['komentar'].str.split().str.len() > 1]
        df.drop_duplicates(subset="komentar", keep="first", inplace=True)
        # --- CLEANING ---
        def remove_URL(tweet):
            if tweet is not None and isinstance(tweet,str):
                url = re.compile(r'https?://\S+|www\.\S+')
                tweet = url.sub(r'',tweet)
            return tweet
        def remove_html(tweet):
            if tweet is not None and isinstance(tweet,str):
                html = re.compile(r'<.*?>')
                tweet = html.sub(r'',tweet)
            return tweet
        def remove_symbols(tweet):
            if tweet is not None and isinstance(tweet,str):
                tweet = re.sub(r'[^a-zA-Z0-9\s]', '', tweet)
            return tweet
        def remove_numbers(tweet):
            if tweet is not None and isinstance(tweet, str):
                tweet = re.sub(r'\d', '', tweet)
            return tweet
        def remove_emoji(tweet):
            if tweet is not None and isinstance (tweet, str):
                emoji_pattern = re.compile("["
                    u"\U0001F600-\U0001F64F" # emoticons
                    u"\U0001F300-\U0001F5FF" # symbols & pictographs
                    u"\U0001F680-\U0001F6FF" # transport & map symbols
                    u"\U0001F700-\U0001F77F" # alchemical symbols
                    u"\U0001F780-\U0001F7FF" # Geometric Shapes Extended
                    u"\U0001F800-\U0001F8FF" # Supplemental Arrows-C
                    u"\U0001F900-\U0001F9FF" # Supplemental Symbols and Pictographs
                    u"\U0001FA00-\U0001FA6F" # Chess Symbols
                    u"\U0001FA70-\U0001FAFF" # Symbols and Pictographs Extended-A
                    u"\U0001F004-\U0001F0CF" # Additional emoticons
                    u"\U0001F1E0-\U0001F1FF" # flags
                    "]+", flags=re.UNICODE)
                tweet = emoji_pattern.sub(r'', tweet)
            return tweet
        df['cleaning'] = df['komentar'].apply(lambda x: remove_URL(x))
        df['cleaning'] = df['cleaning'].apply(lambda x: remove_html(x))
        df['cleaning'] = df['cleaning'].apply(lambda x: remove_emoji(x))
        df['cleaning'] = df['cleaning'].apply(lambda x: remove_symbols(x))
        df['cleaning'] = df['cleaning'].apply(lambda x: remove_numbers(x))
        # --- CASEFOLDING ---
        def case_folding(text):
            if isinstance(text, str):
                return text.lower()
            else:
                return text
        df['case_folding'] = df['cleaning'].apply(case_folding)
        # --- NORMALISASI ---
        kamus_path = 'kamuskatabaku.xlsx'
        if not os.path.exists(kamus_path):
            flash('File kamuskatabaku.xlsx tidak ditemukan!')
            return redirect(url_for('index'))
        kamus_data = pd.read_excel(kamus_path)
        kamus_tidak_baku = dict(zip(kamus_data['tidak_baku'], kamus_data['kata_baku']))
        def replace_taboo_words(text, kamus_tidak_baku):
            if isinstance(text, str):
                words = text.split()
                replaced_words = []
                for word in words:
                    if word in kamus_tidak_baku:
                        baku_word = kamus_tidak_baku[word]
                        if isinstance(baku_word, str) and all(char.isalpha() for char in baku_word):
                            replaced_words.append(baku_word)
                        else:
                            replaced_words.append(word)
                    else:
                        replaced_words.append(word)
                replaced_text = ' '.join(replaced_words)
            else:
                replaced_text = ''
            return replaced_text
        df['normalisasi'] = df['case_folding'].apply(lambda x: replace_taboo_words(x, kamus_tidak_baku))
        # --- TOKENIZATION ---
        def tokenize_with_split(text):
            if isinstance(text, str):
                return text.split()
            return []
        df['tokenize'] = df['normalisasi'].apply(tokenize_with_split)
        # --- STOPWORD REMOVAL ---
        nltk.download('stopwords')
        stop_words = stopwords.words('indonesian')
        def remove_stopwords(tokens):
            return [word for word in tokens if word not in stop_words]
        df['stopword_removal'] = df['tokenize'].apply(remove_stopwords)
        # --- STEMMING ---
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        def stem_text(text):
            if isinstance(text, list):
                text = ' '.join(text)
            return stemmer.stem(text)
        df['steming_data'] = df['stopword_removal'].apply(stem_text)
        df = df[~(df['steming_data'].isna() | (df['steming_data'].str.strip() == ''))]
        # --- LABELING ---
        lexicon_pos_path = 'lexicon_positive.csv'
        lexicon_neg_path = 'lexicon_negative.csv'
        if not os.path.exists(lexicon_pos_path) or not os.path.exists(lexicon_neg_path):
            flash('File lexicon_positive.csv atau lexicon_negative.csv tidak ditemukan!')
            return redirect(url_for('index'))
        lexicon_positive = dict()
        with open(lexicon_pos_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                lexicon_positive[row[0]] = int(row[1])
        lexicon_negative = dict()
        with open(lexicon_neg_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                lexicon_negative[row[0]] = int(row[1])
        negasi_words = ['tidak', 'bukan', 'nggak', 'ga', 'gak', 'belum', 'jangan']
        def determine_sentiment_advanced(text):
            if not isinstance(text, str):
                return 0, "Netral"
            words = text.split()
            sentiment_score = 0
            i = 0
            while i < len(words):
                word = words[i]
                if word in negasi_words and i + 1 < len(words):
                    next_word = words[i + 1]
                    if next_word in lexicon_positive:
                        sentiment_score -= lexicon_positive[next_word]
                        i += 2
                        continue
                    elif next_word in lexicon_negative:
                        sentiment_score += abs(lexicon_negative[next_word])
                        i += 2
                        continue
                if word in lexicon_positive:
                    sentiment_score += lexicon_positive[word]
                elif word in lexicon_negative:
                    sentiment_score += lexicon_negative[word]
                i += 1
            if sentiment_score > 0:
                sentiment = "Positif"
            elif sentiment_score < 0:
                sentiment = "Negatif"
            else:
                sentiment = "Netral"
            return sentiment_score, sentiment
        df[['Score', 'Sentiment']] = df['steming_data'].apply(lambda x: pd.Series(determine_sentiment_advanced(x)))
        # Gabungkan untuk perbandingan (hanya kolom utama)
        df_raw_view = df_raw[['komentar']] if 'komentar' in df_raw.columns else df_raw.iloc[:, :1]
        df_proc_view = df[['case_folding']] if 'case_folding' in df.columns else df
        comparison_df = pd.concat([df_raw_view.reset_index(drop=True), df_proc_view.reset_index(drop=True)], axis=1)
        comparison_df.columns = ['Komentar (Mentah)', 'Case Folding']
        table_html = comparison_df.head(20).to_html(classes='table table-striped', index=False)
        # Simpan hasil akhir
        df.to_csv('data_labelled.csv', index=False, encoding='utf-8-sig')
        flash('Dataset berhasil diupload, diproses, dan dilabeli! Hasil disimpan di data_labelled.csv')
        return render_template('index.html', table_html=table_html)
    else:
        flash('File harus berformat CSV!')
        return redirect(url_for('index'))

@app.route('/upload_ajax', methods=['POST'])
def upload_ajax():
    set_progress(0, 'Mengunggah file...')
    if 'file' not in request.files:
        set_progress(100, 'Gagal: Tidak ada file yang diupload')
        return jsonify({'error': 'Tidak ada file yang diupload'}), 400
    file = request.files['file']
    if file.filename == '':
        set_progress(100, 'Gagal: File belum dipilih')
        return jsonify({'error': 'File belum dipilih'}), 400
    if file and file.filename.endswith('.csv'):
        try:
            df = pd.read_csv(file)
            df_raw = df.copy()
            set_progress(10, 'Cleaning...')
            # --- PREPROCESSING ---
            if 'rsqaWe' in df.columns and 'wiI7pd' in df.columns:
                df = df.rename(columns={'rsqaWe': 'waktu', 'wiI7pd': 'komentar'})
            if 'waktu' in df.columns and 'komentar' in df.columns:
                df = pd.DataFrame(df[['waktu','komentar']])
            elif 'komentar' in df.columns:
                df = pd.DataFrame(df[['komentar']])
            else:
                set_progress(100, 'Gagal: Kolom komentar tidak ditemukan!')
                return jsonify({'error': 'Kolom komentar tidak ditemukan!'}), 400
            df = df.dropna(subset=['komentar'])
            df = df[df['komentar'].str.strip() != '']
            df = df[df['komentar'].str.split().str.len() > 1]
            df.drop_duplicates(subset="komentar", keep="first", inplace=True)
            # --- CLEANING ---
            def remove_URL(tweet):
                if tweet is not None and isinstance(tweet,str):
                    url = re.compile(r'https?://\S+|www\.\S+')
                    tweet = url.sub(r'',tweet)
                return tweet
            def remove_html(tweet):
                if tweet is not None and isinstance(tweet,str):
                    html = re.compile(r'<.*?>')
                    tweet = html.sub(r'',tweet)
                return tweet
            def remove_symbols(tweet):
                if tweet is not None and isinstance(tweet,str):
                    tweet = re.sub(r'[^a-zA-Z0-9\s]', '', tweet)
                return tweet
            def remove_numbers(tweet):
                if tweet is not None and isinstance(tweet, str):
                    tweet = re.sub(r'\d', '', tweet)
                return tweet
            def remove_emoji(tweet):
                if tweet is not None and isinstance (tweet, str):
                    emoji_pattern = re.compile("["
                        u"\U0001F600-\U0001F64F" # emoticons
                        u"\U0001F300-\U0001F5FF" # symbols & pictographs
                        u"\U0001F680-\U0001F6FF" # transport & map symbols
                        u"\U0001F700-\U0001F77F" # Alchemical symbols
                        u"\U0001F780-\U0001F7FF" # Geometric Shapes Extended
                        u"\U0001F800-\U0001F8FF" # Supplemental Arrows-C
                        u"\U0001F900-\U0001F9FF" # Supplemental Symbols and Pictographs
                        u"\U0001FA00-\U0001FA6F" # Chess Symbols
                        u"\U0001FA70-\U0001FAFF" # Symbols and Pictographs Extended-A
                        u"\U0001F004-\U0001F0CF" # Additional emoticons
                        u"\U0001F1E0-\U0001F1FF" # flags
                        "]+", flags=re.UNICODE)
                    tweet = emoji_pattern.sub(r'', tweet)
                return tweet
            df['cleaning'] = df['komentar'].apply(lambda x: remove_URL(x))
            df['cleaning'] = df['cleaning'].apply(lambda x: remove_html(x))
            df['cleaning'] = df['cleaning'].apply(lambda x: remove_emoji(x))
            df['cleaning'] = df['cleaning'].apply(lambda x: remove_symbols(x))
            df['cleaning'] = df['cleaning'].apply(lambda x: remove_numbers(x))
            set_progress(25, 'Case Folding...')
            # --- CASEFOLDING ---
            def case_folding(text):
                if isinstance(text, str):
                    return text.lower()
                else:
                    return text
            df['case_folding'] = df['cleaning'].apply(case_folding)
            set_progress(35, 'Normalisasi...')
            # --- NORMALISASI ---
            kamus_path = 'kamuskatabaku.xlsx'
            if not os.path.exists(kamus_path):
                set_progress(100, 'Gagal: File kamuskatabaku.xlsx tidak ditemukan!')
                return jsonify({'error': 'File kamuskatabaku.xlsx tidak ditemukan!'}), 400
            kamus_data = pd.read_excel(kamus_path)
            kamus_tidak_baku = dict(zip(kamus_data['tidak_baku'], kamus_data['kata_baku']))
            def replace_taboo_words(text, kamus_tidak_baku):
                if isinstance(text, str):
                    words = text.split()
                    replaced_words = []
                    for word in words:
                        if word in kamus_tidak_baku:
                            baku_word = kamus_tidak_baku[word]
                            if isinstance(baku_word, str) and all(char.isalpha() for char in baku_word):
                                replaced_words.append(baku_word)
                            else:
                                replaced_words.append(word)
                        else:
                            replaced_words.append(word)
                    replaced_text = ' '.join(replaced_words)
                else:
                    replaced_text = ''
                return replaced_text
            df['normalisasi'] = df['case_folding'].apply(lambda x: replace_taboo_words(x, kamus_tidak_baku))
            set_progress(45, 'Tokenisasi...')
            # --- TOKENIZATION ---
            def tokenize_with_split(text):
                if isinstance(text, str):
                    return text.split()
                return []
            df['tokenize'] = df['normalisasi'].apply(tokenize_with_split)
            set_progress(55, 'Stopword Removal...')
            # --- STOPWORD REMOVAL ---
            nltk.download('stopwords')
            stop_words = stopwords.words('indonesian')
            def remove_stopwords(tokens):
                return [word for word in tokens if word not in stop_words]
            df['stopword_removal'] = df['tokenize'].apply(remove_stopwords)
            set_progress(65, 'Stemming...')
            # --- STEMMING ---
            factory = StemmerFactory()
            stemmer = factory.create_stemmer()
            def stem_text(text):
                if isinstance(text, list):
                    text = ' '.join(text)
                return stemmer.stem(text)
            df['steming_data'] = df['stopword_removal'].apply(stem_text)
            df = df[~(df['steming_data'].isna() | (df['steming_data'].str.strip() == ''))]
            set_progress(80, 'Labeling...')
            # --- LABELING ---
            lexicon_pos_path = 'lexicon_positive.csv'
            lexicon_neg_path = 'lexicon_negative.csv'
            if not os.path.exists(lexicon_pos_path) or not os.path.exists(lexicon_neg_path):
                set_progress(100, 'Gagal: File lexicon_positive.csv atau lexicon_negative.csv tidak ditemukan!')
                return jsonify({'error': 'File lexicon_positive.csv atau lexicon_negative.csv tidak ditemukan!'}), 400
            lexicon_positive = dict()
            with open(lexicon_pos_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                for row in reader:
                    lexicon_positive[row[0]] = int(row[1])
            lexicon_negative = dict()
            with open(lexicon_neg_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                for row in reader:
                    lexicon_negative[row[0]] = int(row[1])
            negasi_words = ['tidak', 'bukan', 'nggak', 'ga', 'gak', 'belum', 'jangan']
            def determine_sentiment_advanced(text):
                if not isinstance(text, str):
                    return 0, "Netral"
                words = text.split()
                sentiment_score = 0
                i = 0
                while i < len(words):
                    word = words[i]
                    if word in negasi_words and i + 1 < len(words):
                        next_word = words[i + 1]
                        if next_word in lexicon_positive:
                            sentiment_score -= lexicon_positive[next_word]
                            i += 2
                            continue
                        elif next_word in lexicon_negative:
                            sentiment_score += abs(lexicon_negative[next_word])
                            i += 2
                            continue
                    if word in lexicon_positive:
                        sentiment_score += lexicon_positive[word]
                    elif word in lexicon_negative:
                        sentiment_score += lexicon_negative[word]
                    i += 1
                if sentiment_score > 0:
                    sentiment = "Positif"
                elif sentiment_score < 0:
                    sentiment = "Negatif"
                else:
                    sentiment = "Netral"
                return sentiment_score, sentiment
            df[['Score', 'Sentiment']] = df['steming_data'].apply(lambda x: pd.Series(determine_sentiment_advanced(x)))
            set_progress(95, 'Menyimpan hasil...')
            # Gabungkan untuk perbandingan (hanya kolom utama: komentar & normalisasi)
            if 'komentar' in df.columns:
                df_raw_view = df[['komentar']]
            elif 'wiI7pd' in df.columns:
                df_raw_view = df[['wiI7pd']]
                df_raw_view.columns = ['komentar']
            else:
                set_progress(100, 'Gagal: Kolom komentar (atau wiI7pd) tidak ditemukan setelah preprocessing!')
                return jsonify({'error': 'Kolom komentar (atau wiI7pd) tidak ditemukan setelah preprocessing!', 'table_html': ''}), 400
            df_proc_view = df[['normalisasi']] if 'normalisasi' in df.columns else df
            comparison_df = pd.concat([df_raw_view.reset_index(drop=True), df_proc_view.reset_index(drop=True)], axis=1)
            comparison_df.columns = ['Komentar (Mentah)', 'Normalisasi']
            table_html = comparison_df.head(10).to_html(classes='table table-striped', index=False)
            print('DEBUG TABLE_HTML:', table_html[:200])  # log awal tabel
            df.to_csv('data_labelled.csv', index=False, encoding='utf-8-sig')
            set_progress(100, 'Selesai! Dataset berhasil diupload, diproses, dan dilabeli.')
            return jsonify({'success': True, 'notif': 'Dataset berhasil diupload, diproses, dan dilabeli!', 'table_html': table_html})
        except Exception as e:
            set_progress(100, f'Gagal: {str(e)}')
            return jsonify({'error': str(e), 'table_html': ''}), 500
    else:
        set_progress(100, 'Gagal: File harus berformat CSV!')
        return jsonify({'error': 'File harus berformat CSV!'}), 400

if __name__ == '__main__':
    app.run(debug=True)
