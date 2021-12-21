import nltk
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import string 
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
# from keras.layers import Sequential 
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
import re
from nltk import word_tokenize
from keras.layers import Dense, Input, Flatten
# from keras.layers import GlobalAveragePooling1D
from keras.models import Model
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.tokenize import WordPunctTokenizer
from keras.layers.embeddings import Embedding
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from keras.optimizers import Adam
from keras import regularizers
from keras.layers import Activation
from sklearn.model_selection import StratifiedKFold, KFold
from app import app
from flask import flash, render_template, request, redirect, url_for, Response
from werkzeug.utils import secure_filename
from bs4 import BeautifulSoup
from tqdm import tqdm
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import os
import pandas
import re
import json
import string

UPLOAD_FOLDER = './upload'
SAVED_FOLDER = './saved'
ALLOWED_EXTENSIONS = set(['csv'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SAVED_FOLDER'] = SAVED_FOLDER
app.secret_key = 'ini_rahasia'

# initial route
@app.route('/')

# index route : show rendered template dashboard (index.html)
@app.route('/index')
def index():
    return render_template('index.html')

# file process
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def csv_convert_testing(fname):
    filename = (os.path.join(app.config['UPLOAD_FOLDER'], fname))
    table = pandas.read_csv(filename, encoding='utf8')
    rows = table.shape[0]
    cols = table.shape[1]
    tableID = BeautifulSoup(table.to_html(classes="table table-striped table-bordered", index_names=False, index=False), "html.parser")
    tableID.find('table')['id'] = 'table-testing'
    dataTable = {'Table': tableID, 'Rows': rows, 'Cols': cols}
    return dataTable

def csv_convert_result_pre(fname):
    filename = (os.path.join(app.config['SAVED_FOLDER'], fname))
    table = pandas.read_csv(filename, encoding='utf8')
    rows = table.shape[0]
    cols = table.shape[1]
    tableID = BeautifulSoup(table.to_html(classes="table table-striped table-bordered", index_names=False, index=False), "html.parser")
    tableID.find('table')['id'] = 'table-testing'
    dataTable = {'Table': tableID, 'Rows': rows, 'Cols': cols}
    return dataTable

# preprocessing route
@app.route('/preprocessing')
def preprocessing():
    return render_template('pre-upload.html')

@app.route('/preprocessing/testing', methods = ['GET', 'POST'])
def upload_file_testing():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('Tidak ada file yang dipilih')
            return redirect(url_for('preprocessing'))
        file = request.files['file']
        if file.filename == '':
            flash('Tidak ada file yang dipilih')
            return redirect(url_for('preprocessing'))
        if allowed_file(file.filename) == False:
            flash('Format file harus CSV')
            return redirect(url_for('preprocessing'))
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            dataTable = csv_convert_testing(filename)
            return render_template('pre-upload.html', tableTesting = dataTable['Table'],\
            rows = dataTable['Rows'], cols = dataTable['Cols'],\
            filename = filename, dnTesting = False)

# def tokenizer(text):
#     return text.split()

# membersihkan dokumen
def clean(doc):
    # import file yang berisi kata-kata tidak penting
    data_stopword = json.load(open('stopwords-id.json','r'))
    # menjadikan array stopword menjadi unordered collection (set())
    # agar dapat dilakukan operasi matematis seperti union, intersection, symmetric difference
    stopword = set(data_stopword)
    punctuation = set(string.punctuation)

    # menghilangkan kata tidak penting pada stopword
    stop_free = " ".join([i for i in doc.lower().split() if i not in stopword])
    # menjadi kata dasar
    stemmer = StemmerFactory().create_stemmer()
    normalized = stemmer.stem(stop_free)
    stop_free = " ".join([i for i in normalized.split() if i not in stopword])
    # membuat satu dokumen menjadi array berisi tiap kata
    y = stop_free.split()
    return y

# membersihkan dokumen berupa array
def clean_with_loop(arr):
    hasil = []
    progress = tqdm(arr)
    for item in progress:
        progress.set_description("Membersihkan dokumen")
        cleaned = clean(item)
        cleaned = ' '.join(cleaned)
        hasil.append(cleaned)
    return hasil

@app.route('/hasil_preprocessing')
def hasil_preprocessing():
    df = pandas.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], 'datatest_check.csv'))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: x.lower())

    #hapus punctuation
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'[^\w\s]', ' ', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'^\b[^\s]*\d[^\s]\b', '', x))

    # hapus link
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'https://[A-Za-z0-9./]+', ' ', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'https', ' ', x))

    # hapus numeric
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\d+', ' ', x))

    # hapus simbol
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'[^a-zA-Z]', ' ', x))

    #NORMALISASI TEXT
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bwoles\b', 'santai', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'(\bgwa\b|\bgua\b|\bgw\b|\bsya\b|\bsy\b)', 'saya', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bphp\b', 'palsu', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'(\bibuk\b|\bbuk\b|\bbu\b|\bnyokap\b)', 'ibu', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'(\bbpak\b|\bbpk\b|\bbp\b|\bpa\b|\bbpa\b|\bbokap\b)', 'bapak', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\badany\b', 'adanya', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bad\b', 'ada', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\badek\b', 'adik', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\badlh\b', 'adalah', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bags\b', 'agustus', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bakn\b', 'akan', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\balesan\b', 'alasan', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\balmt\b', 'alamat', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bapr\b', 'april', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'(\bataw\b|\batw\b|\bato\b)', 'atau', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'(\bbales\b|\bbls\b|\bblz\b)', 'balas', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bbanget\b', 'sangat', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bbbrp\b', 'beberapa', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bbdg\b', 'bandung', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bbener\b', 'benar', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bbgr\b', 'bogor', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bbgs\b', 'bagus', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bbgt\b', 'banget', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bbhkn\b', 'bahkan', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bbhw\b', 'bahwa', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bbkan\b', 'bukan', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bbkn\b', 'bukan', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bblh\b', 'boleh', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bblkg\b', 'belakang', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bbln\b', 'bulan', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'(\bbneran\b|\bbner\b|\bbnr\b)', 'benar', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bbndg\b', 'bandung', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'(\bbngks\b|\bbnks\b)', 'bungkus', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bbngng\b', 'bingung', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'(\bbnget\b|\bbngt\b|\bsngt\b)', 'sangat', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bbngunan\b', 'bangunan', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bbnjr\b', 'banjir', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'(\bbnyak\b|\bbnyk\b)', 'banyak', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'(\bbrapa\b|\bbrp\b)', 'berapa', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bbras\b', 'beras', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bbrhk\b', 'berhak', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bbrkt\b', 'berikut', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bbrtny\b', 'bertanya', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bbsa\b', 'bisa', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bbsk\b', 'besok', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bbsr\b', 'besar', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bbwh\b', 'bawah', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bbwt\b', 'buat', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bbyr\b', 'bayar', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'(\bcepet\b|\bcpt\b)', 'cepat', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bcmn\b', 'cuma', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bcra\b', 'cara', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'(\bdalem\b|\bdlm\b)', 'dalam', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'(\bdateng\b|\bdtang\b|\bdtng\b|\bdtg\b)', 'datang', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bdes\b', 'desember', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bdijwb\b', 'dijawab', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bditny\b', 'ditanya', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bdksih\b', 'dikasih', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bdkat\b', 'dekat', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'(\bkmana\b|\bkmn\b|\bmna\b|\bdmn\b)', 'mana', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'(\bdpan\b|\bdpn\b)', 'depan', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bdri\b', 'dari', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bdsn\b', 'dusun', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bds\b', 'desa', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'(\bemang\b|\bemg\b)', 'memang', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bfeb\b', 'februari', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bgd\b', 'gedung', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'(\bgimana\b|\bgmana\b|\bbgmn\b|\bgmn\b)', 'bagaimana', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bgitu\b', 'begitu', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'(\bhlm\b|\bhal\b)', 'halaman', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bhbs\b', 'habis', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bhdp\b', 'hidup', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bhnya\b', 'hanya', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'(\bhrga\b|\bhrg\b)', 'harga', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bhrs\b', 'harus', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bilang\b', 'hilang', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bjan\b', 'januari', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'(\bjngan\b|\bjgn\b)', 'jangan', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'(\bjga\b|\bjg\b)', 'juga', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bjkt\b', 'jakarta', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'(\bjlas\b|\bjls\b)', 'jelas', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'(\bjlan\b|\bjln\b|\bjl\b)', 'jalan', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'(\bjumblah\b|\bjmlh\b)', 'jumlah', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bjul\b', 'juli', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bjun\b', 'juni', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'(\bjwab\b|\bjwb\b)', 'jawab', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bkab\b', 'kabupaten', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bkb\b', 'kabupaten', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'(\bkmpng\b|\bkamp\b)', 'kampung', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'(\bkayak\b|\bky\b|\bspt\b)', 'seperti', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bkcl\b', 'kecil', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'(\bkcmtn\b|\bkec\b)', 'kecamatan', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bkeliatannya\b', 'kelihatannya', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bkeliatan\b', 'kelihatan', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bkemaren\b', 'kemarin', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'(\bkluarga\b|\bklrg\b)', 'keluarga', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bkel\b', 'kelurahan', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bkmi\b', 'kami', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'(\bknpa\b|\bknp\b)', 'kenapa', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bkntr\b', 'kantor', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bkomp\b', 'komplek', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'(\bkpda\b|\bkpd\b)', 'kepada', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'(\bkpan\b|\bkpn\b)', 'kapan', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'(\bkrtu\b|\bkrt\b)', 'kartu', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bkwatir\b', 'khawatir', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'(\blbh\b|\blbih\b)', 'lebih', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\blgsg\b', 'langsung', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'(\blgi\b|\blg\b)', 'lagi', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bsblm\b', 'sebelum', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'(\bblom\b|\bblum\b|\bblm\b|\blum\b)', 'belum', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\blwt\b', 'lewat', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bmaap\b', 'maaf', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'(\bmakasih\b|\bmakasi\b|\bmksh\b|\bthanks\b)', 'terimakasih', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'(\bpake\b|\bmake\b)', 'pakai', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bmanggil\b', 'memanggil', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bmar\b', 'maret', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'(\bmasi\b|\bmsh\b|\bmsi\b)', 'masih', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'(\bmendptkan\b|\bmndptkn\b|\bmdptkan\b)', 'mendapatkan', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'(\bmendpt\b|\bmndpt\b)', 'mendapat', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bmentri\b', 'menteri', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'(\bslamat\b|\bslmt\b|\bmet\b)', 'selamat', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'(\bmhon\b|\bmhn\b)', 'mohon', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bmlh\b', 'malah', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'(\bmnjdi\b|\bmnjd\b)', 'menjadi', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bmngkn\b', 'mungkin', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bmnjadikan\b', 'menjadikan', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'(\bmnrima\b|\bmnrma\b|\bmnrm\b|\bnerima\b)', 'menerima', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bmnrt\b', 'menurut', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bmnta\b', 'minta', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bmnum\b', 'minum', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bmnunjukan\b', 'menunjukkan', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'(\bmoga\b|\bsmg\b)', 'semoga', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bmontor\b', 'motor', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bmrk\b', 'mereka', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'(\bmsalah\b|\bmslh\b)', 'masalah', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bmsl\b', 'misal', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bnegri\b', 'negeri', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bngadu\b', 'mengadu', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bngambil\b', 'mengambil', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bngantri\b', 'mengantri', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bni\b', 'ini', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'(\borng\b|\borg\b)', 'orang', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bpadet\b', 'padat', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bpemprov\b', 'pemerintah', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'(\bpengen\b|\bpgn\b)', 'ingin', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bpdhl\b', 'padahal', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bpinter\b', 'pintar', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bpjbt\b', 'pejabat', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bpnjng\b', 'panjang', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'(\bpnya\b|\bpny\b)', 'punya', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'(\bprnah\b|\bprnh\b)', 'pernah', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bprov\b', 'provinsi', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bpsti\b', 'pasti', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bptgs\b', 'petugas', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\brmh\b', 'rumah', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\brp\b', 'rupiah', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bsampe\b', 'sampai', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bsbg\b', 'sebagai', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bscr\b', 'secara', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bsdg\b', 'sedang', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bsdgkan\b', 'sedangkan', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bttg\b', 'tentang', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'(\bsdkt\b|\bdikit\b)', 'sedikit', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'(\bskolah\b|\bsekola\b|\bsklh\b)', 'sekolah', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bsjk\b', 'sejak', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bskali\b', 'sekali', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'(\bskrang\b|\bskrng\b|\bskrg\b|\bskr\b)', 'sekarang', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'(\bkrng\b|\bkrg\b)', 'kurang', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bsktr\b', 'sekitar', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'(\bslalu\b|\bsll\b)', 'selalu', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'(\bslama\b|\bslm\b)', 'selama', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bslh\b', 'salah', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bsmua\b', 'semua', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bspy\b', 'supaya', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bsyapa\b', 'siapa', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\btau\b', 'tahu', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bttp\b', 'tetap', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bthn\b', 'tahun', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\btlah\b', 'telah', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'(\btlong\b|\btlng\b)', 'tolong', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\btmpt\b', 'tempat', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\btp\b', 'tapi', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\btsb\b', 'tersebut', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'(\bunt\b|\butk\b)', 'untuk', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bwalkot\b', 'walikota', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'(\bwly\b|\bwil\b)', 'wilayah', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'(\byng\b|\byg\b)', 'yang', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'(\bajah\b|\baje\b|\baj\b)', 'saja', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\baer\b', 'air', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bagk\b', 'agak', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'(\bak\b|\baq\b|\bgwe\b|\bgw\b)', 'aku', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'(\bape\b|\bap\b)', 'apa', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\badaa\b', 'ada', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'(\bdapet\b|\bdpat\b|\bdpt\b)', 'dapat', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bank\b', 'anak', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'(\bkasi\b|\bksh\b)', 'kasih', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bbngun\b', 'bangun', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bjd\b', 'jadi', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'(\bkarna\b|\bkrn\b)', 'karena', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'(\bdngan\b|\bdgn\b|\bdg\b)', 'dengan', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'(\bkoq\b|\bko\b)', 'kok', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'(\bmw\b|\bmo\b)', 'mau', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'(\bmikum\b|\bass\b)', 'assalamualaikum', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bd\b', 'di', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'(\bnggak\b|\bengga\b|\bngga\b|\bngk\b|\btdk\b|\btak\b|\bgak\b|\bga\b|\bg\b)', 'tidak', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\bk\b', 'ke', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'(\bkalaw\b|\bkalo\b|\bklo\b|\bklw\b|\bkl\b|\bklu\b)', 'kalau', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'(\budh\b|\buda\b|\bsdh\b|\bdah\b)', 'sudah', x))

    #hapus multispace,depan kosong,belakang kosong
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\s\s+', ' ', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'^\s+', '', x))
    df['IsiLaporan'] = df['IsiLaporan'].apply(lambda x: re.sub(r'\s+$', '', x))

    # menyimpan pada csv 
    df = df.to_csv(os.path.join(app.config['SAVED_FOLDER'], 'check_clean.csv'), index=False, columns=['IsiLaporan'])

    # import dataset 
    span = pandas.read_csv(os.path.join(app.config['SAVED_FOLDER'], 'check_clean.csv'))

    spanlapor = []
    for index, row in span.iterrows():
        spanlapor.append(row["IsiLaporan"])
    # print("Jumlah laporan: ", len(spanlapor))
    spanbersih = clean_with_loop(spanlapor)

    #simpan ke csv baru (yang sudah bersih) 
    new = pandas.DataFrame(spanbersih)
    new = new.to_csv(os.path.join(app.config['SAVED_FOLDER'], 'check_donepreprocessing.csv'), index=False, header=['IsiLaporan'])

    filename = 'check_donepreprocessing.csv'
    dataTable = csv_convert_result_pre(filename)
    return render_template('pre-done.html', tableTesting = dataTable['Table'],\
    rows = dataTable['Rows'], cols = dataTable['Cols'],\
    filename = filename, dnTesting = False)

@app.route('/process_menu')
def process_menu():
    return render_template('process-menu.html')

@app.route('/process')
def process():

    df1 = pandas.read_csv('datatrain_8_117_119.csv')
    from gensim.models.word2vec import Word2Vec

    Bigger_list = []
    for i in df1['IsiLaporan']:
        li = list(i.split(" "))
        Bigger_list.append(li)

    w2v_model = Word2Vec(Bigger_list, size = 100, window = 5, min_count = 10, workers = 8)

    texts=df1.IsiLaporan
    tokenizer = Tokenizer(num_words = None)
    tokenizer.fit_on_texts(texts)
    sequences_train = tokenizer.texts_to_sequences(df1.IsiLaporan)

    vocab = list(w2v_model.wv.vocab)

    word_index = {}
    for i, word in enumerate(vocab, 1): 
        word_index[word] = i
        
    vocab_size = len(word_index)+1  
    embedding_matrix = np.zeros((vocab_size, 100))

    for word, i in word_index.items():
        if word in w2v_model.wv:
            embedding_matrix[i] = w2v_model.wv[word] 
    print('Shape of embedding matrix :', embedding_matrix.shape)

    def text_to_int(df1, word_index, max_len):

        X = np.zeros((df1.shape[0], max_len)) 
        for i, df2 in enumerate(df1.IsiLaporan):
            words = list(df2.split(" "))
            j = 0
            for word in reversed(words):
                if word in word_index.keys():   
                    X[i, max_len-1-j] = word_index[word]
                    j += 1
        return X

    max_len = 0
    for list_ in Bigger_list:
        if len(list_)>max_len:
            max_len = len(list_)
    X = text_to_int(df1, word_index, max_len)

    Y = df1["kategori"]
    kat = Y.to_numpy()

    label_encoder = LabelEncoder()
    vec = label_encoder.fit_transform(Y)

    X_train,X_test,y_train,y_test=train_test_split(X,vec,test_size=0.2)

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    y_train = to_categorical(y_train, num_classes=3)
    y_test = to_categorical(y_test, num_classes=3)

    MAX_SEQUENCE_LENGTH = X_train.shape[1]
    EMBEDDING_DIM = 100
    embedding_layer = Embedding(len(word_index)+1, EMBEDDING_DIM, weights=[embedding_matrix], input_length = MAX_SEQUENCE_LENGTH, trainable=False)
    X_train = np.array(X_train)

    max_words = 10000
    max_phrase_len = 82

    kfold = KFold(n_splits= 5, random_state=None, shuffle=True)
    cvscores = []
    model = Sequential()
    model.add(embedding_layer)
      # model.add(SpatialDropout1D(0.5))
    model.add(LSTM(512, dropout = 0.5, recurrent_dropout = 0.5))
    model.add(Dense(3, activation = 'softmax'))
    model.compile(
        loss='categorical_crossentropy',
        optimizer='Adam',
        metrics=['accuracy'])
    # for train, test in kfold.split(X_train,y_train):
    #   history = model.fit(X_train, y_train, epochs=10, batch_size=64,validation_split=0.1)

    model.load_weights('lstm_softmax_adam_8_117_119_windowsize=5.h5')

    newdata = pandas.read_csv(os.path.join(app.config['SAVED_FOLDER'], 'check_donepreprocessing.csv'))
    padded =  text_to_int(newdata, word_index, max_len)
    kategoriteks = model.predict(padded,batch_size=1,verbose = 2)
    pred = np.round(kategoriteks, decimals=2)
    labels = ['8','117','119']

    spanlapor =[]
    for lapor in pred:
      spanlapor.append(labels[np.argmax(lapor)])
      label = pd.DataFrame(data=spanlapor,columns=['kode kategori'])

    dfn = pandas.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], 'datatest_check.csv'))
    hasil = pd.concat([dfn,newdata,label], axis=1)
    
    nama = hasil['kode kategori'].replace(['8','117','119'],['Infrastruktur','BSM','Raskin'])
    hasil1 = pd.concat([dfn,newdata,label,nama], axis=1)
    hasil1.columns.values[3] = "Kategori"
    hasil1.columns.values[1] = "IsiLaporan Setelah Preprocessing"
    hasil_new = hasil1.to_csv(os.path.join(app.config['SAVED_FOLDER'], 'check_hasil_new_data.csv'), mode = 'w', index=False)
    
    filename_new = 'check_hasil_new_data.csv'
    dataTable = csv_convert_result_pre(filename_new)
    return render_template('process-done.html', tableTesting = dataTable['Table'],\
    	rows = dataTable['Rows'], cols = dataTable['Cols'],
    	filename = filename_new, dnTesting = False)

# remove cache
@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r
