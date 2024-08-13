import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
from wordcloud import WordCloud
from scipy.sparse import csr_matrix

# Memuat model dan vectorizer yang sudah disimpan
model = joblib.load('vectorizer.pkl')

# Memuat data tambahan jika diperlukan
dataset = pd.read_excel('dataset_clean.xlsx')


def load_data():
    return dataset


def preprocess_data(data):
    X_raw = data["clean_text"]
    y_raw = data["Label"]

    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X_TFIDF = vectorizer.fit_transform(X_raw)

    return X_TFIDF, y_raw, vectorizer


def train_model(X_train, y_train):
    NB = GaussianNB()
    X_train_dense = csr_matrix.toarray(X_train)
    NB.fit(X_train_dense, y_train)
    return NB


def display_evaluation(y_test, y_pred):
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    columns = sorted(y_test.unique())
    confm = confusion_matrix(y_test, y_pred, labels=columns)
    df_cm = pd.DataFrame(confm, index=columns, columns=columns)

    st.write("Confusion Matrix:")
    st.write(df_cm)


def display_wordclouds(data):
    st.write("Word Cloud untuk Semua Data:")
    all_text = ' '.join(data['clean_text'])
    wordcloud_all = WordCloud(width=800, height=400,
                              background_color='white').generate(all_text)
    st.image(wordcloud_all.to_array(), use_column_width=True)

    st.write("Word Cloud untuk Fakta:")
    fakta = data[data['Label'] == 1]
    all_text_fakta = ' '.join(fakta['clean_text'])
    wordcloud_fakta = WordCloud(
        width=800, height=400, background_color='white').generate(all_text_fakta)
    st.image(wordcloud_fakta.to_array(), use_column_width=True)

    st.write("Word Cloud untuk Hoax:")
    hoax = data[data['Label'] == 0]
    all_text_hoax = ' '.join(hoax['clean_text'])
    wordcloud_hoax = WordCloud(
        width=800, height=400, background_color='white').generate(all_text_hoax)
    st.image(wordcloud_hoax.to_array(), use_column_width=True)


def main():
    st.markdown("<h2 style='text-align: center;'>Sistem Deteksi Berita Hoax Naive Bayes</h2>",
                unsafe_allow_html=True)

    st.write(" ")

    # Membuat layout untuk komponen input dan tombol di tengah
    st.markdown("""
        <style>
        .container {
            display: flex;
            justify-content: center;
            align-items: center;
            flex-direction: column;
        }
        </style>
    """, unsafe_allow_html=True)

    # Kontainer untuk elemen-elemen di tengah
    st.markdown('<div class="container">', unsafe_allow_html=True)

    # Load data dan preprocess
    data = load_data()
    X_features, y_labels, vectorizer = preprocess_data(data)

    # Memisahkan data untuk pelatihan dan pengujian
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y_labels, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)

    # Input teks untuk diprediksi
    st.markdown("**Masukkan Judul Prediksi**")
    input_text = st.text_area("", height=150)

    # Tombol deteksi
    detect_button = st.button("Deteksi")

    st.markdown('</div>', unsafe_allow_html=True)

    # Bagian untuk menampilkan hasil deteksi
    st.write("Hasil deteksi:")
    if detect_button and input_text:
        # Transformasi teks dengan vectorizer yang digunakan untuk melatih model
        input_text_tfidf = vectorizer.transform([input_text])
        input_text_dense = csr_matrix.toarray(input_text_tfidf)

        # Prediksi menggunakan model yang telah dimuat
        prediction = model.predict(input_text_dense)
        sentiment = "Fakta" if prediction[0] == 0 else "Hoax"

        # Menampilkan hasil
        st.markdown(f"**{sentiment}**")

        # Evaluasi model
        y_pred = model.predict(csr_matrix.toarray(X_test))
        display_evaluation(y_test, y_pred)

        # Tampilkan Word Cloud di bawah hasil
        display_wordclouds(data)


if __name__ == '__main__':
    main()
