import streamlit as st
import os
import sys
import pandas as pd

folder_tujuan_path = os.path.abspath('./modelling')
sys.path.append(folder_tujuan_path)
from modelling import give_recommendation

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Sistem Rekomendasi Anime",
    page_icon=':car:'
)

# Judul aplikasi Streamlit
st.title('Sistem Rekomendasi Anime')
st.write('List Anime')

# Membaca dataset anime
df = pd.read_csv('C:/SEMESTER 5/STKI/Tugas 3 STKI/data/anime.csv')

# Pilih kolom yang relevan
dataByColumn = df[["name", "genre"]]

# Tampilkan data anime di Streamlit
st.dataframe(dataByColumn)

# Sidebar untuk input pencarian anime
st.sidebar.header('Search Your Anime')

def user_input_features():
    anime_title = st.sidebar.text_input('Insert an Anime Title')
    return anime_title

# Mendapatkan input dari pengguna
user_anime = user_input_features()

# Menghasilkan rekomendasi jika input anime diberikan
if user_anime:
    # Mengimpor fungsi rekomendasi dari file modelling.py
    recommendations = give_recommendation(user_anime)

    st.write(f"Rekomendasi untuk **{user_anime}**:")
    st.write(recommendations)
else:
    st.write("Masukkan judul anime untuk melihat rekomendasi.")
