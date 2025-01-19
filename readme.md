# **Sistem Rekomendasi Anime**  

Aplikasi ini adalah sistem rekomendasi anime yang dibangun menggunakan model K-Nearest Neighbors (KNN). Aplikasi dapat memberikan rekomendasi anime berdasarkan input pengguna, seperti nama anime favorit, genre, atau rating.  

## **Alur Program dan Cara Pengaplikasiannya**  

### **1. Unduh Dataset**  
Langkah pertama adalah mengunduh dataset rekomendasi anime dari Kaggle melalui tautan berikut:  
[**Anime Recommendations Database**](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database).  
Dataset ini mencakup informasi lengkap tentang anime, termasuk rating dan preferensi pengguna, yang digunakan untuk membangun sistem rekomendasi.  

### **2. Proses Preprocessing Data**  
Setelah dataset berhasil diunduh, langkah berikutnya adalah melakukan preprocessing data untuk memastikan data siap digunakan dalam proses modeling:  
- Buka file notebook **`preprocessing/preprocessing.ipynb`**, yang mencakup semua langkah preprocessing.  
- Tahapan preprocessing meliputi:  
  - Penggabungan dua dataset utama.  
  - Pembersihan data (menghapus data yang tidak relevan atau kosong).  
  - Persiapan fitur untuk model rekomendasi.  

### **3. Modeling dengan KNN**  
Setelah data diproses, Anda dapat membangun model rekomendasi:  
- Buka notebook **`modelling/modelling.ipynb`** untuk proses modeling.  
- Model **K-Nearest Neighbors (KNN)** digunakan untuk membuat rekomendasi berdasarkan kemiripan pengguna atau item.  
- Notebook ini mencakup langkah-langkah seperti:  
  - Pemisahan data.  
  - Pelatihan model.  
  - Evaluasi performa model.  

### **4. Integrasi Model ke Aplikasi**  
Setelah model selesai dibuat dan diuji, langkah terakhir adalah mengintegrasikan model ke dalam aplikasi berbasis web:  
- Aplikasi dibuat menggunakan **Streamlit**, framework Python untuk membangun aplikasi web.  
- Gunakan file **`recommendationApp.py`** untuk menjalankan aplikasi. Aplikasi ini akan memanfaatkan model yang sudah dilatih untuk memberikan rekomendasi berdasarkan input pengguna.  

### **5. Dokumentasi Proyek**  
Seluruh dokumentasi proyek, termasuk penjelasan dataset dan alur proses, tersedia di notebook:  
- **`Dokumentasi Project/dokumentasi.ipynb`**.  

---

## **Cara Menggunakan Aplikasi**  

### **Menggunakan Lokal**  
1. Clone repository ini:  
   ```bash  
   git clone https://github.com/Isaaja/STKI-14416-UAS.git  
   cd STKI-14416-UAS  
   ```  

2. Install dependensi menggunakan `requirements.txt`:  
   ```bash  
   pip install -r requirements.txt  
   ```  

3. Jalankan aplikasi menggunakan Streamlit:  
   ```bash  
   streamlit run recommendationApp.py  
   ```  

4. Buka aplikasi di browser Anda di `http://localhost:8501`.  

### **Menggunakan Website**  
1. Akses aplikasi melalui tautan berikut:  
   [**STKI-14416 Anime Recommendation App**](https://stki-14416-app.streamlit.app/).  

2. Masukkan nama anime favorit Anda di sidebar.  

3. Aplikasi akan menampilkan daftar rekomendasi di bagian bawah berdasarkan input yang diberikan.  

4. Rekomendasi mencakup **10 anime terbaik** yang relevan, berdasarkan genre dan rating.  

---

## **Struktur Folder Proyek**  
```
STKI-14416-UAS/  
├── preprocessing/  
│   └── preprocessing.ipynb      # Notebook untuk preprocessing data  
├── modelling/  
│   └── modelling.ipynb           # Notebook untuk training model KNN  
│   └── modelling.py              # Untuk pemodelan yang akan diambil di aplikasi streamlit  
├── Dokumentasi Project/  
│   └── dokumentasi.ipynb         # Dokumentasi proyek  
├── recommendationApp.py          # Aplikasi utama berbasis Streamlit  
├── requirements.txt              # Daftar dependensi aplikasi  
└── fulldata.csv                  # Gabungan dari 2 dataset yang telah saya download
└── README.md                     # Dokumentasi proyek ini  
```  

---

## **Dependencies**  

Aplikasi ini membutuhkan beberapa library Python berikut:  

### **Core Libraries**  
- **numpy**: Komputasi numerik dan operasi array.  
- **pandas**: Manipulasi dan analisis data.  
- **scipy**: Algoritma ilmiah dan komputasi tambahan.  
- **scikit-learn**: Model machine learning dan evaluasi.  

### **Visualization**  
- **matplotlib**: Visualisasi data dalam bentuk grafik.  
- **seaborn**: Visualisasi statistik berbasis matplotlib.  

### **Application Framework**  
- **streamlit**: Framework untuk membuat aplikasi web berbasis Python.  

### **Other Utilities**  
- **joblib**: Serialisasi objek untuk menyimpan dan memuat model.  
- **Rapidfuzz**: Algoritma pencocokan string yang efisien.  
- **Supabase**: Backend-as-a-service untuk penyimpanan data (anime.csv).  

---

Untuk menginstal dependensi, jalankan:  
```bash  
pip install -r requirements.txt  
```  
---