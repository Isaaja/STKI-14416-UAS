### **Alur Program Sistem Rekomendasi Anime**  

1. **Unduh Dataset**  
   Langkah pertama adalah mengunduh dataset rekomendasi anime dari Kaggle melalui tautan berikut:  
   [**Anime Recommendations Database**](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database).  
   Dataset ini berisi informasi lengkap tentang anime, termasuk rating dan user preferences yang akan digunakan untuk membangun sistem rekomendasi.  

2. **Proses Preprocessing Data**  
   Setelah dataset berhasil diunduh, langkah berikutnya adalah melakukan preprocessing data untuk memastikan data siap digunakan dalam proses modeling.  
   - File notebook **`preprocessing/preprocessing.ipynb`** sudah mencakup semua tahapan preprocessing yang perlu dilakukan.  
   - Proses ini melibatkan penggabungan dua dataset yang tersedia, pembersihan data, dan persiapan fitur untuk model.  

3. **Modeling dengan KNN**  
   Setelah data selesai diproses, tahap selanjutnya adalah membangun model rekomendasi.  
   - Buka notebook **`modelling/modelling.ipynb`** untuk memulai proses modeling.  
   - Model K-Nearest Neighbors (KNN) digunakan untuk membuat sistem rekomendasi berbasis kemiripan pengguna atau item.  
   - Notebook ini mencakup semua langkah yang dibutuhkan, mulai dari pelatihan model hingga evaluasi performanya.  

4. **Integrasi Model ke Aplikasi**  
   Setelah model selesai dibuat dan diuji, langkah terakhir adalah mengintegrasikan model tersebut ke dalam sebuah aplikasi.  
   - Aplikasi dibuat menggunakan **Streamlit**, framework Python untuk membangun aplikasi berbasis web dengan cepat dan mudah.  
   - Gunakan file **`recommendationApp.py`** untuk mengimplementasikan aplikasi rekomendasi anime Anda. Aplikasi ini akan memanfaatkan model yang sudah dilatih untuk memberikan rekomendasi anime berdasarkan input pengguna.  
5. **Dokumentasi**
   Semua penjelasan tentang dataset, alur dari project ini ada di File notebook **`Dokumentasi Project/dokumentasi.ipynb`**.