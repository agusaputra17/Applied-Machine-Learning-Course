# Laporan Proyek Machine Learning - Agus Saputra Kambea

## Project Overview

Sistem rekomendasi anime telah menjadi topik yang menarik dalam bidang teknologi informasi. Sistem rekomendasi adalah suatu program yang melakukan prediksi sesuatu item, seperti rekomendasi film, musik, buku, berita dan lain sebagainya yang menarik user. Penggunaan sistem rekomendasi dalam dunia digital semakin berkembang pesat. Sistem rekomendasi dapat membantu pengguna dalam memperoleh informasi yang relevan dengan preferensi mereka [[1]](https://doi.org/10.35308/jti.v2i2.7787). Dalam konteks anime, ribuan judul anime tersedia untuk ditonton, membuat pengguna kesulitan memilih yang sesuai dengan minat mereka. Sistem rekomendasi dapat digunakan untuk menyederhanakan proses ini, memberikan rekomendasi anime berdasarkan preferensi pengguna, dan meningkatkan pengalaman mereka dalam menemukan anime baru. 

Berdasarkan permasalahan yang telah dijelaskan, proyek ini akan membuat sistem rekomendasi anime menggunakan _content-based filtering_ dan _collaborative filtering_. _Content-based filtering_ bekerja dengan menganalisis kesamaan antar konten anime, khususnya pada fitur genre. Sistem akan merekomendasikan anime kepada pengguna berdasarkan genre yang mirip dengan anime yang pernah mereka tonton atau beri rating sebelumnya. Metode kedua adalah _collaborative filtering_, yaitu metode yang bekerja berdasarkan analisis interaksi antara pengguna dengan item tertentu, seperti rating atau riwayat konsumsi konten. Berbeda dengan _content-based filtering_ yang menggunakan kesamaan fitur antar konten, _collaborative filtering_ memanfaatkan pola perilaku pengguna untuk memberikan rekomendasi.

## Business Understanding

### Problem Statements

Berdasarkan latar belakang, berikut rumusan masalah dalam proyek ini:
- Bagaimana membuat sistem rekomendasi anime berdasarkan genre yang relevan dengan preferensi pengguna?
- Bagaimana membuat sistem rekomendasi anime berdasarkan pola rating dari pengguna lain yang memiliki preferensi serupa?

### Goals

Berdasarkan pertanyaan masalah di atas, berikut tujuan yang ingin dicapai dalam proyek ini:
- Dapat membuat sistem rekomendasi berdasarkan genre yang relevan dengan preferensi pengguna.
- Dapat membuat sistem rekomendasi anime berdasarkan pola dari pengguna lain dengan preferensi serupa.

### Solution statements
Berikut solusi yang dapat diberikan untuk menyelesaikan permasalahan diatas:
- Membuat sistem rekomendasi anime berbasis content-based filtering yang menggunakan genre sebagai faktor utama dalam memberikan rekomendasi.
- Mengimplementasikan metode collaborative filtering untuk merekomendasikan anime berdasarkan pola rating dari pengguna lain yang memiliki preferensi serupa.

## Data Understanding
### Data Source
Dataset yang digunakan dalam proyek ini adalah [Anime Recommendations Database](CooperUnion/anime-recommendations-database) yang dapat diunduh melalui [Kaggle](CooperUnion/anime-recommendations-database). Dataset ini berisi informasi tentang data preferensi pengguna dari 73.516 pengguna terhadap 12.294 anime. Setiap pengguna dapat menambahkan anime ke daftar yang telah mereka selesaikan dan memberikan rating. Dataset ini merupakan kompilasi dari rating-rating tersebut.

### Data Description

- anime.csv

  Variabel anime memiliki 12294 jumlah data. Berikut deskripsi dari tiap fitur pada variabel anime.

  | Column           | Data Type | Missing Value | Description                                                     |
  |------------------|-----------|---------------|-----------------------------------------------------------------|
  | anime_id             | int64      | 0             | ID unik dari myanimelist.net yang mengidentifikasi sebuah anime                             |
  | name | object      | 0             | Judul lengkap dari anime. |
  | genre            | object    | 62            | Daftar genre anime yang dipisahkan dengan tanda koma.               |
  | type             | object      | 25             | Jenis anime, seperti movie, TV, OVA, dll.                                           |
  | episodes              | object     | 0             | Jumlah episode (1 jika berupa movie).                         |
  | rating              | float64     | 230             | Rata-rata rating dari user.                        |
  | members               | int64     | 0            | Jumlah member komunitas tiap anime yang tergabung dalam grup.                              |

  berdasarkan tabel di atas, terdapat missing value pada kolom genre sebanyak 62, object sebanyak 25, dan rating sebanyak 230.

- rating.csv

  Variabel rating memiliki 7.813.611 jumlah data. Berikut deskripsi dari tiap fitur pada variabel rating.

  | Column           | Data Type | Missing Value | Description                                                     |
  |------------------|-----------|---------------|-----------------------------------------------------------------|
  | user_id             | int64      | 0             | ID unik dari myanimelist.net yang mengidentifikasi user                            |
  | anime_id | int64      | 0             | ID unik dari myanimelist.net yang mengidentifikasi sebuah anime  |
  | rating            | int64    | 0            | Rating yang diberikan pengguna terhadap anime dengan skala 1-10 (-1 jika pengguna menonton namun tidak memberikan rating)               |

### Exploratory Data Analysis - Univariate Analysis
isinya disini

## Data Preparation

### Mengatasi missing value

  Langkah selanjutnya adalah menangani data yang hilang atau missing value. Missing value dapat memengaruhi performa sistem rekomendasi, sehingga diperlukan penanganan yang tepat. Penanganan missing value pada projek ini dilakukan dengan cara menghapus data yang kosong. Alasana pemilihan metode penanganan missing value tersebut dikarenakan jumlah dataset yang begitu besar sehingga dengan metode menghapus data yang mengandung missing value tidak berpengaruh banyak terhadap data yang digunakan. Pada proses ini saya mengatasi missing value dengan fungsi `dropna()` Mengatasi missing value memastikan data yang digunakan bebas dari masalah, sehingga algoritma dapat bekerja secara optimal. 

### Drop Column

  Pada tahap ini dilakukan drop kolom yang tidak digunakan seperti kolom episodes dan members pada tabel anime.

### Content-Based Filtering

- Memisahkan Genre menjadi Bentuk Terpisah (Explode),
  - Genre pada kolom `anime['genre']` yang awalnya berbentuk string dipisahkan berdasarkan delimiter koma (`, `).
  - Kemudian, setiap genre dipisahkan menjadi baris-baris individual menggunakan fungsi `explode()`.

- Feature Engineering (Membuat Fitur Gabungan untuk Genre),
Genre yang dipisahkan koma diubah menjadi spasi untuk mempermudah pemrosesan pada tahap berikutnya.

- Membuat Matriks TF-IDF
  - **`TfidfVectorizer`**: Digunakan untuk mengubah data teks (genre) menjadi representasi vektor berdasarkan frekuensi kata (TF-IDF).
  - **Parameter:**
    `stop_words='english'` untuk Menghapus kata-kata umum dalam bahasa Inggris yang tidak memiliki informasi penting (stop words).
  - **Output:** Matriks TF-IDF yang merepresentasikan genre dalam bentuk angka.

- Mengubah Matriks TF-IDF ke Bentuk Matriks Kepadatan
  - Matriks TF-IDF yang sparsed (jarang) diubah menjadi bentuk matriks kepadatan (dense) menggunakan fungsi `todense()`.

- Membuat DataFrame dari Matriks TF-IDF
  - Matriks TF-IDF diubah menjadi DataFrame dengan kolom berupa fitur kata (genre) dan baris berupa nama anime.
  - Fungsi `sample()` digunakan untuk menampilkan sebagian data secara acak.
  ![Screenshot 2025-01-15 134230](https://github.com/user-attachments/assets/02833850-c90a-4118-8b2b-f7624cc9afaa)


- Menghitung Similarity antar Anime (Cosine Similarity)

  - Menggunakan **Cosine Similarity** untuk menghitung kesamaan antara anime berdasarkan vektor TF-IDF.
  - Cosine Similarity mengukur sudut antara dua vektor dalam ruang multidimensi; semakin kecil sudutnya, semakin besar kesamaannya.

- Membuat Indeks Anime

  ```python
  indices = pd.Series(anime.index, index=anime['name'])
  ```
  Membuat indeks DataFrame berdasarkan nama anime agar proses pencarian lebih efisien.

- Membuat DataFrame untuk Cosine Similarity
  - DataFrame dibuat dari matriks Cosine Similarity.
  - Baris dan kolomnya diberi label berupa nama anime untuk mempermudah identifikasi kesamaan antar anime.
  - Shape dari matriks menunjukkan jumlah total anime yang dibandingkan.

  ![Screenshot 2025-01-15 135113](https://github.com/user-attachments/assets/d65b3c11-1d76-41de-9df7-f01ede91db26)


### Collaborative Filtering

- Mengubah `user_id` menjadi List tanpa Nilai yang Sama
Setiap `user_id` diubah menjadi daftar unik menggunakan fungsi `unique` dan `tolist`.

- Melakukan Encoding `user_id`
`user_id` dikonversi menjadi nilai numerik menggunakan dictionary comprehension untuk pemetaan.

- Melakukan Proses Encoding Angka ke `user_id`
Peta terbalik dibuat untuk mengonversi kembali nilai numerik ke `user_id` asli.

- Mengubah `anime_id` menjadi List tanpa Nilai yang Sama
Setiap `anime_id` diubah menjadi daftar unik mirip dengan `user_id`.

- Melakukan Proses Encoding `anime_id`
`anime_id` juga dikonversi menjadi nilai numerik menggunakan metode pemetaan serupa.

- Melakukan Proses Encoding Angka ke `anime_id`
Peta terbalik dibuat untuk mengonversi kembali nilai numerik ke `anime_id` asli.

- Mapping `user_id` dan `anime_id` ke DataFrame
Kolom baru ditambahkan ke DataFrame untuk menyimpan hasil pemetaan `user_id` dan `anime_id` ke nilai numerik.

- Mendapatkan Jumlah `user` dan `anime`
Jumlah total `user` dan `anime` dihitung menggunakan panjang dari dictionary encoding.

- Mengubah Rating menjadi Nilai Float
Kolom `rating` dikonversi ke tipe data `float32` agar dapat digunakan dalam perhitungan lebih lanjut.

- Menentukan Nilai Minimum dan Maksimum Rating
Nilai minimum dan maksimum `rating` dihitung untuk normalisasi.

- Mengacak Dataset
Dataset diacak untuk memastikan distribusi data yang merata selama proses training.

- Membuat Variabel `x` dan `y`
  - `x`: Berisi pasangan nilai numerik `user` dan `anime`.
  - `y`: Berisi nilai `rating` yang dinormalisasi berdasarkan nilai minimum dan maksimum.

- Membagi Data menjadi Training dan Validation
Dataset dibagi menjadi 80% data training dan 20% data validation untuk evaluasi model.

- Membuat Dataset dengan Prefetching
  - **Training Data**:
    ```python
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_data = train_data.shuffle(buffer_size=10000).batch(512).prefetch(tf.data.AUTOTUNE)
    ```
  - **Validation Data**:
    ```python
    val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_data = val_data.batch(512).prefetch(tf.data.AUTOTUNE)
    ```

## Model & Result

### 1. Content-Based Filtering
Content-Based Filtering adalah metode rekomendasi yang memanfaatkan informasi atau atribut yang terkait dengan item (misalnya genre, deskripsi, fitur teknis, atau metadata lainnya) untuk merekomendasikan item kepada pengguna. Pendekatan ini berfokus pada preferensi pengguna berdasarkan interaksi mereka sebelumnya dengan item yang serupa.

Dalam proyek ini akan digunakan content-based filtering dimana rekomendasi akan diberikan kepada pengguna sesuai dengan genre anime yang disukai pengguna berdasarkan data historis.

#### **Kelebihan Content-Based Filtering**
- Personalisasi Tinggi,
Sistem menghasilkan rekomendasi yang sangat spesifik untuk setiap pengguna berdasarkan preferensi unik mereka.

- Tidak Memerlukan Data Pengguna Lain,
Berbeda dengan Collaborative Filtering, sistem tidak membutuhkan informasi dari pengguna lain. Ini cocok untuk aplikasi dengan sedikit pengguna.

- Bekerja Baik dengan Data Baru (Cold Start untuk Pengguna),
Selama informasi tentang item tersedia, sistem dapat mulai memberikan rekomendasi tanpa bergantung pada riwayat interaksi pengguna lain.

- Rekomendasi Terkait Minat:,
Item yang direkomendasikan memiliki fitur yang relevan dengan minat pengguna.

#### **Kekurangan Content-Based Filtering**
- Cold Start untuk Item Baru,
Item baru dengan fitur yang belum dikenal (atau minim metadata) sulit direkomendasikan.
- Kurang Beragam,
Sistem cenderung merekomendasikan item yang sangat mirip dengan preferensi sebelumnya (sering disebut serendipity problem), sehingga pengguna mungkin tidak menemukan sesuatu yang benar-benar baru.
- Ketergantungan pada Metadata,
Jika data atribut item (metadata) tidak lengkap atau kurang representatif, rekomendasi bisa menjadi tidak akurat.
- Tidak Dapat Menangkap Preferensi Kolektif,
Sistem tidak bisa merekomendasikan item yang populer di kalangan pengguna lain jika pengguna belum memiliki interaksi langsung dengan item serupa.
- Kompleksitas Fitur,
Jika fitur atau atribut item terlalu kompleks, sistem bisa menjadi lambat atau tidak efektif tanpa preprocessing yang tepat.


#### **Menampilkan Hasil Rekomendasi (Top 10)**
Pada tahap ini akan menampilkan hasil rekomendasi berdasarkan genre dari data historis.
Berikut adalah data anime "Noragami" berdasarkan data historis.

![Screenshot 2025-01-15 140454](https://github.com/user-attachments/assets/753c6c94-2473-4402-ac0c-1f53c766e3ce)

Berikut merupakan hasil rekomendasi dari anime "Noragami" berdasarkan content-based filtering

![Screenshot 2025-01-15 225222](https://github.com/user-attachments/assets/82ffc3cb-2ccd-4050-aa02-0d42f6ba09b4)

Hasil di atas berhasil menampilkan 10 rekomendasi anime berdasarkan kemiripan genre dari anime "Noragami"

### 2. Collaborative Filtering
Collaborative Filtering adalah metode dalam sistem rekomendasi yang memanfaatkan data interaksi pengguna dengan item, seperti rating, ulasan, atau klik, untuk memberikan rekomendasi. Pendekatan ini berasumsi bahwa pengguna yang memiliki preferensi serupa di masa lalu akan cenderung menyukai item yang sama di masa depan.

Dalam proyek ini, akan dikembangkan sistem rekomendasi berdasarkan anime yang telah diberi rating oleh pengguna. Sistem ini akan mengidentifikasi pengguna yang memiliki kesukaan terhadap anime yang sama, kemudian memberikan rekomendasi anime berdasarkan rating tertinggi yang diberikan oleh pengguna lain dengan preferensi serupa.

#### **Kelebihan Collaborative Filtering**
- Tidak Membutuhkan Metadata Item,
Sistem hanya memerlukan data interaksi pengguna dengan item tanpa memeriksa atribut atau metadata item.

- Rekomendasi yang Beragam,
Sistem dapat merekomendasikan item yang tidak berhubungan secara langsung dengan preferensi pengguna tetapi populer di kalangan pengguna lain dengan preferensi serupa.

- Menangkap Preferensi Kolektif,
Sistem dapat mengenali pola populer di antara kelompok pengguna dan memberikan rekomendasi berdasarkan pola tersebut.

- Cocok untuk Data yang Kompleks,
Collaborative Filtering bekerja baik untuk dataset dengan banyak pengguna dan item.

#### **Kekurangan Collaborative Filtering**

- Masalah Cold Start untuk Pengguna Baru,
Pengguna baru tanpa riwayat interaksi tidak dapat menerima rekomendasi yang relevan karena sistem tidak memiliki informasi awal.

- Masalah Cold Start untuk Item Baru,
Item baru yang belum pernah diberi rating atau diinteraksi oleh pengguna tidak bisa direkomendasikan.

- Masalah Sparsity,
Matriks interaksi pengguna-item sering kali sangat jarang (sparse), sehingga sulit untuk menemukan kesamaan antar pengguna atau item.

- Skalabilitas,
Ketika jumlah pengguna atau item sangat besar, proses komputasi kesamaan menjadi mahal dan lambat.

Berikut langkah implementasi model collaborative filtering dalam proyek ini.

#### **Membuat Arsitektur Model Collaborative Filtering**
- **Parameter**:
  - `num_users`: Jumlah total pengguna.
  - `num_anime`: Jumlah total anime.
  - `embedding_size`: Ukuran dimensi embedding (default: 30).
- **Layer Utama**:
  - Embedding untuk `user` dan `anime`.
  - Bias untuk `user` dan `anime`.
  - Dot product antara embedding `user` dan `anime`.
  - Aktivasi sigmoid untuk mengembalikan nilai probabilitas.

#### **Menginisialisasi Model**
Model diinisialisasi dengan parameter jumlah pengguna, anime, dan ukuran embedding.

#### **Compile Model**
- **Loss**: Binary Crossentropy.
- **Optimizer**: AdamW dengan learning rate 0.001 dan weight decay 1e-5.
- **Metrik Evaluasi**: Root Mean Squared Error (RMSE).

#### **Early Stopping**
Callback EarlyStopping diterapkan untuk menghentikan pelatihan jika tidak ada peningkatan pada validasi loss:
```python
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
```

#### **Training Model**
Model dilatih menggunakan data yang telah diproses:
- **Batch Size**: 512.
- **Epochs**: 20.
- **Validation Data**: Data validasi digunakan untuk mengukur performa model selama training.
- **Callbacks**: Menggunakan EarlyStopping untuk menghindari overfitting.

#### **Menyimpan Model**
Model yang telah dilatih disimpan ke dalam file `.h5` untuk digunakan di masa depan:
```python
model.save('model')
```

#### **Menampilkan Hasil Rekomendasi dari Collaborative Filtering (Top 10 Rekomendasi)**

![Screenshot 2025-01-16 004249](https://github.com/user-attachments/assets/8540833d-2adb-4037-b075-fbafec565554)

Dari hasil di atas, anime dengan genre genre action, comedy yang merupakan rating tertinggi yang diberikan user dengan `user_id` = 61631, sehingga sistem menampilkan 10 rekomendasi berdasarkan preferensi user lainnya yang juga menyukai anime yang diberi rating tinggi oleh user tersebut.

## Evaluation

### Content-Based Filtering
Input:

![Screenshot 2025-01-15 140454](https://github.com/user-attachments/assets/b6421f02-0477-4435-9901-9cc1b92cb995)

Hasil Rekomendasi:

![Screenshot 2025-01-15 225222](https://github.com/user-attachments/assets/120bb26d-9854-4a58-9a27-f0a745eee2f9)


Berdasarkan hasil evaluasi pada hasil rekomendasi berdasarkan anime "Noragami" yang mempunyai genre "Action, Adventure, Shounen, Supernatural" di atas didapatkan hasil 10 rekomendasi anime dengan presisi kemiripan genre 9/10 atau 90%.

### Collaborative Filtering
- **Metrik Evaluasi**: Root Mean Squared Error (RMSE).
- **Formula**:

  ![Screenshot 2025-01-15 221401](https://github.com/user-attachments/assets/5ae61131-deb4-4b22-9fc7-bedf532c9738)
![Screenshot 2025-01-15 222551](https://github.com/user-attachments/assets/99b6f885-ea36-4469-9ea0-588950562b87)

  
- **Hasil**:

![output](https://github.com/user-attachments/assets/585d1e2a-f412-4777-b27e-198dfba21079)

  - Training RMSE: 0.3831
  - Validation RMSE: 0.3521
  - Hasil menunjukkan bahwa model Collaborative Filtering dapat merekomendasikan item dengan performa yang baik.

## Model Impact on Business Understanding

### Evaluasi terhadap Problem Statements
- **Bagaimana membuat sistem rekomendasi anime berdasarkan genre yang relevan dengan preferensi pengguna?**  
  Model **content-based filtering** telah berhasil menjawab pertanyaan ini dengan memberikan rekomendasi anime berdasarkan kemiripan genre menggunakan metode **TF-IDF** dan **cosine similarity**. Model ini mampu mengidentifikasi hubungan antara genre anime dan preferensi pengguna, sehingga menghasilkan rekomendasi yang relevan.

- **Bagaimana membuat sistem rekomendasi anime berdasarkan pola rating dari pengguna lain yang memiliki preferensi serupa?**  
  Model **collaborative filtering** yang diimplementasikan dengan pendekatan embedding telah menjawab pertanyaan ini. Dengan menggunakan data rating pengguna, model dapat mengenali pola preferensi antar pengguna dan memberikan rekomendasi anime berdasarkan kesamaan pola tersebut.

### Evaluasi terhadap Goals
- **Dapat membuat sistem rekomendasi berdasarkan genre yang relevan dengan preferensi pengguna.**  
  Tujuan ini berhasil dicapai melalui pendekatan content-based filtering. Model menunjukkan performa yang baik dengan merekomendasikan anime berdasarkan relevansi genre yang dihitung menggunakan **cosine similarity**. Hal ini memungkinkan pengguna untuk menemukan anime yang sesuai dengan minat mereka.

- **Dapat membuat sistem rekomendasi anime berdasarkan pola dari pengguna lain dengan preferensi serupa.**  
  Collaborative filtering telah mencapai tujuan ini dengan mencocokkan preferensi pengguna melalui pola rating. Model ini menunjukkan akurasi yang baik berdasarkan metrik evaluasi seperti **RMSE** pada data validasi.

### Evaluasi terhadap Solution Statements
- **Membuat sistem rekomendasi anime berbasis content-based filtering yang menggunakan genre sebagai faktor utama dalam memberikan rekomendasi.**  
  Solusi ini memberikan dampak positif dengan menghasilkan rekomendasi yang sangat relevan untuk pengguna yang memiliki preferensi genre spesifik. Hal ini juga memungkinkan pengguna untuk mengeksplorasi anime serupa yang mungkin belum mereka ketahui sebelumnya.

- **Mengimplementasikan metode collaborative filtering untuk merekomendasikan anime berdasarkan pola rating dari pengguna lain yang memiliki preferensi serupa.**  
  Solusi ini terbukti memberikan hasil yang memuaskan dengan merekomendasikan anime yang dihargai tinggi oleh pengguna dengan pola preferensi serupa. Pendekatan ini membantu meningkatkan personalisasi dan kepuasan pengguna terhadap sistem rekomendasi.

### Kesimpulan
Model yang dievaluasi dalam proyek ini telah memberikan dampak yang signifikan terhadap pemahaman dan penyelesaian problem statements, pencapaian goals, dan penerapan solusi statements. Dengan kombinasi content-based filtering dan collaborative filtering, sistem rekomendasi yang dihasilkan dapat memenuhi kebutuhan pengguna dan memberikan pengalaman yang lebih personal dalam menjelajahi anime.


## Referensi
[1] Sitanggang, A. (2023). Sistem Rekomendasi Anime Menggunakan Metode Singular Value Decomposition (SVD) dan Cosine Similarity. Jurnal Teknologi Informasi, 2(2), 90-94. DOI: https://doi.org/10.35308/jti.v2i2.7787
