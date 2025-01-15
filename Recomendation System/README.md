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

- Mengatasi missing value

  Langkah selanjutnya adalah menangani data yang hilang atau missing value. Missing value dapat memengaruhi performa sistem rekomendasi, sehingga diperlukan penanganan yang tepat. Penanganan missing value pada projek ini dilakukan dengan cara menghapus data yang kosong. Alasana pemilihan metode penanganan missing value tersebut dikarenakan jumlah dataset yang begitu besar sehingga dengan metode menghapus data yang mengandung missing value tidak berpengaruh banyak terhadap data yang digunakan. Pada proses ini saya mengatasi missing value dengan fungsi `dropna()` Mengatasi missing value memastikan data yang digunakan bebas dari masalah, sehingga algoritma dapat bekerja secara optimal. 

- Merged data

  Dataset pengguna dan dataset anime perlu digabungkan (merge) agar dapat menghasilkan informasi yang lengkap, termasuk genre anime yang diminati oleh pengguna. Proses ini dilakukan berdasarkan kolom anime_id yang merupakan penghubung antara kedua dataset. Penggabungan data diperlukan agar setiap rating yang diberikan pengguna dapat dihubungkan dengan informasi tambahan tentang anime, seperti genre dan jumlah episode.

- Drop & Rename Column

  Pada tahap ini dilakukan drop kolom yang tidak digunakan seperti kolom rating_y, members, episodes, serta mengubah nama kolom rating_x menjadi rating. Proses ini dilakukan agar data lebih clean sehinga lebih mudah dibaca.

## Modeling

### 1. Content-Based Filtering
Content-Based Filtering adalah metode rekomendasi yang memanfaatkan informasi atau atribut yang terkait dengan item (misalnya genre, deskripsi, fitur teknis, atau metadata lainnya) untuk merekomendasikan item kepada pengguna. Pendekatan ini berfokus pada preferensi pengguna berdasarkan interaksi mereka sebelumnya dengan item yang serupa.

Dalam proyek ini akan digunakan content-based filtering dimana rekomendasi akan diberikan kepada pengguna sesuai dengan genre anime yang disukai pengguna berdasarkan data historis.

#### **Kelebihan Content-Based Filtering**
- Personalisasi Tinggi

Sistem menghasilkan rekomendasi yang sangat spesifik untuk setiap pengguna berdasarkan preferensi unik mereka.

- Tidak Memerlukan Data Pengguna Lain

Berbeda dengan Collaborative Filtering, sistem tidak membutuhkan informasi dari pengguna lain. Ini cocok untuk aplikasi dengan sedikit pengguna.

- Bekerja Baik dengan Data Baru (Cold Start untuk Pengguna)

Selama informasi tentang item tersedia, sistem dapat mulai memberikan rekomendasi tanpa bergantung pada riwayat interaksi pengguna lain.

- Rekomendasi Terkait Minat:

Item yang direkomendasikan memiliki fitur yang relevan dengan minat pengguna.

#### **Kekurangan Content-Based Filtering**
- Cold Start untuk Item Baru

Item baru dengan fitur yang belum dikenal (atau minim metadata) sulit direkomendasikan.
- Kurang Beragam

Sistem cenderung merekomendasikan item yang sangat mirip dengan preferensi sebelumnya (sering disebut serendipity problem), sehingga pengguna mungkin tidak menemukan sesuatu yang benar-benar baru.
- Ketergantungan pada Metadata

Jika data atribut item (metadata) tidak lengkap atau kurang representatif, rekomendasi bisa menjadi tidak akurat.
Tidak Dapat Menangkap Preferensi Kolektif:

Sistem tidak bisa merekomendasikan item yang populer di kalangan pengguna lain jika pengguna belum memiliki interaksi langsung dengan item serupa.
- Kompleksitas Fitur

Jika fitur atau atribut item terlalu kompleks, sistem bisa menjadi lambat atau tidak efektif tanpa preprocessing yang tepat.

#### **TF-IDF Vectorizer**
Pada tahap ini membangun sistem rekomendasi berdasarkan genre yang disukai pengguna. Pada tahap ini akan dicari bobot TF-IDF untuk setiap genre anime.

#### **Cosine Similarity**

### 2. Colaborative Filtering
Tahapan ini membahas mengenai model sisten rekomendasi yang Anda buat untuk menyelesaikan permasalahan. Sajikan top-N recommendation sebagai output.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menyajikan dua solusi rekomendasi dengan algoritma yang berbeda.
- Menjelaskan kelebihan dan kekurangan dari solusi/pendekatan yang dipilih.

## Evaluation
Pada bagian ini Anda perlu menyebutkan metrik evaluasi yang digunakan. Kemudian, jelaskan hasil proyek berdasarkan metrik evaluasi tersebut.

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

## Referensi
[1] Sitanggang, A. (2023). Sistem Rekomendasi Anime Menggunakan Metode Singular Value Decomposition (SVD) dan Cosine Similarity. Jurnal Teknologi Informasi, 2(2), 90-94. DOI: https://doi.org/10.35308/jti.v2i2.7787