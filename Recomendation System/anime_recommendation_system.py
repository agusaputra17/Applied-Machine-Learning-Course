# -*- coding: utf-8 -*-
"""anime_recommendation_system.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1DTNhI_2Os2JCCHFf_CkpsQCfI6-2ED_N

# Proyek: Sistem Rekomendasi Anime

# Latar Belakang

Sistem rekomendasi anime telah menjadi topik yang menarik dalam bidang teknologi informasi. Sistem rekomendasi adalah suatu program yang melakukan prediksi sesuatu item, seperti rekomendasi film, musik, buku, berita dan lain sebagainya yang menarik user. Penggunaan sistem rekomendasi dalam dunia digital semakin berkembang pesat. Sistem rekomendasi dapat membantu pengguna dalam memperoleh informasi yang relevan dengan preferensi mereka [[1]](https://doi.org/10.35308/jti.v2i2.7787). Dalam konteks anime, ribuan judul anime tersedia untuk ditonton, membuat pengguna kesulitan memilih yang sesuai dengan minat mereka. Sistem rekomendasi dapat digunakan untuk menyederhanakan proses ini, memberikan rekomendasi anime berdasarkan preferensi pengguna, dan meningkatkan pengalaman mereka dalam menemukan anime baru.

Berdasarkan permasalahan yang telah dijelaskan, proyek ini akan membuat sistem rekomendasi anime menggunakan _content-based filtering_ dan _collaborative filtering_. _Content-based filtering_ bekerja dengan menganalisis kesamaan antar konten anime, khususnya pada fitur genre. Sistem akan merekomendasikan anime kepada pengguna berdasarkan genre yang mirip dengan anime yang pernah mereka tonton atau beri rating sebelumnya. Metode kedua adalah _collaborative filtering_, yaitu metode yang bekerja berdasarkan analisis interaksi antara pengguna dengan item tertentu, seperti rating atau riwayat konsumsi konten. Berbeda dengan _content-based filtering_ yang menggunakan kesamaan fitur antar konten, _collaborative filtering_ memanfaatkan pola perilaku pengguna untuk memberikan rekomendasi.

# Business Understanding

## Problem Statements

Berdasarkan latar belakang, berikut rumusan masalah dalam proyek ini:
- Bagaimana membuat sistem rekomendasi anime berdasarkan genre yang relevan dengan preferensi pengguna?
- Bagaimana membuat sistem rekomendasi anime berdasarkan pola rating dari pengguna lain yang memiliki preferensi serupa?

## Goals

Berdasarkan pertanyaan masalah di atas, berikut tujuan yang ingin dicapai dalam proyek ini:
- Dapat membuat sistem rekomendasi berdasarkan genre yang relevan dengan preferensi pengguna.
- Dapat membuat sistem rekomendasi anime berdasarkan pola dari pengguna lain dengan preferensi serupa.

## Solutions

Berikut solusi yang dapat diberikan untuk menyelesaikan permasalahan diatas:
- Membuat sistem rekomendasi anime berbasis content-based filtering yang menggunakan genre sebagai faktor utama dalam memberikan rekomendasi.
- Mengimplementasikan metode collaborative filtering untuk merekomendasikan anime berdasarkan pola rating dari pengguna lain yang memiliki preferensi serupa.

# Datasets

Dataset yang digunakan dalam proyek ini adalah [Anime Recommendations Database](CooperUnion/anime-recommendations-database) yang dapat diunduh melalui [Kaggle](CooperUnion/anime-recommendations-database). Dataset ini berisi informasi tentang data preferensi pengguna dari 73.516 pengguna terhadap 12.294 anime. Setiap pengguna dapat menambahkan anime ke daftar yang telah mereka selesaikan dan memberikan rating. Dataset ini merupakan kompilasi dari rating-rating tersebut.

# Import Library

Langkah pertama adalah mengimpor pustaka yang diperlukan untuk memproses data, pembuatan model, sampai evaluasi.
"""

# Import library
import pandas as pd
import numpy as np
import shutil

import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

"""# Load Data

Pada tahap ini merupakan proses mendownload data dari kaggle. File yang diunduh memiliki format .zip, lalu akan dilakukan unzip file dengan perintah shutil.
"""

# download dataset dari kaggle
!kaggle datasets download CooperUnion/anime-recommendations-database

# Mengekstrak file
zip_path = "anime-recommendations-database.zip"
shutil.unpack_archive(zip_path, '.')

print("File berhasil di-unzip")

"""Tahap ini merupakan proses membaca file CSV dan mengubahnya menjadi DataFrame pandas."""

# menyimpan dataset ke dalam variabel
anime = pd.read_csv('anime.csv')
rating = pd.read_csv('rating.csv')

"""menampilkan contoh data anime"""

anime.head()

"""menampilkan contoh data rating"""

rating.head()

"""# Exploratory Data Analysis (EDA)

## Info Data

Langkah ini bertujuan untuk memahami struktur dataset, termasuk jumlah baris, kolom, tipe data, dan nilai non-null di setiap kolom.

### anime variabel
"""

anime.info()

print('jumlah data anime: ', len(anime.anime_id.unique()))
print('jumlah genre: ', len(anime.genre.unique()))
print('jumlah type: ', len(anime.type.unique()))

"""### rating variabel"""

rating.info()

rating.shape

print('jumlah data user: ', len(rating.user_id.unique()))

"""## Exploratory Data Analysis

Pada tahap ini dilakukan pengecekan distirbusi data pada beberapa atribut dalam tabel anime dan rating.

### Anime
"""

# Count anime by type
anime_type_counts = anime['type'].value_counts()
print(anime_type_counts)

# Create a bar plot for anime type distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='type', data=anime)
plt.title('Number of Anime per Type')
plt.xlabel('Anime Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

"""Gambar di atas menampilkan distribusi anime berdasarkan type. Berdasarkan hasil visualisasi terlihat bahwa type TV memiliki distribusi tertinggi yaitu 3787 dan type Music memiliki distribusi terendah yaitu 488."""

# Count anime by genre (explode genres first)
genre_exploded = anime['genre'].str.split(', ').explode()
genre_counts = genre_exploded.value_counts()
print(genre_counts)

# Create a bar plot for anime genre distribution
plt.figure(figsize=(12, 6))
sns.barplot(x=genre_counts.index, y=genre_counts.values)
plt.xticks(rotation=90)
plt.xlabel('Genre')
plt.ylabel('Number of Animes')
plt.title('Distribution of Anime Genres')
plt.show()

"""Gambar di atas menampilkan distribusi anime berdasarkan genre. Berdasarkan hasil visualisasi terlihat bahwa genre Comedy memiliki distribusi terbanyak yaitu 4645 dan genre Yaoi memiliki distribusi terendah yaitu 39.

### Rating

Pada tahap ini menampilkan top 5 anime dan genre dengan rating tertinggi dan terendah
"""

# Visualisasi top 5 anime dengan rating tertinggi
top_5_anime = anime.sort_values(by='rating', ascending=False).head(5)

plt.figure(figsize=(12, 6))
sns.barplot(x='name', y='rating', data=top_5_anime)
plt.xticks(rotation=90)
plt.xlabel('Anime Name')
plt.ylabel('Rating')
plt.title('Top 5 Anime with Highest Ratings')
plt.show()

"""Gambar di atas menampilkan top 5 anime dengan rating tertinggi. Berdasarkan visualisasi, 5 anime dengan rating tertinggi yaitu Taka no Tsume 8: Yoshide-kun no X-Files, Spoon-hime no Swing Kitchen, Mogura no Motoro, Kimi no Na wa, dan Kahei no Umi."""

# Visualisasi top 5 genre dengan rating tertinggi (agregasi rating per genre)
genre_ratings = anime.copy()
genre_ratings['genre'] = genre_ratings['genre'].str.split(', ')
genre_ratings = genre_ratings.explode('genre')
genre_ratings = genre_ratings.groupby('genre')['rating'].mean().reset_index()
top_5_genres = genre_ratings.sort_values(by='rating', ascending=False).head(5)

plt.figure(figsize=(12, 6))
sns.barplot(x='genre', y='rating', data=top_5_genres)
# plt.xticks(rotation=0)
plt.xlabel('Genre')
plt.ylabel('Average Rating')
plt.title('Top 5 Genres with Highest Average Ratings')
plt.show()

"""Gambar di atas menampilkan top 5 genre dengan rating tertinggi. Berdasarkan visualisasi, 5 genre dengan rating tertinggi yaitu Josei, Thriller, Mystery, Police, dan Shounen."""

# Visualisasi 5 anime dengan rating terbawah
bottom_5_anime = anime.sort_values(by='rating', ascending=True).head(5)

plt.figure(figsize=(12, 6))
sns.barplot(x='name', y='rating', data=bottom_5_anime)
plt.xticks(rotation=90)
plt.xlabel('Anime Name')
plt.ylabel('Rating')
plt.title('5 Anime with Lowest Ratings')
plt.show()

"""Gambar di atas menampilkan top 5 anime dengan rating terendah. Berdasarkan visualisasi, 5 anime dengan rating terendah yaitu Platonic Chain: Ansatsu Jikkouchuu, Hi Gekiga Ukiyoe Senya Ichiya, tenkuu Danzai Skelter+Heaven, Utsu Musume Sayuri, dan Hametsu no Mars."""

# Visualisasi 5 genre dengan rating terbawah (agregasi rating per genre)
genre_ratings = anime.copy()
genre_ratings['genre'] = genre_ratings['genre'].str.split(', ')
genre_ratings = genre_ratings.explode('genre')
genre_ratings = genre_ratings.groupby('genre')['rating'].mean().reset_index()
bottom_5_genres = genre_ratings.sort_values(by='rating', ascending=True).head(5)

plt.figure(figsize=(12, 6))
sns.barplot(x='genre', y='rating', data=bottom_5_genres)
# plt.xticks(rotation=90)
plt.xlabel('Genre')
plt.ylabel('Average Rating')
plt.title('5 Genres with Lowest Average Ratings')
plt.show()

"""Gambar di atas menampilkan top 5 genre dengan rating terendah. Berdasarkan visualisasi, 5 genre dengan rating terendah yaitu Dementia, Music, Yuri, Kids, dan Hentai.

# Data Preprocessing

## Handling Missing Value

Langkah ini untuk memeriksa apakah ada nilai yang hilang (missing value) di dataset.
"""

# Mengecek missing value pada dataframe anime
anime.isnull().sum()

"""Selanjutnya penanganan missing value yang dilakukan dengan cara menghapus data yang kosong dengan fungsi `dropna()`."""

# menghapus data missing value
anime = anime.dropna()

# Mengecek missing value pada dataframe anime
anime.isnull().sum()

# Mengecek missing value pada dataframe rating
rating.isnull().sum()

"""Data di atas menampilkan bahwa tabel rating bebas dari missing value.

## Drop Column

Pada tahap ini dilakukan drop kolom yang tidak digunakan.
"""

# Menghapus kolom rating_y dan members
anime.drop(['members', 'episodes'], axis=1, inplace=True)
anime

"""# Modeling

## 1. Content Based Filtering

### TF-IDF Vectorizer

Tahap ini merupakan tahapan membagi kolom genre (yang berisi daftar genre dalam satu string) menjadi nilai individual.
"""

genre_exploded = anime['genre'].str.split(', ').explode()
genre_exploded

"""Selanjutnya membuat fitur baru (genre_combined) yang menggabungkan semua genre tanpa koma. Kolom genre_combined dapat digunakan sebagai input untuk menghitung kesamaan berdasarkan genre. Lalu, menggunakan metode TF-IDF untuk mengubah teks (genre) menjadi representasi numerik."""

# Feature Engineering (Create a combined feature for genres)
anime['genre_combined'] = anime['genre'].str.replace(',', ' ')

# Content-Based Filtering (Based on genre similarity)
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(anime['genre_combined'])

tfidf_matrix.shape

"""Mengonversi matriks sparse ke bentuk dense untuk mempermudah manipulasi dan visualisasi."""

# Mengubah vektor tf-idf dalam bentuk matriks dengan fungsi todense()
tfidf_matrix.todense()

"""Menampilkan subset kecil dari matriks TF-IDF untuk memahami strukturnya. DataFrame yang menampilkan bobot TF-IDF untuk sebagian kecil anime dan genre."""

pd.DataFrame(
    tfidf_matrix.todense(),
    columns=tfidf.get_feature_names_out(),
    index=anime['name']
).sample(22, axis=1).sample(10, axis=0)

"""### Cosine Similarity

Berikut tahapan mengukur kesamaan antara anime berdasarkan genre menggunakan Cosine Similarity.
"""

cosine_sim = cosine_similarity(tfidf_matrix)

# Function to recommend animes
indices = pd.Series(anime.index, index=anime['name'])

cosine_sim

# Membuat dataframe dari variabel cosine_sim dengan baris dan kolom berupa nama anime
cosine_sim_df = pd.DataFrame(cosine_sim, index=anime['name'], columns=anime['name'])
print('Shape:', cosine_sim_df.shape)

# Melihat similarity matrix pada setiap resto
cosine_sim_df.sample(5, axis=1).sample(10, axis=0)

"""### Menampilkan Rekomendasi

Tahap ini merupakan tahapan dalam membuat function Rekomendasi Anime berdasarkan kesamaan genre.
"""

def anime_recommend(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    anime_indices = [i[0] for i in sim_scores]
    recommended_animes = anime['name'].iloc[anime_indices]
    recommended_genres = anime['genre'].iloc[anime_indices]

    # Create a DataFrame for better presentation
    recommendations = pd.DataFrame({'Anime Name': recommended_animes, 'Genres': recommended_genres})
    return recommendations

"""Melihat informasi anime "Noragami" pada tabel dataset."""

anime[anime.name.eq('Noragami')]

"""Selanjutnya menerapkan fungsi rekomendasi untuk anime "Noragami"."""

# Hasil
recommendations = anime_recommend('Noragami')
recommendations

"""Hasil di atas berhasil menampilkan 5 rekomendasi anime berdasarkan kemiripan genre dari anime "Noragami".

## 2. Collaborative Filtering
"""

# membaca dataset rating
df = rating
df

"""### Preparation

Tahap ini digunakan untuk mengubah user_id menjadi List Tanpa Duplikasi. Lalu, melakukan Encoding untuk user_id dan melakukan Dekoding Angka ke user_id.
"""

# mengubah user_id menjadi list tanpa nilai yang sama
user_ids = df['user_id'].unique().tolist()
print('list user_id: ', user_ids)

# Melakukan encoding user_id
user_to_user_encoded = {x: i for i, x in enumerate(user_ids)}
print('encoded user_id : ', user_to_user_encoded)

# Melakukan proses encoding angka ke ke user_id
user_encoded_to_user = {i: x for i, x in enumerate(user_ids)}
print('encoded angka ke user_id: ', user_encoded_to_user)

"""Selanjutnya, mengubah anime_id menjadi List Tanpa Duplikasi. Lalu seperti tahap sebelumnya, disini dilakukan encoding dan decoding pada anime_id."""

# Mengubah anime_id menjadi list tanpa nilai yang sama
anime_ids = df['anime_id'].unique().tolist()
print('list anime_id: ', anime_ids)

# Melakukan proses encoding anime_id
anime_id_to_anime_id_encoded = {x: i for i, x in enumerate(anime_ids)}
print('encoded anime_id: ', anime_id_to_anime_id_encoded)

# Melakukan proses encoding angka ke anime_id
anime_id_encoded_to_anime_id = {i: x for i, x in enumerate(anime_ids)}
print('encoded angka ke anime_id: ', anime_id_encoded_to_anime_id)

"""Tahap selanjutnya, menambahkan kolom user (encoded user ID) dan anime (encoded anime ID) ke DataFrame."""

# Mapping user_id ke dataframe user
df['user'] = df['user_id'].map(user_to_user_encoded)

# Mapping anime_id ke dataframe anime
df['anime'] = df['anime_id'].map(anime_id_to_anime_id_encoded)

# Mendapatkan jumlah user
num_users = len(user_to_user_encoded)
print(num_users)

# Mendapatkan jumlah anime
num_anime = len(anime_id_encoded_to_anime_id)
print(num_anime)

# Mengubah rating menjadi nilai float
df['rating'] = df['rating'].values.astype(np.float32)

# Nilai minimum rating
min_rating = min(df['rating'])

# Nilai maksimal rating
max_rating = max(df['rating'])

print('Number of User: {}, Number of Anime: {}, Min Rating: {}, Max Rating: {}'.format(
    num_users, num_anime, min_rating, max_rating
))

"""### Membagi Data untuk Training dan Validasi

Tahap selanjutnya adalah mengacak dataset untuk menghindari bias saat training.
"""

# Mengacak dataset
df = df.sample(frac=1, random_state=42)
df

"""**Membuat Variabel x dan y**
- x: Input berupa pasangan user dan anime.
- y: Output berupa rating yang dinormalisasi ke rentang [0, 1].

**Split Data**

Data dibagi menjadi 80% untuk training dan 20% untuk validasi.
"""

# Membuat variabel x untuk mencocokkan data user dan anime menjadi satu value
x = df[['user', 'anime']].values

# Membuat variabel y untuk membuat rating dari hasil
y = df['rating'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values

# Membagi menjadi 80% data train dan 20% data validasi
train_indices = int(0.80 * df.shape[0])
x_train, x_val, y_train, y_val = (
    x[:train_indices],
    x[train_indices:],
    y[:train_indices],
    y[train_indices:]
)

print(x, y)

# Dataset untuk pelatihan
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.shuffle(buffer_size=10000).batch(512).prefetch(tf.data.AUTOTUNE)

# Dataset untuk validasi
val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_data = val_data.batch(512).prefetch(tf.data.AUTOTUNE)

"""### Proses Training

Selanjutnya tahapan dalam membuat model rekomendasi dengan embedding untuk user dan anime.
"""

class RecommenderNet(tf.keras.Model):

  # Insialisasi fungsi
  def __init__(self, num_users, num_anime, embedding_size, **kwargs):
    super(RecommenderNet, self).__init__(**kwargs)
    self.num_users = num_users
    self.num_resto = num_anime
    self.embedding_size = embedding_size
    self.user_embedding = layers.Embedding( # layer embedding user
        num_users,
        embedding_size,
        embeddings_initializer = 'he_normal',
        embeddings_regularizer = keras.regularizers.l2(1e-6)
    )
    self.user_bias = layers.Embedding(num_users, 1) # layer embedding user bias
    self.resto_embedding = layers.Embedding( # layer embeddings resto
        num_anime,
        embedding_size,
        embeddings_initializer = 'he_normal',
        embeddings_regularizer = keras.regularizers.l2(1e-6)
    )
    self.resto_bias = layers.Embedding(num_anime, 1) # layer embedding resto bias

  def call(self, inputs):
    user_vector = self.user_embedding(inputs[:,0]) # memanggil layer embedding 1
    user_bias = self.user_bias(inputs[:, 0]) # memanggil layer embedding 2
    resto_vector = self.resto_embedding(inputs[:, 1]) # memanggil layer embedding 3
    resto_bias = self.resto_bias(inputs[:, 1]) # memanggil layer embedding 4

    dot_user_resto = tf.tensordot(user_vector, resto_vector, 2)

    x = dot_user_resto + user_bias + resto_bias

    return tf.nn.sigmoid(x) # activation sigmoid

"""Tahapan berikutnya adalah Inisialisasi dan Compile Model.
- Model diinisialisasi dengan embedding size 30.
- Loss function: Binary Crossentropy.
- Optimizer: AdamW dengan learning rate 0.001.
- Metrics: Root Mean Squared Error (RMSE).
"""

model = RecommenderNet(num_users, num_anime, 30) # inisialisasi model

# model compile
model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=1e-5),
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)

"""Selanjutnya tahapan training model, dimana model dilatih menggunakan batch size 512 dan 10 epoch dan Validasi dilakukan pada 5% data."""

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Memulai training
history = model.fit(
    train_data,
    batch_size = 512,
    epochs = 20,
    validation_data = val_data,
    callbacks=[early_stopping]
)

# Simpan model ke dalam file .h5
model.save('model')

"""### Mendapatkan Rekomendasi Anime"""

# Mengambil sample user
user_id = df.user_id.sample(1).iloc[0]
anime_watched_by_user = df[df.user_id == user_id]

# Operator bitwise (~), bisa diketahui di sini https://docs.python.org/3/reference/expressions.html
anime_not_watched = anime[~anime['anime_id'].isin(anime_watched_by_user.anime_id.values)]['anime_id']
anime_not_watched = list(
    set(anime_not_watched)
    .intersection(set(anime_id_to_anime_id_encoded.keys()))
)

anime_not_watched = [[anime_id_to_anime_id_encoded.get(x)] for x in anime_not_watched]
user_encoder = user_to_user_encoded.get(user_id)
user_array = np.hstack(
    ([[user_encoder]] * len(anime_not_watched), anime_not_watched)
)

ratings = model.predict(user_array).flatten()

top_ratings_indices = ratings.argsort()[-10:][::-1]
recommended_anime_ids = [
    anime_id_encoded_to_anime_id.get(anime_not_watched[x][0]) for x in top_ratings_indices
]

print('Menampilkan hasil rekomendasi untuk user_id: {}'.format(user_id))
print('====' * 9)
print('Anime dengan rating tertinggi dari user')
print('----' * 8)

top_anime_user = (
    anime_watched_by_user.sort_values(
        by = 'rating',
        ascending=False
    )
    .head(5)
    .anime_id.values
)

anime_df_rows = anime[anime['anime_id'].isin(top_anime_user)]
for row in anime_df_rows.itertuples():
    print(row.name, ':', row.genre)

print('----' * 8)
print('Top 10 anime recommendation')
print('----' * 8)

recommended_anime = anime[anime['anime_id'].isin(recommended_anime_ids)]
for row in recommended_anime.itertuples():
    print(row.name, ':', row.genre)

"""Dari hasil di atas, anime dengan genre genre action, comedy yang merupakan rating tertinggi yang diberikan user dengan `user_id` = 61631, sehingga sistem menampilkan 10 rekomendasi berdasarkan preferensi user lainnya yang juga menyukai anime yang diberi rating tinggi oleh user tersebut.

### Visualisasi Metrik

Pada tahap ini, dilakukan visualisasi metrik loss (training loss dan validation loss) selama proses pelatihan. Hal ini membantu kita untuk memantau performa model, mendeteksi overfitting, atau underfitting.
"""

plt.plot(history.history['root_mean_squared_error'])
plt.plot(history.history['val_root_mean_squared_error'])
plt.title('model_metrics')
plt.ylabel('root_mean_squared_error')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

print('Training loss (rmse)   : ', history.history['root_mean_squared_error'][-1])
print('Validation loss (rmse) : ', history.history['val_root_mean_squared_error'][-1])

"""Hasil di atas menunjukkan nilai training loss sebesar 0.38 dan validation loss sebesar 3.35 yang artinya nilai error cukup rendah."""