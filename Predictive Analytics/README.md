# Laporan Proyek Machine Learning - Agus Saputra Kambea

## Domain Proyek

Kualitas udara menjadi salah satu indikator penting dalam menjaga kesehatan masyarakat dan kelestarian lingkungan, terutama di daerah perkotaan yang padat penduduk dan memiliki tingkat polusi yang tinggi. Peningkatan aktivitas manusia, seperti transportasi, industri, dan pembakaran bahan bakar fosil, berkontribusi signifikan terhadap pencemaran udara. Hal ini menyebabkan peningkatan konsentrasi polutan seperti CO, PM10, PM2.5, O₃, dan NO₂, yang secara langsung memengaruhi kesehatan manusia dan lingkungan.

Beberapa isu utama yang melatarbelakangi pentingnya proyek ini:
1. Dampak Kesehatan:
Paparan polutan udara dapat menyebabkan berbagai masalah kesehatan, termasuk penyakit pernapasan, kardiovaskular, dan bahkan kematian dini. World Health Organization (WHO) telah menyatakan bahwa kualitas udara yang buruk merupakan salah satu ancaman kesehatan global.
2. Dampak Lingkungan:
Polutan seperti ozon dan nitrogen dioksida tidak hanya berdampak pada kesehatan manusia tetapi juga menyebabkan kerusakan lingkungan, seperti penurunan kualitas tanah dan air serta gangguan ekosistem.

## Business Understanding

### Problem Statements

Dengan meningkatnya polusi udara di daerah perkotaan, diperlukan sistem yang dapat memprediksi kualitas udara. Berdasarkan hal tersebut, batasan masalah yang dapat diselesaikan dalam proyek ini yaitu:
> Bagaimana membuat model _machine learning_ yang dapat digunakan untuk memprediksi kualitas udara?

### Goals

> Memprediksi nilai AQI (Air Quality Index) berdasarkan parameter polusi udara menggunakan machine learning.

### Solution statements
- Pemrosesan Data: Melakukan normalisasi data untuk memastikan semua parameter berada dalam skala yang sama.
- Membagi data menjadi subset pelatihan dan validasi untuk menghindari overfitting.
- Pemilihan Model: Menggunakan model Neural Networks (CNN + LSTM) untuk menangkap pola temporal.
- Memanfaatkan teknik dropout untuk menghindari overfitting.
- Evaluasi Model: Menggunakan Mean Absolute Error (MAE) sebagai metrik evaluasi.
- Membandingkan prediksi model dengan data aktual melalui visualisasi.

## Data Understanding
### Data Source
**Sumber dataset** : [air quality index daily values report](https://www.epa.gov/outdoor-air-quality-data/air-quality-index-daily-values-report)

Dataset yang digunakan memiliki format `.csv` yang berisi data kualitas udara harian Chicago-Naperville-Elgin, IL-IN-WI dari tahun 2015-2024 dengan jumlah dataset 3634 _rows_. 

### Data Description
Berikut deskripsi tiap kolom dalam dataset yang digunakan.

| Column           | Data Type | Missing Value | Description                                                     |
|------------------|-----------|---------------|-----------------------------------------------------------------|
| Date             | date      | 0             | Tanggal                             |
| Overall AQI Value| int64     | 0             | Indeks kualitas udara yang mencerminkan tingkat risiko kesehatan. |
| CO               | float64   | 99            | Konsentrasi gas CO (Karbon Monoksida) dalam udara               |
| Ozone            | int64     | 0             | Konsentrasi gas ozon.                                           |
| PM10             | int64     | 0             | Partikel udara dengan diameter ≤ 10 µm.                         |
| PM25             | int64     | 0             | Partikel udara dengan diameter ≤ 2.5 µm.                        |
| NO2              | float64   | 42            | Konsentrasi gas nitrogen dioksida.                              |

Berdasarkan tabel diatas, terdapat _missing value_ pada kolom `CO` sebanyak 99 baris dan pada kolom `NO2` sebanyak 42 baris.

### Outlier
![outlier](https://github.com/user-attachments/assets/29b44ce5-df79-4b00-9753-1667882e11f6)


Berdasarkan gambar di atas, terdapat outlier yang ditunjukkan dengan adanya titik-titik yang berada di luar batas whisker (garis vertikal yang menunjukkan jangkauan data non-outlier) pada masing-masing boxplot untuk setiap fitur, seperti CO, Ozone, PM10, PM25, NO2, dan Overall AQI Value. Outlier ini menunjukkan data yang berada jauh dari rentang nilai mayoritas.

### Data Distribution
![distribution](https://github.com/user-attachments/assets/53dbfe99-3aeb-4ac1-b939-cc46039fb46e)

### Data Correlation
![corr_feature](https://github.com/user-attachments/assets/7143f05b-9cfd-42cd-81d0-96c63bc9478a)

Gambar di atas menunjukan Korelasi data. Korelasi data adalah ukuran yang menunjukkan sejauh mana dua variabel atau lebih berhubungan satu sama lain. Semakin kuat korelasi maka nilai koefisien akan mendekati 1 atau bernilai 1.

## Data Preparation
- Penanganan Missing Values, penanganan Missing Values dilakukan dengan menggunakan teknik interpolasi berdasarkan waktu  (`method='time'`) memanfaatkan informasi dari indeks waktu dalam dataset untuk memperkirakan nilai yang hilang secara linier atau sesuai pola temporal data. Missing values dapat memengaruhi hasil model, terutama pada metode pembelajaran mesin yang sensitif terhadap data yang tidak lengkap.
- Menangani Outlier, penanganan outlier dilakukan dengan menggunakan teknik clipping berbasis percentile. Metode clipping memotong nilai-nilai pada data yang berada di luar rentang tertentu sehingga tidak melebihi batas bawah (lower bound) dan batas atas (upper bound). Penanganan outlier dilakukan untuk mencegah overfitting, dimana outlier dapat menyebabkan model menjadi terlalu kompleks karena mencoba menyesuaikan data yang tidak representatif.
- Normalisasi Data, Data dinormalisasi menggunakan rumus berikut:
  ```math
  normalized\_value = \frac{value - min}{max - min}
  ```
  Normalisasi Data digunakan untuk menyamakan skala fitur. Banyak algoritma machine learning (seperti KNN, SVM, atau Neural Network) yang sensitif terhadap perbedaan skala antar fitur. Normalisasi membantu menyamakan pengaruh setiap fitur dan data yang dinormalisasi membantu algoritma optimisasi bekerja lebih efisien dan stabil.
- Splitting Data (Train-Test Split), Dataset dibagi menjadi dua bagian, yaitu data latih (80%) dan data validasi (20%).
- Pembuatan Windowed Dataset, 
  - Data waktu diubah menjadi format yang cocok untuk pelatihan model menggunakan windowing.
  - Setiap window terdiri dari 𝑁past langkah waktu sebagai input dan 𝑁future langkah waktu sebagai target output.
  - Data yang telah diwindowing dan diacak diolah menjadi batch berukuran tetap (Batch=64).
  - Fungsi prefetch digunakan untuk mempercepat pengambilan data selama pelatihan.

## Modeling
### LSTM 
LSTM adalah varian dari Recurrent Neural Network (RNN) yang dirancang untuk menangkap hubungan temporal dalam data deret waktu. Model ini memiliki mekanisme forget gate, input gate, dan output gate untuk mengelola informasi yang relevan selama pelatihan.

**Parameter**
- N_FEATURES: Jumlah fitur input (jumlah kolom pada dataset). Dalam kasus ini, terdapat 6 fitur: CO, Ozone, PM10, PM25, NO2, dan Overall AQI Value.
- BATCH_SIZE: Ukuran batch yang digunakan dalam training adalah 64.
- N_PAST: Panjang jendela waktu historis yang digunakan untuk memprediksi masa depan, yaitu 7 (data dari 7 waktu sebelumnya digunakan).
- N_FUTURE: Panjang periode waktu yang diprediksi di masa depan, yaitu 7.
- SHIFT: Perpindahan antara waktu saat ini ke waktu target prediksi, yaitu 1 (prediksi mulai dari waktu berikutnya).
  
**Arsitektur Model**: 

```
model = tf.keras.models.Sequential([
    Conv1D(filters=32, 
            kernel_size=3,
            padding="causal",
            activation="relu",
            input_shape=[N_PAST, 6],
            ),
    Bidirectional(LSTM(32, return_sequences=True)),
    Dense(64, activation="relu"),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dropout(0.3),
    Dense(8, activation="relu"),
    Dropout(0.1),
    Dense(N_FEATURES)
])
```

- Conv1D:
  - Layer konvolusi 1D dengan 32 filter dan ukuran kernel 3.
  - Padding: Menggunakan "causal" untuk memastikan data di masa depan tidak bocor ke data historis.
  - Activation: Fungsi aktivasi ReLU digunakan untuk menangkap pola non-linear.
  - Input Shape: Data masukan memiliki bentuk [N_PAST, 6] (jendela waktu historis dan jumlah fitur).
- Bidirectional LSTM:
  - Layer LSTM (Long Short-Term Memory) yang bersifat bidirectional untuk menangkap pola dalam urutan data historis dari kedua arah.
  - return_sequences: Diatur ke True untuk menghasilkan urutan output dari setiap langkah waktu.
- Dense Layers:
  - Dense(64, activation="relu"): Layer padat dengan 64 unit dan fungsi aktivasi ReLU.
  - Dropout(0.2): Digunakan untuk mengurangi risiko overfitting dengan menghilangkan 20% unit secara acak.
  - Dense(32, activation="relu"): Layer padat kedua dengan 32 unit.
  - Dropout(0.3): Dropout tingkat kedua, menghilangkan 30% unit.
  - Dense(8, activation="relu"): Layer padat ketiga dengan 8 unit.
  - Dropout(0.1): Dropout tingkat ketiga, menghilangkan 10% unit.
- Output Layer (Dense):
  - Dense(N_FEATURES): Layer output dengan jumlah unit sama dengan jumlah fitur (6).

## Evaluation

Mean Absolute Error (MAE) adalah salah satu metrik evaluasi yang umum digunakan untuk mengukur kesalahan dalam model regresi. MAE menghitung rata-rata dari nilai absolut selisih antara nilai prediksi dengan nilai aktual.

Formula MAE:

$$
MAE = \frac{1}{n} \sum_{i=1}^{n} | y_i - \hat{y}_i |
$$

Keterangan:
- $$y_i$$ : Nilai aktual  
- $$\hat{y}_i\$$ : Nilai prediksi  
- $$n$$ : Jumlah sampel  

MAE memberikan gambaran langsung tentang rata-rata kesalahan prediksi model dalam satuan yang sama dengan data asli. Metrik ini mudah diinterpretasikan: semakin kecil nilai MAE, semakin baik kinerja model.

![train_val_loss](https://github.com/user-attachments/assets/5cee41f9-cfdc-4e8b-be99-dbb35874b0d4)

Training loss (MAE): 0.11522477120161057
Ini berarti bahwa kesalahan rata-rata model pada data pelatihan adalah sekitar 0.1152 unit. Nilai ini menggambarkan seberapa akurat model dalam memprediksi data pelatihan.

Validation loss (MAE): 0.12497211992740631
Ini menunjukkan kesalahan rata-rata model pada data validasi adalah sekitar 0.1250 unit. Nilai ini menunjukkan seberapa baik model dapat menggeneralisasi terhadap data yang tidak dilihat sebelumnya (data validasi).

Referensi:

[Predict Air Quality with Machine Learning | Science Project. (2015). Science Buddies. https://www.sciencebuddies.org/science-fair-projects/project-ideas/ArtificialIntelligence_p022/artificial-intelligence/air-quality](https://www.sciencebuddies.org/science-fair-projects/project-ideas/ArtificialIntelligence_p022/artificial-intelligence/air-quality)
