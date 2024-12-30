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
- Pemrosesan Data: Melakukan normalisasi menggunakan metode Min-Max Scaling untuk memastikan semua parameter berada dalam skala yang sama.
Membagi data menjadi subset pelatihan dan validasi untuk menghindari overfitting.
- Pemilihan Model: Menggunakan model Neural Networks (CNN + LSTM) untuk menangkap pola temporal.
Memanfaatkan teknik dropout untuk menghindari overfitting.
- Evaluasi Model: Menggunakan Mean Absolute Error (MAE) sebagai metrik evaluasi utama.
Membandingkan prediksi model dengan data aktual melalui visualisasi.

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
![Image](outlier.png)

Berdasarkan gambar diatas, terdapat outlier yang ditunjukkan dengan adanya titik-titik yang berada di luar batas whisker (garis vertikal yang menunjukkan jangkauan data non-outlier) pada masing-masing boxplot untuk setiap fitur, seperti CO, Ozone, PM10, PM25, NO2, dan Overall AQI Value. Outlier ini menunjukkan data yang berada jauh dari rentang nilai mayoritas.
### Data Correlation
![Image](corr_feature.png)

## Data Preparation
- Penanganan Missing Values, penanganan Missing Values dilakukan dengan menggunakan teknik interpolasi berdasarkan waktu  (`method='time'`) memanfaatkan informasi dari indeks waktu dalam dataset untuk memperkirakan nilai yang hilang secara linier atau sesuai pola temporal data. Missing values dapat memengaruhi hasil model, terutama pada metode pembelajaran mesin yang sensitif terhadap data yang tidak lengkap.
- Menangani Outlier, penanganan outlier dilakukan dengan menggunakan teknik clipping berbasis percentile. Metode clipping memotong nilai-nilai pada data yang berada di luar rentang tertentu sehingga tidak melebihi batas bawah (lower bound) dan batas atas (upper bound). Penanganan outlier dilakukan untuk mencegah overfitting, dimana outlier dapat menyebabkan model menjadi terlalu kompleks karena mencoba menyesuaikan data yang tidak representatif.
- Normalisasi Data, Data dinormalisasi menggunakan rumus berikut:

$\text{normalized_value} = \frac{value-min}{max-min}$ ​ 
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

**Arsitektur Model**: 

Model: "sequential"

| Layer (type)                  | Output Shape  | Param # | 
|-------------------------------|---------------|---------|
| conv1d (Conv1D)               | (None, 7, 32) | 608     |  
| bidirectional (Bidirectional) | (None, 7, 64) | 6640    |                                                           
| dense (Dense)                 | (None, 7, 64) | 4160    |  
| dropout (Dropout)             | (None, 7, 64) | 0       |  
| dense_1 (Dense)               | (None, 7, 32) | 2080    |  
| dropout_1 (Dropout)           | (None, 7, 32) | 0       |  
| dense_2 (Dense)               | (None, 7, 8)  | 264     |  
| dropout_2 (Dropout)           | (None, 7, 8)  | 0       |  
| dense_3 (Dense)               | (None, 7, 6)  | 54      |  

## Evaluation

![Image](train_val_loss.png)

Mean Absolute Error (MAE) adalah salah satu metrik evaluasi yang umum digunakan untuk mengukur kesalahan dalam model regresi. MAE menghitung rata-rata dari nilai absolut selisih antara nilai prediksi dengan nilai aktual.

Formula MAE:

$$
MAE = \frac{1}{n} \sum_{i=1}^{n} | y_i - \hat{y}_i |
$$

Keterangan:
- \( y_i \): Nilai aktual  
- \( \hat{y}_i \): Nilai prediksi  
- \( n \): Jumlah sampel  

MAE memberikan gambaran langsung tentang rata-rata kesalahan prediksi model dalam satuan yang sama dengan data asli. Metrik ini mudah diinterpretasikan: semakin kecil nilai MAE, semakin baik kinerja model.

Referensi:

[Predict Air Quality with Machine Learning | Science Project. (2015). Science Buddies. https://www.sciencebuddies.org/science-fair-projects/project-ideas/ArtificialIntelligence_p022/artificial-intelligence/air-quality](https://www.sciencebuddies.org/science-fair-projects/project-ideas/ArtificialIntelligence_p022/artificial-intelligence/air-quality)
