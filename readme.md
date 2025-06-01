# Laporan Proyek Machine Learning - Adrian Ramdhany

## Project Overview

Sistem rekomendasi telah menjadi tulang punggung dalam berbagai platform digital seperti Netflix, Spotify, YouTube, dan e-commerce. Dalam konteks industri hiburan Jepang, seperti anime, pengguna dihadapkan dengan ribuan judul yang terus bertambah setiap tahunnya. Hal ini membuat pencarian anime yang relevan dan sesuai minat menjadi tantangan tersendiri.

Proyek ini bertujuan untuk membangun sebuah sistem rekomendasi anime yang mampu menyaring dan merekomendasikan judul-judul yang sesuai dengan preferensi pengguna. Sistem ini memanfaatkan dua pendekatan utama yang umum digunakan dalam dunia *recommender system*, yaitu **Content-Based Filtering** dan **Collaborative Filtering**.

Dataset yang digunakan diambil dari [MyAnimeList Database via Kaggle](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database), yang berisi informasi lengkap mengenai ribuan judul anime serta jutaan entri rating dari pengguna.

## Business Understanding

### Problem Statements

* Bagaimana cara menyajikan rekomendasi anime yang relevan untuk setiap pengguna berdasarkan histori dan preferensi?
* Bagaimana cara memastikan bahwa sistem rekomendasi dapat beradaptasi terhadap berbagai jenis pengguna (baru/lama)?
* Sejauh mana sistem ini dapat meminimalisir kesalahan prediksi rekomendasi yang diberikan?

### Goals

* Mengembangkan sistem rekomendasi anime yang memberikan hasil **Top-N rekomendasi** berdasarkan dua pendekatan utama.
* Menganalisis kelebihan dan kekurangan masing-masing metode untuk membantu pengembangan sistem yang lebih kompleks ke depannya.

### Solution Approach

1. **Content-Based Filtering**
   Sistem akan menganalisis karakteristik konten dari anime yang pernah disukai atau ditonton pengguna (genre, tipe, dll) dan merekomendasikan anime dengan konten serupa.

2. **Collaborative Filtering (User-Based)**
   Sistem mencari kesamaan antara pengguna yang berbeda berdasarkan rating yang mereka berikan terhadap anime. Rekomendasi dibuat berdasarkan preferensi pengguna yang mirip.

---

## Data Understanding

### Struktur Dataset

* `anime.csv` (\~12.000 anime): Informasi judul, genre, rating, dll.
* `rating.csv` (\~7 juta entri): Rating pengguna terhadap berbagai anime.

### Kondisi Data

* Terdapat missing values pada kolom `genre`, `episodes`, dan `rating`.
* Nilai rating -1 digunakan sebagai indikator bahwa pengguna belum memberikan rating.

### Fitur Dataset

* `anime.csv`: `anime_id`, `name`, `genre`, `type`, `episodes`, `rating`, `members`
* `rating.csv`: `user_id`, `anime_id`, `rating`

### Visualisasi dan EDA

* **Distribusi Type Anime**
  ![1](https://github.com/user-attachments/assets/0dc33260-486f-4efd-a32b-1703fc18608e)

  > Anime dengan tipe TV merupakan yang terbanyak, mencapai 30.9% dari seluruh dataset.

* **Distribusi Rata-Rata Rating**
  ![2](https://github.com/user-attachments/assets/7a92033e-a383-45ae-a6ae-a19ddc6c1380)

  > Sebagian besar anime mendapatkan rating sekitar 7, menandakan adanya kecenderungan nilai tengah.

* **Anime dengan Komunitas Terbanyak**
  ![3](https://github.com/user-attachments/assets/6d0958bd-b615-4c4d-8bbb-f7c86ccea977)

  > Death Note menjadi anime dengan jumlah penonton terbanyak dalam dataset ini.

* **Anime dengan Rata-Rata Rating Tertinggi**
  ![4](https://github.com/user-attachments/assets/908a71ab-8ceb-46a9-9399-5cb84a34bf6e)

  > "Taka no Tsume 8: Yoshida-kun no X-Files" memiliki rating tertinggi meski tidak terlalu populer.

---

## Data Preparation

### Langkah-Langkah Pra-Pemrosesan:

* Menghapus rating = -1 (artinya user belum memberikan rating):

```python
ratings = ratings[ratings['rating'] != -1]
```

* Menghapus baris anime tanpa nama:

```python
anime.dropna(subset=['name'], inplace=True)
```

* Mengisi genre kosong dengan string kosong:

```python
anime['genre'] = anime['genre'].fillna('')
```

* Melakukan TF-IDF vectorization pada kolom genre:

```python
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(anime['genre'])
```

### Tujuan:

* Membersihkan data yang tidak valid.
* Menyusun format input yang siap digunakan untuk perhitungan similarity.
* Membuat representasi numerik dari genre yang dapat dihitung secara matematis.

---

## Modeling and Result

### Content-Based Filtering

Menggunakan pendekatan TF-IDF pada kolom `genre` dan cosine similarity antar vektor, sistem dapat memberikan rekomendasi anime serupa.

#### Top 10 Recommendations for 'Naruto':

| Rank | Name                                                                       |
| ---- | -------------------------------------------------------------------------- |
| 1    | Naruto: Shippuuden                                                         |
| 2    | Naruto                                                                     |
| 3    | Boruto: Naruto the Movie - Naruto ga Hokage ni...                          |
| 4    | Naruto x UT                                                                |
| 5    | Naruto: Shippuuden Movie 4 - The Lost Tower                                |
| 6    | Naruto: Shippuuden Movie 3 - Hi no Ishi wo Tsugu Mono                      |
| 7    | Naruto Shippuuden: Sunny Side Battle                                       |
| 8    | Naruto Soyokazeden Movie: Naruto to Mashin to Mittsu no Onegai Dattebayo!! |
| 9    | Kyutai Panic Adventure!                                                    |
| 10   | Naruto: Shippuuden Movie 6 - Road to Ninja                                 |

#### Genre Table:

| Name                                                  | Genre                                                 |
| ----------------------------------------------------- | ----------------------------------------------------- |
| Naruto: Shippuuden                                    | Action, Comedy, Martial Arts, Shounen, Super Power    |
| Naruto                                                | Action, Comedy, Martial Arts, Shounen, Super Power    |
| Boruto: Naruto the Movie - Naruto ga Hokage ni...     | Action, Comedy, Martial Arts, Shounen, Super Power    |
| Naruto x UT                                           | Action, Comedy, Martial Arts, Shounen, Super Power    |
| Naruto: Shippuuden Movie 4 - The Lost Tower           | Action, Comedy, Martial Arts, Shounen, Super Power    |
| Naruto: Shippuuden Movie 3 - Hi no Ishi wo Tsugu Mono | Action, Comedy, Martial Arts, Shounen, Super Power    |
| Naruto Shippuuden: Sunny Side Battle                  | Action, Comedy, Martial Arts, Shounen, Super Power    |
| Naruto Soyokazeden Movie: Naruto to Mashin to...      | Action, Comedy, Martial Arts, Shounen, Super Power    |
| Kyutai Panic Adventure!                               | Action, Martial Arts, Shounen, Super Power            |
| Naruto: Shippuuden Movie 6 - Road to Ninja            | Action, Adventure, Martial Arts, Shounen, Super Power |

### Collaborative Filtering (User-Based)

Menggunakan user-anime rating matrix untuk menghitung kesamaan antar pengguna.

* Cosine similarity antar pengguna
* Prediksi rating untuk anime yang belum ditonton
* Memberikan Top-N rekomendasi

#### Output untuk User ID = 1:

**Collaborative Filtering | Top 10 Recommendations:**

| Rank | Anime Name                       |
| ---- | -------------------------------- |
| 1    | Hamster Club                     |
| 2    | Play Ball 2nd                    |
| 3    | Live On Cardliver Kakeru         |
| 4    | Gozonji! Gekkou Kamen-kun        |
| 5    | Ryoujoku Joshi Gakuen            |
| 6    | Midoriyama Koukou Koushien-hen   |
| 7    | Chargeman Ken!                   |
| 8    | Nono-chan                        |
| 9    | Asari-chan: Ai no Marchen Shoujo |
| 10   | Seton Doubutsuki                 |

### Analisis

* **Content-Based**: Lebih cocok untuk user baru atau ketika hanya ada data rating dari 1 user. Tidak bergantung pada komunitas.
* **Collaborative**: Mampu menangkap preferensi komunitas, tetapi memerlukan data pengguna yang cukup besar agar efektif.

---

## Evaluation

### Metrik: Root Mean Squared Error (RMSE)

RMSE digunakan untuk mengukur seberapa jauh prediksi sistem terhadap nilai rating aktual.

#### Formula:

$$
\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2}
$$

Semakin kecil nilai RMSE, maka semakin baik prediksi yang diberikan.

### Hasil Evaluasi

* **RMSE** pada subset collaborative filtering (1000 sample): **1.3930**

### Interpretasi:

* Nilai RMSE sekitar 1.39 menunjukkan bahwa prediksi rating berbeda sekitar 1.39 poin dari nilai sebenarnya. Ini masih termasuk toleransi yang bisa diterima dalam sistem rekomendasi dasar.

---

## Referensi

1. Aggarwal, Charu C. (2016). *Recommender Systems: The Textbook*. Springer.
2. MyAnimeList Dataset: [https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database)
3. Scikit-learn documentation: [https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine\_similarity.html](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html)
4. TF-IDF Vectorization: [https://en.wikipedia.org/wiki/Tf%E2%80%93idf](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
5. Surprise Library for Collaborative Filtering: [https://surprise.readthedocs.io/](https://surprise.readthedocs.io/)
