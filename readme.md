# Laporan Proyek Machine Learning - Adrian Ramdhany

## Project Overview

Sistem rekomendasi telah menjadi tulang punggung dalam berbagai platform digital seperti Netflix, Spotify, YouTube, dan e-commerce. Dalam konteks industri hiburan Jepang, seperti anime, pengguna dihadapkan dengan ribuan judul yang terus bertambah setiap tahunnya. Hal ini membuat pencarian anime yang relevan dan sesuai minat menjadi tantangan tersendiri.

Proyek ini bertujuan untuk membangun sebuah sistem rekomendasi anime yang mampu menyaring dan merekomendasikan judul-judul yang sesuai dengan preferensi pengguna. Sistem ini memanfaatkan dua pendekatan utama yang umum digunakan dalam dunia *recommender system*, yaitu **Content-Based Filtering** dan **Collaborative Filtering**.

Dataset yang digunakan diambil dari [MyAnimeList Database via Kaggle](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database), yang berisi informasi lengkap mengenai ribuan judul anime serta jutaan entri rating dari pengguna.

## Business Understanding

### Problem Statements

* Bagaimana cara menyajikan rekomendasi anime yang relevan untuk setiap pengguna berdasarkan histori dan preferensi?
* Bagaimana cara memastikan bahwa sistem rekomendasi dapat beradaptasi terhadap berbagai jenis pengguna (baru/lama)?

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

- `anime.csv`: 12.294 baris × 7 kolom
- `rating.csv`: 7.813.737 baris × 3 kolom

### Kondisi Data

- Kolom `genre` memiliki missing value dan diisi dengan string kosong.
- Kolom `episodes` tidak memiliki missing value.
- Rating `-1` menandakan user belum memberikan rating.

### Fitur Dataset

* `anime.csv`: `anime_id`, `name`, `genre`, `type`, `episodes`, `rating`, `members`
* `rating.csv`: `user_id`, `anime_id`, `rating`

# Deskripsi DataFrame Anime

DataFrame ini berisi informasi mengenai anime dengan total 12.294 baris dan 7 kolom sebagai berikut:

| No | Nama Kolom | Tipe Data | Jumlah Data Tidak Kosong | Keterangan                |
|----|------------|-----------|-------------------------|---------------------------|
| 0  | anime_id   | int64     | 12.294                  | ID unik untuk setiap anime |
| 1  | name       | object    | 12.294                  | Nama anime                |
| 2  | genre      | object    | 12.232                  | Genre anime, terdapat 62 nilai kosong |
| 3  | type       | object    | 12.269                  | Jenis anime (TV, Movie, dll), ada 25 nilai kosong |
| 4  | episodes   | object    | 12.294                  | Jumlah episode, disimpan sebagai string |
| 5  | rating     | float64   | 12.064                  | Rating anime, ada 230 nilai kosong |
| 6  | members    | int64     | 12.294                  | Jumlah anggota yang mengikuti anime |

## Penjelasan:

- **anime_id** merupakan kolom numerik bertipe integer yang berfungsi sebagai identifikasi unik setiap anime.
- **name** adalah nama anime dalam bentuk teks (string).
- **genre** berisi genre anime dan memiliki beberapa nilai kosong (missing values), sehingga perlu penanganan jika ingin diolah lebih lanjut.
- **type** menunjukkan jenis anime (seperti TV, OVA, Movie, dll) dan juga memiliki beberapa nilai kosong.
- **episodes** disimpan dalam tipe data objek (string), ini biasanya karena beberapa anime mungkin memiliki format episode yang tidak berupa angka tetap (misal "Unknown", "Special").
- **rating** adalah nilai rating berupa angka desimal (float), yang juga memiliki beberapa data yang hilang.
- **members** menunjukkan jumlah anggota atau pengguna yang mengikuti atau menonton anime tersebut, berupa integer.

# Deskripsi DataFrame Rating Anime

DataFrame ini berisi data rating anime oleh pengguna dengan total 7.813.737 baris dan 3 kolom sebagai berikut:

| No | Nama Kolom | Tipe Data | Keterangan                        |
|----|------------|-----------|---------------------------------|
| 0  | user_id    | int64     | ID unik pengguna yang memberi rating |
| 1  | anime_id   | int64     | ID unik anime yang diberi rating |
| 2  | rating     | int64     | Nilai rating yang diberikan pengguna |

## Penjelasan:

- **user_id** adalah kolom numerik bertipe integer yang mengidentifikasi setiap pengguna secara unik.
- **anime_id** merupakan kolom integer yang menunjukkan anime yang dirating oleh pengguna.
- **rating** adalah nilai rating dalam bentuk integer yang diberikan oleh pengguna pada anime tersebut.


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

1. **Menghapus entri anime tanpa nama**:
```python
anime.dropna(subset=['name'], inplace=True)
```

Beberapa entri dalam dataset `anime.csv` tidak memiliki judul (name). Karena judul anime penting sebagai identitas utama dalam sistem rekomendasi (terutama pada Content-Based Filtering), maka baris-baris yang tidak memiliki nama dihapus untuk menjaga konsistensi dan integritas data.

2. **Menghapus rating tidak sah (rating = -1)**:
```python
ratings = ratings[ratings['rating'] != -1]
```

Dalam dataset `rating.csv`, rating `-1` menandakan bahwa pengguna belum memberikan rating yang sebenarnya. Karena nilai ini tidak mencerminkan preferensi nyata, maka baris-baris tersebut dihapus agar tidak mengganggu perhitungan kesamaan atau prediksi rating.

3. **Mengisi genre kosong dengan string kosong**:
```python
anime['genre'] = anime['genre'].fillna('')
```

`Genre` yang kosong akan menyebabkan error saat dilakukan proses vektorisasi dengan TF-IDF. Oleh karena itu, nilai `NaN` pada kolom `genre` diganti dengan string kosong (`''`) agar tetap bisa diproses secara tekstual.

4. **TF-IDF vectorization pada kolom genre**:
```python
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(anime['genre'])
```

`TF-IDF` digunakan untuk mengubah data genre yang berupa teks menjadi representasi numerik. Bobot yang diberikan oleh TF-IDF mencerminkan seberapa penting suatu genre dalam konteks keseluruhan dataset. Vektor inilah yang kemudian digunakan untuk menghitung kemiripan antar anime menggunakan cosine similarity.

5. **Membentuk matriks user-anime**:
```python
user_anime_matrix = ratings.pivot_table(index='user_id', columns='anime_id', values='rating')
user_anime_matrix.fillna(0, inplace=True)
```

Langkah ini menyusun data rating menjadi sebuah matriks dua dimensi (pivot table) di mana baris adalah pengguna dan kolom adalah anime. Matriks ini digunakan dalam Collaborative Filtering. Nilai `NaN` diisi dengan nol untuk menyatakan bahwa pengguna belum menilai anime tersebut.


6. **Membagi data untuk evaluasi Content-Based**:
```python
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)
```

Dataset dibagi menjadi data latih dan data uji. Hal ini penting agar sistem dapat dievaluasi secara objektif. Data latih digunakan untuk membangun model, sedangkan data uji digunakan untuk menghitung performa sistem seperti Precision@10 dan Recall@10.

---

### Tujuan:

* Membersihkan data yang tidak valid.
* Menyusun format input yang siap digunakan untuk perhitungan similarity.
* Membuat representasi numerik dari genre yang dapat dihitung secara matematis.
* Mempersiapkan data untuk pengujian dan mengetahui hasil performa.

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

### Metrik Evaluasi

1. **Content-Based Filtering**  
   Menggunakan metrik **Precision@10** dan **Recall@10** untuk mengukur seberapa relevan hasil rekomendasi dibanding anime yang benar-benar ditonton user dalam data uji.

   #### Hasil:
   * **Average Precision@10**: *0.0158*  
   * **Average Recall@10**: *0.0177*

   **Interpretasi:**  
   - Precision@10 sebesar 1.58% berarti dari 10 anime yang direkomendasikan, rata-rata sekitar 0.158 di antaranya relevan atau ditonton oleh pengguna.
   - Recall@10 sebesar 1.77% menunjukkan bahwa dari semua anime yang seharusnya direkomendasikan, hanya 1.77% yang berhasil ditangkap oleh sistem dalam 10 rekomendasi teratas.

2. **Collaborative Filtering (User-Based)**  
   Menggunakan **Root Mean Squared Error (RMSE)** untuk mengukur deviasi prediksi terhadap rating aktual.

   #### Formula:
   $$
   \text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2}
   $$

   #### Hasil:
   * **RMSE** pada subset collaborative filtering (1000 sample): **1.3930**

   **Interpretasi:**  
   Nilai RMSE sekitar 1.39 menandakan bahwa rata-rata prediksi sistem berbeda sekitar 1.39 poin dari rating sebenarnya — cukup wajar untuk sistem rekomendasi dasar.

## Referensi

1. Aggarwal, Charu C. (2016). *Recommender Systems: The Textbook*. Springer.
2. MyAnimeList Dataset: [https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database)
3. Scikit-learn documentation: [https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine\_similarity.html](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html)
4. TF-IDF Vectorization: [https://en.wikipedia.org/wiki/Tf%E2%80%93idf](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
5. Surprise Library for Collaborative Filtering: [https://surprise.readthedocs.io/](https://surprise.readthedocs.io/)
