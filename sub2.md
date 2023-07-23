# Laporan Proyek Machine Learning - Muhammad Faturachman Atthaariq  
  

## Domain Proyek  
Anime (アニメ) adalah tayangan video animasi dari jepang yang memiliki banyak penonton dari seluruh dunia. Kepopuleran ini karena penggunaan animasi yang menarik dan cerita dengan beragam genre, baik fiksi dan non-fiksi. Anime memiliki visual yang atraktif untuk menarik penonton, terutama di kalangan remaja hingga dewasa.

Menurut survey dari Crunchyroll, hanya 6% kalangan "Gen Z" yang tidak mengetahui tentang anime [[1]](https://www.animenewsnetwork.com/interest/2021-07-09/crunchyroll-market-research-only-6-percent-of-gen-z-dont-know-what-anime-is/.174962).

Kepopuleran ini mendorong berbagai penyedia layanan tayangan *online* atau *streaming* mencoba menyediakan berbagai anime untuk menarik lebih banyak penonton. Salah satu penyedia *streaming* film yaitu Netflix mencatat lebih dari 50% pengguna global mereka setidaknya pernah menonton satu judul anime [[2]](https://www.animenewsnetwork.com/daily-briefs/2022-03-30/netflix-more-than-half-of-members-globally-watched-anime-last-year/.184167).

Ketika seseorang menonton anime dan merasa cocok dengan anime tersebut, seringkali penonton ingin mencari judul anime lain yang mirip atau serupa dengan anime yang cocok tersebut. Apa yang akan dilakukan mereka? Secara manual, mereka harus mencari sendiri judul anime dengan melihat *genre*, rating, atau membaca review dari berbagai anime untuk melihat mana judul yang mungkin cocok dengan mereka.

Dari hal tersebut, sebuah *tool* rekomendasi diperlukan agar pengguna dapat melihat judul anime lain yang mirip atau alternatif tayangan yang baru namun tidak jauh berbeda dengan anime yang telah ditonton. Rekomendasi diharapkan dapat menaikkan jumlah tayangan anime secara signifikan karena rekomendasi dapat mendorong pengguna untuk terus menonton anime baru. Rekomendasi dapat berdasarkan dari popluaritas dan *rating* anime atau memberi rekomendasi anime yang mirip dari anime yang sedang ditonton pengguna.

## Business Understanding  

### Problem Statements  

Berdasarkan permasalahan dari latar belakang, masalah yang akan dijawab adalah:  
1. Bagaimana membuat rekomendasi anime berdasarkan popularitasnya?
2. Bagaimana membuat rekomendasi anime dari judul anime yang sedang ditonton?

### Goals  
Tujuan proyek ini adalah menghasilkan rekomendasi judul anime berdasarkan popularitas & rating atau judul anime yang sedang ditonton melalui pendekatan machine learning.

### Solution Statements  
  1. Membuat rekomendasi anime dengan konsep *popularity-based*.
  2. Membuat model rekomendasi anime dari judul anime yang sedang ditonton dengan *machine learning* melalui dua pendekatan:
	  - *Collaborative filtering*
	  - *Content-based filtering*

## Data Understanding  

Data yang digunakan pada proyek ini didapatkan dari Kaggle pada tautan berikut ini *[Anime Recommendations Database](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database)*.

Ada 2 *dataset* yang dipakai yaitu:
* **anime.csv**: Data yang berisi daftar seluruh anime.
* **rating.csv**: Data yang berisi daftar rating anime yang diberikan pengguna

Variabel-variabel pada data *Anime Recommendations Database* adalah sebagai berikut: 

Untuk **anime.csv**:
* anime_id: berisi id anime yang diambil dari situs myanimelist.net
* name: judul lengkap anime
* genre: genre anime, masing-masing dipisah dengan koma
* type: tipe anime, seperti movie, OVA, TV
* episodes: banyak episode pada anime tersebut (1 jika tipe movie)
* rating: rating rata-rata untuk anime (rentang 0-10)
* members: banyak pengguna yang menonton anime ini

Untuk **rating.csv**:
* user_id: id untuk mengidentifikasi pengguna (bersifat anonim & angka random)
* anime_id: judul anime yang diberi rating oleh pengguna
* rating: nilai rating yang diberikan pengguna dengan rentang 1-10, dan -1 jika sudah menonton tetapi tidak memberikan rating

Selanjutnya, akan dilakukan analisis pada *dataset* untuk melihat pola karakteristik data yang akan digunakan.

### Data Understanding & Analysis
Pada **anime.csv**:
1. Terdapat 12294 baris data dalam *dataset* dengan 7 kolom.
2. Terdapat 3 kolom numerik dari awal, yaitu 1 kolom numerik kontinyu seperti 'rating' dan 2 kolom numerik diskrit seperti 'anime_id' & 'members'.  
3. Sebanyak 1 kolom data lainnya awalnya merupakan kolom kategorikal yaitu 'type'.
4. Kolom 'episodes' sebenarnya merupakan kolom numerik diskrit, namun terdapat data lain seperti 'Unknown'.
5. Kolom 'name' memiliki 12292 ragam data, sedangkan 'genre' memiliki 3264 ragam data.
6. Kolom 'tipe' memiliki 6 ragam data, sedangkan 'episodes' memiliki 187 ragam data.
7. Tidak ada duplikasi kolom pada *dataset* ini.
8. Terdapat nilai null / `NaN`  pada kolom 'genre', 'type', dan 'rating'.
9. Hapus semua nilai null dengan `dropna()`
10. Setelah nilai null dihapus, *dataset* memiliki 12017 baris data.
11. Keragaman data pada *dataset* ini menjadi:
	- `anime_id`: 12017
	- `name`: 12105
	- `genre`: 3229
	- `type`: 6
	- `episodes`: 187
	- `rating`: 598
	- `members`: 6596
 12. Korelasi antar data sebagai berikut:
![Korelasi dataset anime](https://i.ibb.co/QmG5vY7/corr-anime.png)
*Heatmap* menunjukkan korelasi tertinggi pada kolom 'rating' dan 'members' dengan nilai 0.39, sedangkan kolom yang lain memiliki korelasi negatif.

Untuk **rating.csv**:

 1. Terdapat 7813737 baris data dalam *dataset* dengan 3 kolom.
 2. Semua kolom merupakan data numerik diskrit.
 3. Ragam data pada 'user_id' adalah 73515 data, pada 'anime_id' adalah 11200 data, dan pada 'rating' adalah 11 data.
 4. 'rating' memiliki rentang data dari 1-10 untuk nilai rating serta -1 pada anime yang ditonton tetapi tidak diberi rating.
 5. Tidak ada nilai null / `Nan` pada *dataset* ini.
 6. Terdapat duplikasi 1 kolom pada *dataset* ini, sehingga harus di-*drop* dengan `drop_duplicated()` terlebih dahulu.
 7. Korelasi antar data sebagai berikut:
![Korelasi dataset rating](https://i.ibb.co/Wsftdg4/corr-rating.png)
*Heatmap* tidak menunjukkan korelasi yang berarti pada seluruh kolom.

### Create Dataset for Recommender System
Kedua *dataset* digabungkan dengan `merge()` berdasarkan kolom `anime_id` dan diberi nama `df`.
Hasilnya adalah *dataset* dengan 9 kolom, di mana terdapat kolom `rating_x` yang berisi rating rata-rata rating anime dari *dataset* `anime` & `rating_y` yang berisi rating dari *dataset* `rating` yaitu rating pengguna.
Kemudian kolom`rating_y` dihapus dan `rating_x` diubah namanya menjadi `user_rating` karena nilai rating rata-rata ini yang akan digunakan sebagai nilai rating setiap anime pada tiap pengguna.

Kolom `rating_y` tidak dipakai karena terdapai nilai `-1` (sudah ditonton tetapi belum diberi rating), hal ini akan mengganggu rentang nilai yang dipakai pada kolom rating, yaitu dari 1 sampai 10, yang jika dihapus, maka akan kehilangan lebih banyak data pengguna yang telah menonton anime. Nilai `-1` pada `rating_y`  tergantikan oleh `rating_x`, sehingga penggunaan `rating_x` tidak akan menghilangkan data lebih banyak.

### Data Preparation
#### Handling Missing Values
* Cek nilai `NaN` dengan `isna()`. Hasilnya tidak ada nilai `NaN` pada *dataset* `df`.
#### Cleaning Text
* Mengecilkan teks (*lowercase*).
* Menghapus simbol dan  _special characters_  dalam teks dengan  `re.sub()`
* Menghapus tanda baca (*punctuation*) dengan  `translate()`.

*Cleaning* dilakukan pada *dataset* `anime` dan `df`.

## Modelling  & Result

Tahapan ini berisi pembuatan model untuk rekomendasi anime yang terdiri dari:
1. ***Popularity-based***: Rekomendasi berdasarkan popularitas yang dilihat dari rata-rata rating anime oleh pengguna, dengan tujuan untuk kategori tertentu.
2.  Model rekomendasi dengan pendekatan ***Clustering*** yang dipadukan dengan basis rekomendasi:
	- *Collaborative Filtering*
	- *Content-based Filtering*

### Popularity-based Recommender
Rekomendasi dengan *popularity-based* dilakukan melalui tahapan berikut:
* Mengelompokkan data & mencari nilai rata-rata anime dengan `groupby()` & `agg()`.
* Kemudian, rata-rata tersebut diurutkan dengan `sort_values()`.
* Setelah itu, akan ditampilkan 10 data teratas dengan `head(10)`.

Dari sini, akan dilihat beberapa daftar rekomendasi berdasarkan parameter berikut:
* 10 anime dengan rating teratas (tanpa melihat jenis anime):

| No | ID anime    | Nama                                              | Rating |
|----|-------|---------------------------------------------------|--------|
| 1  |  6469 |                                  mogura no motoro |   9.50 |
| 2  |  4919 |                                     kimi no na wa |   9.37 |
| 3  |  2582 |                   fullmetal alchemist brotherhood |   9.26 |
| 4  |  3033 |                                          gintama° |   9.25 |
| 5  | 10764 |                    yakusoku africa mizu to midori |   9.25 |
| 6  |  9318 |                                        steinsgate |   9.17 |
| 7  |  3292 | haikyuu karasuno koukou vs shiratorizawa gakue... |   9.15 |
| 8  |  3919 |                              hunter x hunter 2011 |   9.13 |
| 9  |  2972 |                              ginga eiyuu densetsu |   9.11 |
| 10 |  3023 |                                 gintama enchousen |   9.11 |

* 10 anime dengan penonton terbanyak kemudian diurutkan dari rating tertinggi:

| No | ID anime | Penonton | Rating |
|---|---|---|---|
| 1 | 6268 | 200630 | 9.37 |
| 2 | 6479 | 793665 | 9.26 |
| 3 | 6016 | 114262 | 9.25 |
| 4 | 6475 | 673572 | 9.17 |
| 5 | 6152 | 151266 | 9.16 |
| 6 | 5879 | 93351 | 9.15 |
| 7 | 6440 | 425855 | 9.13 |
| 8 | 5760 | 80679 | 9.11 |
| 9 | 5768 | 81109 | 9.11 |
| 10 | 5680 | 72534 | 9.10 |

* Rating tertinggi berdasarkan *Genre* anime:

| No | ID genre | Genre Pertama | Rating |
|---|---|---|---|
| 1 | 14 | Josei | 8.463213 |
| 2 | 28 | Sci-Fi | 8.352921 |
| 3 | 24 | Psychological | 8.210347 |
| 4 | 21 | Mystery | 8.199439 |
| 5 | 4 | Dementia | 7.831614 |
| 6 | 6 | Drama | 7.791295 |
| 7 | 9 | Game | 7.741392 |
| 8 | 0 | Action | 7.715880 |
| 9 | 1 | Adventure | 7.676925 |
| 10 | 2 | Cars | 7.661395 |

* Rating tertinggi berdasarkan jenis anime:

| No | Genre Pertama | Rating |
|---|---|---|
| 1 | Movie | 7.809518 |
| 2 | TV | 7.738967 |
| 3 | Special | 7.330543 |
| 4 | OVA | 7.196094 |
| 5 | ONA | 7.053085 |
| 6 | Music | 7.032784 |

### Title-based Recommender
Rekomendasi berdasarkan judul anime yang sedang ditonton pengguna dibangun menjadi 2 pendekatan, yaitu **Clustering + Collaborative Filtering** dan **Clustering + Content-based Filtering**. 
#### 1. Clustering dengan K-Means
Clustering dilakukan menggunakan K-Means, satu diantara algoritma yang populer untuk mengelompokkan n data ke dalam klaster tertentu berdasarkan jarak nilai data terhadap pusat klaster (*centroid*). Tahapannya sebagai berikut:
 1. *pre-processing dataset* `df` yaitu proses *label encoding* menggunakan `LabelEncoder()` pada kolom `t_genre` dan `t_type`.
 2. Model K-Means dibuat dengan `KMeans()` dan parameter:
	* n_clusters: 6 klaster
	* random_state: 42
 3. Kemudian model di-*fit* dengan data `anime_id`, `t_genre`, `t_type`, dan `user_rating` menggunakan `fit_predict()` dan nilai klaster tersebut dimasukkan dalam *dataset* `df`.
 
 #### 2. Set random user & anime title to recommend
Pengguna yang akan diberikan rekomendasi akan dipilih secara acak dari *dataset*. Dari pengguna itu, akan dipilih satu judul anime secara random sebagai bahan untuk melakukan rekomendasi. Hal ini dilakukan untuk menyamakan keadaan awal ketika rekomendasi diberikan melalui dua pendekatan berbeda. Tahapannya adalah sebagai berikut:

1. Membuat indeks random pengguna
2. Membuat *pivot table* dari tiap kluster dengan parameter:
	* index: "name"
	* columns: "user_id"
	* values: "user_rating"
3. Memilih judul anime secara acak dari *pivot table*.

Dalam *running* ini terpilih anime dengan *query* **2810**, berjudul **yoroiden samurai troopers kikoutei densetsu**.

#### 3a. Pendekatan Collaborative Filtering
Collaborative Filtering merupakan pendekatan rekomendasi yang melibatkan parameter-parameter yang secara bersama-sama (kolaboratif) mempengaruhi hasil rekomendasi. *Collaborative filtering* terbagi menjadi 2 jenis, yaitu *model based* dan yang akan dipakai yaitu *memory based*.

Jenis *model based* dapat dipecah menurut tipe pendekatannya, yaitu *user based* dan *item based*. Penelitian ini akan menggunakan *item based* filtering karena karakteristik pada item cenderung tetap, sedangkan karakteristik dari kebiasaan pengguna dapat berubah sewaktu-waktu [[3]](https://towardsdatascience.com/prototyping-a-recommender-system-step-by-step-part-1-knn-item-based-collaborative-filtering-637969614ea).

Rekomendasi dengan pendekatan *item based* akan melihat berbagai judul anime sebagai parameter yang mempengaruhi karena karakteristiknya tetap, bukan dari kebiasaan penonton anime yang dapat berubah secara tiba-tiba. Untuk itu, penelitian ini akan menggunakan algoritma K-Nearest Neighbors (KNN). KNN melihat kemiripan antar item (anime) dan menghitung jarak item dengan item terdekat, lalu akan diurutkan dari yang terdekat.

Implementasi KNN dilakukan menggunakan *library* `sklearn` dengan `NearestNeighbors`. KNN yang dipakai memiliki parameter berikut:
- `metric`: `"cosine"`, yaitu metode penghitungan jarak dengan *Cosine Similarity*
- `algorithm`: `"auto"`, yaitu pemilihan algoritma *Nearest Neightbor* otomatis.

Setelah itu, model di-*fit* pada data pivot dengan inputan *query* judul anime yang akan diberikan rekomendasi dan memberikan parameter `n_neighbors` dengan nilai 11.

Selanjutnya output akan dihasilkan dengan mengisi list `no`, `name`, `rating`, `distance`, dan `genre` yang menghasilkan *top-N recommendation* yaitu maksimal 5 judul rekomendasi. 
 
| No | Anime Name | Rating | Genre | Similarity |
|---|---:|---:|---:|---:|
| 1 | yoroiden samurai troopers gaiden | 6.86 | Adventure, Fantasy, Samurai, Shounen | 0.860990 |
| 2 | yoroiden samurai troopers message | 6.13 | Adventure, Fantasy, Samurai, Shounen | 0.859938 |
| 3 | yoroiden samurai troopers | 7.21 | Adventure, Samurai, Sci-Fi, Shounen | 0.858342 |
| 4 | doramichan minidora sos | 6.49 | Fantasy, Kids | 0.853984 |
| 5 | doramichan a blue straw hat | 5.28 | Fantasy, Kids | 0.852685 |

#### 3b. Pendekatan Content-based FIltering

Pendekatan *content-based* filtering merupakan rekomendasi yang melihat fitur-fitur penting pada item yang akan dicari rekomendasinya, lalu melihat kemiripan fitur tersebut pada item lain, dalam hal ini melihat fitur pada tiap anime.

*Content-based filtering* dilakukan dengan melihat TF-IDF (*Term Frequency Inverse Document Frequency*), yaitu perhitungan untuk melihat seberapa penting sebuah kata (dalam hal ini fitur dalam tiap anime) pada judul anime tersebut untuk dilihat kemiripannya dengan TF-IDF pada judul anime lain.

Implementasi *content-based filtering* dilakukan dengan mengambil klaster hasil *K-Means* yang disimpan dalam `rec_data`. Kemudian TF-IDF digunakan pada data genre setiap anime dengan `TfidfVectorizer` dengan parameter:
- `analyzer`: `"word"`, yaitu kata sebagai fitur. Data genre pada `rec_data`  kemudian di-*fit* pada TF-IDF. Setelah itu, Hasil TF-IDF, akan dilihat kemiripannya menggunakan *Cosine Similarity*

Selanjutnya, untuk melihat rekomendasi pada judul anime tertentu, fungsi rekomendasi dibuat dengan membandingkan TF-IDF anime tersebut pada seluruh anime dan diurutkan hasilnya dari yang memiliki nilai *cosine similarity* tertinggi. Berikut hasil *top-N recommendation* yang dihasilkan sebagai berikut:

| No | Anime Name | Rating | Genre | Similarity |
|---|---:|---:|---:|---:|
| 1 | mujin wakusei survive | 7.72 | Action, Adventure, Fantasy, Sci-Fi, Slice of Life | 0.782989 |
| 2 | kamichu | 7.51 | Comedy, Drama, Slice of Life, Supernatural | 0.701465 |
| 3 | detective conan ova 08 high school girl detect... | 7.30 | Comedy, Mystery, Police, Shounen | 0.647034 |
| 4 | flcl | 8.06 | Action, Comedy, Dementia, Mecha, Parody, Sci-Fi | 0.403561 |
| 5 | ginga densetsu weed | 7.31 | Adventure, Drama, Shounen | 0.344954 |

## Evaluation  
###  *Cosine Similarity*  
Nilai kemiripan pada masing-masing pendekatan rekomendasi diatur dengan *cosine similarity*. Hasil berupa *top-N recommendation* dilihat dari kedua pendekatan tersebut.
![cosine](https://i.ibb.co/F8V2tV4/cosinesimilar.webp)

*Cosine similarity* melihat nilai kosinus dari sudut dua posisi item yang akan dibandingkan. semakin kecil sudut yang dihasilkan, maka nilai kosinus akan mendekati 1 dan kemiripan 100% (*similar vectors*). Jika sudut yang dihasilkan mendekati  90°, maka kosinus akan mendekati nilai 0 dan kemiripan 0% (*orthogonal vectors*). Jika sudut mendekati 180°, maka kosinus akan mendekati nilai -1 dan kemiripan -100% (*opposite vectors*).

### Pembahasan  

Dari hasil *top-N recommendation* pada kedua pendekatan rekomendasi, *Collaborative filtering* menghasilkan *top-5* dengan rata-rata kemiripan sebesar 85.6% dan rekomendasi tertinggi pada anime berjudul **"yoroiden samurai troopers gaiden"** dengan nilai 86%.

*Content-based filtering* memiliki nilai rata-rata kemiripan pada *top-5* yang lebih rendah, yaitu 57.6% dan rekomendasi tertinggi pada judul **"mujin wakusei survive"** dengan nilai 78.3%.

Berikut tabel **nilai similaritas / kemiripan** dari *Top-5* kedua pendekatan:
| Top-5 | Content-based | Collaborative |
|---|---:|---:|
| 1 | 78.3% | 86% |
| 2 | 70.5% | 85.99% |
| 3 | 64.7% | 85.8% |
| 4 | 40.4% | 85.4% |
| 5 | 34.5% | 85.3% |
| Rata-rata | 57.6% | **85.6%** |

Ternyata, rata-rata rating anime yang direkomendasikan pada *content-based filtering* yaitu 7.58, lebih tinggi daripada *collaborative filtering* yaitu 6.34.

Berikut tabel **rating** *Top-5* anime rekomendasi dari kedua pendekatan:
| Top-5 | Content-based | Collaborative |
|---|---:|---:|
| 1 | 7.72 | 6.86 |
| 2 | 7.51 | 6.13 |
| 3 | 7.30 | 7.21 |
| 4 | 8.06 | 6.49 |
| 5 | 7.31 | 5.28 |
| Rata-rata | **7.58** | 6.34 |


### Kesimpulan  

Berdasarkan pembahasan hasil rekomendasi dari kedua pendekatan sistem, keduanya punya hasil yang lebih baik di poin tertentu dan lebih buruk di poin yang lain. Jika ingin melihat hanya dari nilai kemiripan / similaritas judul yang direkomendasikan, maka *collaborative filtering* memiliki hasil yang lebih baik. Namun, jika ingin melihat nilai rating pada anime yang direkomendasikan, maka *content-based filtering* menghasilkan judul rekomendasi dengan rating lebih tinggi, di mana rating tinggi menandakan orang menyukai anime yang ditonton tersebut.

## Referensi  
  
[1]  K. Morrissy & C. May. "Crunchyroll Market Research: Only 6% of Gen Z Don't Know What Anime Is".
https://www.animenewsnetwork.com/interest/2021-07-09/crunchyroll-market-research-only-6-percent-of-gen-z-dont-know-what-anime-is/.174962 (accessed June 4, 2023).
[2]  A. Hazra. "Netflix: More Than Half of Members Globally Watched 'Anime' Last Year". https://www.animenewsnetwork.com/daily-briefs/2022-03-30/netflix-more-than-half-of-members-globally-watched-anime-last-year/.184167 (accessed June 9, 2023).
[3]  K. Liao. "Prototyping a Recommender System Step by Step Part 1: KNN Item-Based Collaborative Filtering". https://towardsdatascience.com/prototyping-a-recommender-system-step-by-step-part-1-knn-item-based-collaborative-filtering-637969614ea (accessed June 23, 2023).


