#!/usr/bin/env python
# coding: utf-8

# Reference: https://www.kaggle.com/code/yamanizm/recommendation-systems-svd-hybrid-k-nn-kmeans
# 
# # Submission 2 MLT - Sistem Rekomendasi Anime
# https://www.dicoding.com/users/atthaariq
# 
# ## Load Dataset
# Download *dataset* dari *kaggle* dengan opendatasets.

# In[1]:


import opendatasets as od
od.download('https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database?select=rating.csv')
od.download('https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database?select=anime.csv')


# **_Dataset_ yang digunakan adalah sebagai berikut:**
# - `anime` yang berisi informasi anime, dipakai untuk target rekomendasi
# - `rating` yang berisi informasi *rating* anime oleh pengguna, dipakai sebagai data yang dilatih.

# In[2]:


import pandas as pd
anime = pd.read_csv('anime-recommendations-database/anime.csv')
rating = pd.read_csv('anime-recommendations-database/rating.csv')


# Melihat 5 data teratas dari *dataset* anime dan rating.

# In[3]:


anime.head()


# In[17]:


rating.head()


# Kita dapat melihat 5 data teratas dari *dataset* anime & rating yang berarti *dataset* ini berhasil di-*load*.

# In[4]:


anime[anime.name=='Pokemon']


# In[5]:


rating[rating.anime_id==400]


# ## Data Understanding & Analysis
# 
# Pertama, melihat berapa banyak kolom dan baris (variabel & data).

# In[6]:


anime.info()


# In[7]:


rating.info()


# In[7]:


print(f'anime shape: {anime.shape}\nrating shape: {rating.shape}')


# - `anime` memiliki 7 kolom data dan 12,294 baris.
# - `rating` memiliki 3 kolom data dan 7,813,737 baris.

# In[8]:


anime.describe()


# Terdapat 3 kolom numerik pada *dataset* `anime`, yaitu 2 kolom numerik diskrit yaitu `anime_id` & `members`, serta kolom numerik kontinyu yaitu `rating`.

# In[9]:


rating.describe()


# Semua kolom *dataset* `rating` adalah data numerik diskrit.

# Kemudian cek apakah ada *missing values* / null pada kedua *dataset*.

# In[10]:


rating.isna().sum()


# In[11]:


anime.isna().sum()


# *Dataset* anime memiliki nilai null pada 3 kolom, dengan null terbanyak pada kolom `rating` (230 data).
# 
# Semua baris data yang memiliki nilai null pada kolom datanya akan di-*drop*.

# In[12]:


anime.dropna(axis = 0, inplace = True)
anime.isna().sum()


# In[13]:


anime.info()


# Tidak ada lagi null pada *dataset* `anime`. Jumlah baris data menjadi 12017.

# In[14]:


anime.describe()


# Melihat keragaman data pada *dataset* `anime` & `rating`.

# In[15]:


anime.nunique()


# In[16]:


rating.nunique()


# Impor *library* untuk keperluan visualisasi data.

# In[17]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Melihat korelasi antar kolom kedua *dataset* dengan *heatmap*.

# In[18]:


plt.figure(figsize=(7,5))
plt.title("Heatmap korelasi dataset anime")
sns.heatmap(anime.corr(),annot = True,cmap = 'mako_r')
plt.figure(figsize=(7,5))
plt.title("Heatmap korelasi dataset anime")
sns.heatmap(rating.corr(),annot = True,cmap = 'mako_r')


# Mengecek duplikasi kolom pada kedua tabel.

# In[19]:


print(anime[anime.duplicated()].shape)
print(rating[rating.duplicated()].shape)


# Terdapat duplikasi pada `rating`, kita hapus dengan `drop_duplicated`.

# In[20]:


rating.drop_duplicates(keep='first',inplace=True)
rating[rating.duplicated()].shape


# ## Create Dataset for Recommender System
# 
# *Dataset* untuk *training* didapatkan dari menggabungkan *dataset* `anime` dan `rating`.

# In[22]:


df = pd.merge(anime,rating, on='anime_id')
df.head(5)


# In[23]:


df.describe()


# Data rating yang akan dipakai adalah `rating_x` karena `rating_y` memiliki data dengan nilai -1. `rating_x` diubah menjadi `user_rating`.

# In[24]:


df = df.rename(columns={"rating_x": "user_rating"})
df = df.drop('rating_y', axis=1)


# ## Data Preparation 
# 
# ### Handling Missing Value
# Mengecek nilai Null pada `df`.

# In[26]:


df = df.copy()
df = df.dropna(axis = 0)
print("Nilai null:")
df.isna().sum()


# ### Cleaning Text
# Sebuah *Function* untuk:
# - Mengubah teks menjadi *lowercase*
# - Menghapus simbol dan *special characters*

# In[28]:


import re
import string

def clean_text(text):
    # to lowercase
    text = text.lower()

    # remove sybmols and other words
    text = re.sub(r'<[^>]*>', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'&quot;', '', text)
    text = re.sub(r'.hack//', '', text)
    text = re.sub(r'&#039;', '', text)
    text = re.sub(r'A&#039;s', '', text)
    text = re.sub(r'I&#039;', 'I\'', text)
    text = re.sub(r'&amp;', 'and', text)
  
    # remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    return text


# *Cleaning* pada data `'name'` di *dataset* `df` dan `anime`.

# In[29]:


import time

start = time.time()
df['name']=df['name'].apply(clean_text)
end = time.time()
print("Preprocessing df['name']: ", end-start, " sec.")

start = time.time()
anime['name'] = anime['name'].apply(clean_text)
end = time.time()
print("Preprocessing anime['name']: ", end-start, " sec.")


# ## Modelling
# ### Popularity-based
# *Function* untuk rekomendasi dengan *popularity-based* melakukan grouping berdasarkan input fitur nanti dengan `groupby()` dan menghitung *mean* dari rating penontonnya.
# 
# Kemudian hasil tersebut diurutkan dan dan ditampilkan dengan 10 hasil teratas.

# In[30]:


def popularity_recommender(df, selected_features):

    # grouping & menghitung rata-rata rating pengguna
    grouped_df = df.groupby(selected_features).agg({'user_rating': 'mean'}).reset_index()
    # mengurutkan berdasarkan rating
    sorted_df = grouped_df.sort_values('user_rating', ascending=False)
    # menampilkan 10 hasil teratas yang diurutkan
    recommendations = sorted_df.head(10)
    return recommendations


# In[31]:


df.columns


# Rating berdasarkan judul anime:

# In[32]:


# berdasarkan judul
selected_features = ['name']
popularity_recommender(df, selected_features)


# Rating berdasarkan 10 judul dengan penonton terbanyak:

# In[33]:


# berdasarkan 10 judul terpopuler
selected_features = ['members']
popularity_recommender(df, selected_features)


# Rating berdasarkan genre terpopuler. Genre yang dipakai hanya genre urutan pertama / *first genre*.

# In[34]:


# membuat kolom first genre
df['first_genre'] = df['genre'].apply(lambda x: x.split(',')[0].strip() if ',' in x else x)

# rekom berdasarkan first genre
selected_features = ['first_genre']
popularity_recommender(df, selected_features)


# Rating berdasarkan jenis / tipe anime:

# In[35]:


#berdasarkan tipe
selected_features = ['type']
popularity_recommender(df, selected_features)


# ### Title-based Recommender
# - **a. Clustering + Collaborative Filtering**
# - **b. Clustering + Content-based Filtering**
# 
# Terdapat dua pendekatan pada sistem rekomendasi berdasarkan judul yang sedang ditonton. Masing-masing pendekatan akan menggunakan klasterisasi dahulu.
# 
# #### 1. Clustering (K-Means)
# 
# - *Label encoding* pada data 'genre' dan 'type' di `df`.
# - Model K-Means dengan parameter:
#     - n_clusters: 6
#     - random_state: 42
# - Kemudian *fit* model pada data 'anime_id', 't_genre', 't_type', dan 'user_rating' di `df`.

# In[41]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

#encoding
le = LabelEncoder()

start = time.time()
df['t_genre']= le.fit_transform(df['genre'])
df['t_type']= le.fit_transform(df['type'])
end = time.time()
print("Label encoding :", end-start, " sec.")

selected_features = ['anime_id','t_genre','t_type', 'user_rating']
 
# k-means model
kmeans = KMeans(n_clusters=6, random_state=42)

start = time.time()
df['cluster'] = kmeans.fit_predict(df[selected_features])
end = time.time()
print("K-Means fit:", end-start, " sec.")


# In[42]:


from collections import Counter

labels = kmeans.labels_

# hitung elemen pada tiap klaster
cluster_counts = Counter(labels)

for cluster_id, count in cluster_counts.items():
    print(f"Klaster {cluster_id}: {count} elemen")


# #### 2. Set random user & anime title to recommend
# 
# Memilih pengguna dan 1 judul anime yang telah ditonton pengguna tersebut secara random untuk diberikan rekomendaasi anime yang mirip. 
# 
# - Memilih klaster random dan user di dalam klaster tersebut secara random dengan `randint()` pada `const_cluster_no`
# - Membuat pivot table berdasarkan klaster terpilih dengan nilai rating dengan `pivot_table()`
# - Memilih judul anime yang bakal diberikan rekomendasi secara random pada `query_no`

# In[44]:


import random

# memilih klaster random dan user di dalam klaster tersebut secara random
const_member_index = random.randint(1, len(df))
const_cluster_no = df.cluster[const_member_index]
const_cluster_no

user_no = df.user_id[const_member_index]

# membuat pivot table berdasarkan klaster terpilih dengan nilai rating
df_pivot = df[df.cluster == const_cluster_no].pivot_table(index="name", columns="user_id", values="user_rating").fillna(0)


# In[49]:


# memilih judul anime yang bakal diberikan rekomendasi secara random
query_no = np.random.choice(df_pivot.shape[0]) 
print(f"Judul anime terpilih untuk diberikan rekomendasi:\nKueri: {query_no}\nJudul: {df_pivot.index[query_no]} ")
anime_const = df_pivot.index[query_no]


# #### A. Collaborative Filtering (KNN)
# - Membuat Matriks *Compressed Sparse Row* (CSR) pada `df_pivot` dengan scipy `csr_matrix`
# - Membuat model dengan scikit learn `NearestNeighbors()` yang memiliki parameter:
#     - metric: "cosine"
#     - algorithm: "auto"
# - Kemudian *fit* `df_matrix()` tadi dalam model KNN
# - Jarak dan indeks tiap anime didefinisikan dengan `model_knn.kneighbors()`
# 
# Kemudian untuk membuat rekomendasinya:
# - Isi list `no`, `name`, `distance`, `rating`, dan `genre` berdasarkan data jarak dan indeks, di mana untuk `rating` dan `genre` menggunakan `flatten()`

# In[52]:


from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# KNN
start = time.time()
df_matrix = csr_matrix(df_pivot.values)
model_knn = NearestNeighbors(metric="cosine", algorithm="auto")
model_knn.fit(df_matrix)

distances, indices = model_knn.kneighbors(df_pivot.iloc[query_no, :].values.reshape(1, -1), n_neighbors=11)

no = []
name = []
distance = []
rating = []
genre = []

# rekomendasi
for i in range(0, len(distances.flatten())):
    if i == 0:
        print(f"Recommendations for '{df_pivot.index[query_no]}' viewers :\n")
    else:
        no.append(i)
        name.append(df_pivot.index[indices.flatten()[i]])
        distance.append(distances.flatten()[i])
        rating.append(*anime[anime["name"] == df_pivot.index[indices.flatten()[i]]]["rating"].values)
        genre.append(*anime[anime["name"] == df_pivot.index[indices.flatten()[i]]]["genre"].values)

        
dic = {"No": no, "Anime Name": name, "Rating": rating, "Genre": genre, "Similarity": distance[::-1]}
recommendation = pd.DataFrame(data=dic)
recommendation.set_index("No", inplace=True)
end = time.time()
print("Collaborative filtering :", end-start, " sec.")
recommendation.head(5)


# #### Content-based Filtering (TF-IDF)
# 
# - Melihat nilai pada fitur penting tiap anime dengan TF-IDF `TfidfVectorizer()` dengan parameter `analyzer`: "word"
# - Ambil data klaster tadi yang telah dipilih acak untuk dijadikan data rekomendasi `rec_data`
# - fitur penting TF-IDF adalah genre
# - *Fit* data genre tadi menjadi matriks 
# - Menghitung kemiripan juga dengan *cosine similarity*
# - Kemudian *drop* duplikasi data pada data indeks dengan `pd.Series()` & `drop_duplicates()`
# 
# Untuk membuat rekomendasinya:
# 
# Urutkan nilai TF-IDF (`cos_sim`) seluruh anime dengan `sorted()` berdasarkan jarak kemiripian fitur tertinggi dalam list `cos_scores` dengan judul anime yang akan diberikan rekomendasi

# In[58]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

start = time.time()
# TF-IDF
tfv = TfidfVectorizer(analyzer="word")

# ambil data klaster yang terpilih
rec_data = df[df.cluster == const_cluster_no].copy()
rec_data.drop_duplicates(subset ="name", keep = "first", inplace = True)
rec_data.reset_index(drop = True, inplace = True)

# jadikan genre sebagai fitur tf-idf
genres = rec_data["name"].str.split(", | , | ,").astype(str)

# matriks tf-idf
tfv_matrix = tfv.fit_transform(genres)
 
# hitung kemiripan
cos_sim = cosine_similarity(tfv_matrix, tfv_matrix)

# drop duplikasi
rec_indices = pd.Series(rec_data.index, index = rec_data["name"]).drop_duplicates()

# function rekomendasi
def give_recommendation(title, cos_sim=cos_sim):
    idx = rec_indices[title]
    cos_scores = list(enumerate(cos_sim[idx]))
    cos_scores = sorted(cos_scores, key=lambda x: x[1], reverse=True)
    cos_scores = cos_scores[1:11]
    anime_indices = [i[0] for i in cos_scores]
    
    # isi data
    sim_scores = [i[1] for i in cos_scores]
    rec_dic = {
        "No": range(1, 11),
        "Anime Name": anime["name"].iloc[anime_indices].values,
        "Rating": anime["rating"].iloc[anime_indices].values,
        "Genre": anime["genre"].iloc[anime_indices].values,
        "Similarity Score": sim_scores,
    }
    dataframe = pd.DataFrame(data=rec_dic)
    dataframe.set_index("No", inplace=True)

    print(f"Recommendations for '{title}' viewers :\n")

    return dataframe

end = time.time()
print("Content-based filtering :", end-start, " sec.")

give_recommendation(anime_const).head()


# In[ ]:




