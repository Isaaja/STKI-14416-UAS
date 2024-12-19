# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# **NAMA  : ISA IANT MAULANA** <br>
# **NIM   : A11.2022.14416** <br>
# **TUGAS STKI**
#
# **JUDUL : SISTEM PENCARI UNTUK MEMBERIKAN REKOMENDASI ANIME**

# ## Import Dataset

# + _kg_hide-input=true
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

# + _kg_hide-input=true
anime = pd.read_csv("./anime.csv")
rating = pd.read_csv("./rating.csv")
# -

# ## Preprocessing
#

# + _kg_hide-input=true
print(f"Shape of The Anime Dataset : {anime.shape}")
print(f"\nGlimpse of The Dataset :")
anime.head().style.set_properties()

# + _kg_hide-input=true
print(f"Informations About Anime Dataset :\n")
print(anime.info())
# -

# **Mari kita lihar sekilas dataset rating.**

# + _kg_hide-input=true
print(f"Shape of The Rating Dataset : {rating.shape}")
print(f"\nGlimpse of The Dataset :")
rating.head().style.set_properties()

# + _kg_hide-input=true
print(f"Informations About Rating Dataset :\n")
print(rating.info())
# -

# **Ringkasan dataset**
#

# + _kg_hide-input=true
print(f"Summary of The Anime Dataset :")
anime.describe().T.style.set_properties()

# + _kg_hide-input=true
anime.describe(include=object).T.style.set_properties()

# + _kg_hide-input=true
print("Null Values of Anime Dataset :")
anime.isna().sum().to_frame().T.style.set_properties()

# + _kg_hide-input=true
print("After Dropping, Null Values of Anime Dataset :")
anime.dropna(axis = 0, inplace = True)
anime.isna().sum().to_frame().T.style.set_properties()

# + _kg_hide-input=true
dup_anime = anime[anime.duplicated()].shape[0]
print(f"There are {dup_anime} duplicate entries among {anime.shape[0]} entries in anime dataset.")

# + _kg_hide-input=true
print(f"Summary of The Rating Dataset :")
rating.describe().T.style.set_properties()

# + _kg_hide-input=true
print("Null Values of Rating Dataset :")
rating.isna().sum().to_frame().T.style.set_properties()

# + _kg_hide-input=true
dup_rating = rating[rating.duplicated()].shape[0]
print(f"There are {dup_rating} duplicate entries among {rating.shape[0]} entries in rating dataset.")

rating.drop_duplicates(keep='first',inplace=True)
print(f"\nAfter removing duplicate entries there are {rating.shape[0]} entries in this dataset.")
# -

# ***Menggabungkan dataset 1 dan dataset 2***

# + _kg_hide-input=true
fulldata = pd.merge(anime,rating,on="anime_id",suffixes= [None, "_user"])
fulldata = fulldata.rename(columns={"rating_user": "user_rating"})

print(f"Shape of The Merged Dataset : {fulldata.shape}")
print(f"\nGlimpse of The Merged Dataset :")

fulldata.head().style.set_properties()
# -

# ## Visualisasi

# ***Custom Palette untuk Visualisasi***

# + _kg_hide-input=true
sns.set_style("white")
sns.set_context("poster",font_scale = .7)

palette = ["#1d7874","#679289","#f4c095","#ee2e31","#ffb563","#918450","#f85e00","#a41623","#9a031e","#d6d6d6","#ffee32","#ffd100","#333533","#202020"]

# -

# *Top Anime Community*
#

# + _kg_hide-input=true
top_anime = fulldata.copy()
top_anime.drop_duplicates(subset ="name", keep = "first", inplace = True)
top_anime_temp1 = top_anime.sort_values(["members"],ascending=False)

plt.subplots(figsize=(20,8))
p = sns.barplot(x=top_anime_temp1["name"][:14],y=top_anime_temp1["members"],palette=palette, saturation=1, edgecolor = "#1c1c1c", linewidth = 2)
p.axes.set_title("\nTop Anime Community\n", fontsize=25)
plt.ylabel("Total Member" , fontsize = 20)
plt.xlabel("\nAnime Name" , fontsize = 20)
# plt.yscale("log")
plt.xticks(rotation = 90)
for container in p.containers:
    p.bar_label(container,label_type = "center",padding = 6,size = 15,color = "black",rotation = 90,
    bbox={"boxstyle": "round", "pad": 0.6, "facecolor": "orange", "edgecolor": "black", "alpha": 1})

sns.despine(left=True, bottom=True)
plt.show()
# -

# **Insights:**
#
# * **Death Note** meraih anggota komunitas tertinggi diikuti oleh **Shingeki no Kyojin** dan **Sword Art Online**

# ***Anime Category***
#

# + _kg_hide-input=true
print("Anime Categories :")
top_anime_temp1["type"].value_counts().to_frame().T.style.set_properties()

# + _kg_hide-input=true
plt.subplots(figsize=(12, 12))

labels = "TV","OVA","Movie","Special","ONA","Music"
size = 0.5

wedges, texts, autotexts = plt.pie([len(top_anime_temp1[top_anime_temp1["type"]=="TV"]["type"]),
                                    len(top_anime_temp1[top_anime_temp1["type"]=="OVA"]["type"]),
                                    len(top_anime_temp1[top_anime_temp1["type"]=="Movie"]["type"]),
                                    len(top_anime_temp1[top_anime_temp1["type"]=="Special"]["type"]),
                                    len(top_anime_temp1[top_anime_temp1["type"]=="ONA"]["type"]),
                                    len(top_anime_temp1[top_anime_temp1["type"]=="Music"]["type"])],
                                    explode = (0,0,0,0,0,0),
                                    textprops=dict(size= 20, color= "white"),
                                    autopct="%.2f%%", 
                                    pctdistance = 0.7,
                                    radius=.9, 
                                    colors = palette, 
                                    shadow = True,
                                    wedgeprops=dict(width = size, edgecolor = "#1c1c1c", 
                                    linewidth = 4),
                                    startangle = 0)

plt.legend(wedges, labels, title="Category",loc="center left",bbox_to_anchor=(1, 0, 0.5, 1))
plt.title("\nAnime Categories Distribution",fontsize=20)
plt.show()

# + _kg_hide-input=true
plt.subplots(figsize = (20,8))
p = sns.countplot(x = top_anime_temp1["type"], order = top_anime_temp1["type"].value_counts().index, palette = palette, saturation = 1, edgecolor = "#1c1c1c", linewidth = 3)
p.axes.set_title("\nAnime Categories Hub\n" ,fontsize = 25)
plt.ylabel("Total Anime" ,fontsize = 20)
plt.xlabel("\nAnime Category" ,fontsize = 20)
plt.xticks(rotation = 0)
for container in p.containers:
    p.bar_label(container,label_type = "center",padding = 10,size = 25,color = "black",rotation = 0,
    bbox={"boxstyle": "round", "pad": 0.4, "facecolor": "orange", "edgecolor": "black", "linewidth" : 3, "alpha": 1})

sns.despine(left = True, bottom = True)
plt.show()
# -

# **Insights:**
#
# * 3402 anime ditayangkan di TV yang merupakan 30,48% dari total anime
# * 2111 anime ditayangkan dalam bentuk Film yang merupakan 18,91% dari total anime
# * 3090 anime ditayangkan sebagai OVA yang merupakan 27,69% dari total anime, juga lebih besar daripada ONA yang mencakup 526 anime yang merupakan 4,71% dari total anime.

# ***Overall Anime Ratings***
#

# + _kg_hide-input=true
top_anime_temp2 = top_anime.sort_values(["rating"],ascending=False)

_, axs = plt.subplots(2,1,figsize=(20,16),sharex=False,sharey=False)
plt.tight_layout(pad=6.0)

sns.histplot(top_anime_temp2["rating"],color=palette[11],kde=True,ax=axs[0],bins=20,alpha=1,fill=True,edgecolor=palette[12])
axs[0].lines[0].set_color(palette[12])
axs[0].set_title("\nAnime's Average Ratings Distribution\n",fontsize = 25)
axs[0].set_xlabel("Rating\n", fontsize = 20)
axs[0].set_ylabel("Total", fontsize = 20)

sns.histplot(fulldata["user_rating"],color=palette[12],kde=True,ax=axs[1],bins="auto",alpha=1,fill=True)
axs[1].lines[0].set_color(palette[11])
# axs[1].set_yscale("log")
axs[1].set_title("\n\n\nUsers Anime Ratings Distribution\n",fontsize = 25)
axs[1].set_xlabel("Rating", fontsize = 20)
axs[1].set_ylabel("Total", fontsize = 20)

sns.despine(left=True, bottom=True)
plt.show()
# -

# **Insights:**
#
# * Sebagian besar peringkat Anime tersebar antara 5.5 - 8.0
# * Sebagian besar peringkat pengguna tersebar antara 6.0 - 10.0
# * Modus dari distribusi peringkat pengguna adalah sekitar 7.0 - 8.0
# * Kedua distribusi tersebut miring ke kiri
# * Peringkat pengguna (-1) adalah outlier dalam peringkat pengguna yang dapat dibuang

# ***Top Animes Based On Ratings***
#

# + _kg_hide-input=true
plt.subplots(figsize=(20,8))
p = sns.barplot(x=top_anime_temp2["name"][:14],y=top_anime_temp2["rating"],palette=palette, saturation=1, edgecolor = "#1c1c1c", linewidth = 2)
p.axes.set_title("\nTop Animes Based On Ratings\n",fontsize = 25)
plt.ylabel("Average Rating",fontsize = 20)
plt.xlabel("\nAnime Title",fontsize = 20)
# plt.yscale("log")
plt.xticks(rotation = 90)
for container in p.containers:
    p.bar_label(container,label_type = "center",padding = 10,size = 15,color = "black",rotation = 0,
    bbox={"boxstyle": "round", "pad": 0.6, "facecolor": "orange", "edgecolor": "black", "alpha": 1})

sns.despine(left=True, bottom=True)
plt.show()
# -

# **Insights:**
#
# * **Mogura no Motoro** meraih mahkota untuk peringkat tertinggi diikuti oleh **Kimi no Na wa.** dan **Fullmetal Alchemist: Brotherhood**

# ***Category-wise Anime Ratings Distribution***
#

# + _kg_hide-input=true
print("Let's explore the ratings distribution of TV category :\n")

_, axs = plt.subplots(1,2,figsize=(20,8),sharex=False,sharey=False)
plt.tight_layout(pad=4.0)

sns.histplot(top_anime_temp2[top_anime_temp2["type"]=="TV"]["rating"],color=palette[0],kde=True,ax=axs[0],bins=20,alpha=1,fill=True,edgecolor=palette[12])
axs[0].lines[0].set_color(palette[11])
axs[0].set_title("\nAnime's Average Ratings Distribution [Category : TV]\n",fontsize=20)
axs[0].set_xlabel("Rating")
axs[0].set_ylabel("Total")

sns.histplot(fulldata[fulldata["type"]=="TV"]["user_rating"],color=palette[0],kde=True,ax=axs[1],bins="auto",alpha=1,fill=True,edgecolor=palette[12])
axs[1].lines[0].set_color(palette[11])
# axs[1].set_yscale("log")
axs[1].set_title("\nUsers Anime Ratings Distribution [Category : TV]\n",fontsize=20)
axs[1].set_xlabel("Rating")
axs[1].set_ylabel("Total")

sns.despine(left=True, bottom=True)
plt.show()
# -

# **Insights:**
#
# * Sebagian besar peringkat Anime tersebar antara 6.0 - 8.0
# * Sebagian besar peringkat pengguna tersebar antara 6.0 - 10.0
# * Modus dari distribusi peringkat pengguna adalah sekitar 7.0 - 9.0
# * Kedua distribusi tersebut miring ke kiri
# * Peringkat pengguna (-1) adalah outlier dalam peringkat pengguna yang dapat dibuang
#

# + _kg_hide-input=true
print("Let's explore the ratings distribution of OVA category :\n")

_, axs = plt.subplots(1,2,figsize=(20,8),sharex=False,sharey=False)
plt.tight_layout(pad=4.0)

sns.histplot(top_anime_temp2[top_anime_temp2["type"]=="OVA"]["rating"],color=palette[1],kde=True,ax=axs[0],bins=20,alpha=1,fill=True,edgecolor=palette[12])
axs[0].lines[0].set_color(palette[11])
axs[0].set_title("\nAnime's Average Ratings Distribution [Category : OVA]\n",fontsize=20)
axs[0].set_xlabel("Rating")
axs[0].set_ylabel("Total")

sns.histplot(fulldata[fulldata["type"]=="OVA"]["user_rating"],color=palette[1],kde=True,ax=axs[1],bins="auto",alpha=1,fill=True,edgecolor=palette[12])
axs[1].lines[0].set_color(palette[11])
# axs[1].set_yscale("log")
axs[1].set_title("\nUsers Anime Ratings Distribution [Category : OVA]\n",fontsize=20)
axs[1].set_xlabel("Rating")
axs[1].set_ylabel("Total")

sns.despine(left=True, bottom=True)
plt.show()
# -

# **Insights:**
#
# * Sebagian besar peringkat Anime tersebar antara 5.5 - 7.5
# * Sebagian besar peringkat pengguna tersebar antara 5,5 - 10,0
# * Modus dari distribusi peringkat pengguna adalah sekitar 7.0 - 8.0
# * Kedua distribusi tersebut miring ke kiri
# * Peringkat pengguna (-1) adalah outlier dalam peringkat pengguna yang dapat dibuang

# + _kg_hide-input=true
print("Let's explore the ratings distribution of MOVIE category :\n")

_, axs = plt.subplots(1,2,figsize=(20,8),sharex=False,sharey=False)
plt.tight_layout(pad=4.0)

sns.histplot(top_anime_temp2[top_anime_temp2["type"]=="Movie"]["rating"],color=palette[2],kde=True,ax=axs[0],bins=20,alpha=1,fill=True,edgecolor=palette[12])
axs[0].lines[0].set_color(palette[3])
axs[0].set_title("\nAnime's Average Ratings Distribution [Category : Movie]\n",fontsize=20)
axs[0].set_xlabel("Rating")
axs[0].set_ylabel("Total")

sns.histplot(fulldata[fulldata["type"]=="Movie"]["user_rating"],color=palette[2],kde=True,ax=axs[1],bins="auto",alpha=1,fill=True,edgecolor=palette[12])
axs[1].lines[0].set_color(palette[3])
# axs[1].set_yscale("log")
axs[1].set_title("\nUsers Anime Ratings Distribution [Category : Movie]\n",fontsize=20)
axs[1].set_xlabel("Rating")
axs[1].set_ylabel("Total")

sns.despine(left=True, bottom=True)
plt.show()
# -

# **Insights:**
#
# * Sebagian besar peringkat Anime tersebar antara 4,5 - 8,5
# * Sebagian besar peringkat pengguna tersebar antara 5.0 - 10.0
# * Modus dari distribusi peringkat pengguna adalah sekitar 7.0 - 9.0
# * Kedua distribusi tersebut miring ke kiri
# * Peringkat pengguna (-1) adalah outlier dalam peringkat pengguna yang dapat dibuang

# + _kg_hide-input=true
print("Let's explore the ratings distribution of SPECIAL category :\n")

_, axs = plt.subplots(1,2,figsize=(20,8),sharex=False,sharey=False)
plt.tight_layout(pad=4.0)

sns.histplot(top_anime_temp2[top_anime_temp2["type"]=="Special"]["rating"],color=palette[3],kde=True,ax=axs[0],bins=20,alpha=1,fill=True,edgecolor=palette[12])
axs[0].lines[0].set_color(palette[11])
axs[0].set_title("\nAnime's Average Ratings Distribution [Category : Special]\n",fontsize=20)
axs[0].set_xlabel("Rating")
axs[0].set_ylabel("Total")

sns.histplot(fulldata[fulldata["type"]=="Special"]["user_rating"],color=palette[3],kde=True,ax=axs[1],bins="auto",alpha=1,fill=True,edgecolor=palette[12])
axs[1].lines[0].set_color(palette[11])
# axs[1].set_yscale("log")
axs[1].set_title("\nUsers Anime Ratings Distribution [Category : Special]\n",fontsize=20)
axs[1].set_xlabel("Rating")
axs[1].set_ylabel("Total")

sns.despine(left=True, bottom=True)
plt.show()
# -

# **Insights:**
#
# * Sebagian besar peringkat Anime tersebar antara 5.5 - 8.0
# * Sebagian besar peringkat pengguna tersebar antara 5.0 - 10.0
# * Modus dari distribusi peringkat pengguna adalah sekitar 7.0 - 8.0
# * Kedua distribusi tersebut miring ke kiri
# * Peringkat pengguna (-1) adalah outlier dalam peringkat pengguna yang dapat dibuang

# + _kg_hide-input=true
print("Let's explore the ratings distribution of ONA category :\n")

_, axs = plt.subplots(1,2,figsize=(20,8),sharex=False,sharey=False)
plt.tight_layout(pad=4.0)

sns.histplot(top_anime_temp2[top_anime_temp2["type"]=="ONA"]["rating"],color=palette[4],kde=True,ax=axs[0],bins=20,alpha=1,fill=True,edgecolor=palette[12])
axs[0].lines[0].set_color(palette[3])
axs[0].set_title("\nAnime's Average Ratings Distribution [Category : ONA]\n",fontsize=20)
axs[0].set_xlabel("Rating")
axs[0].set_ylabel("Total")

sns.histplot(fulldata[fulldata["type"]=="ONA"]["user_rating"],color=palette[4],kde=True,ax=axs[1],bins="auto",alpha=1,fill=True,edgecolor=palette[12])
axs[1].lines[0].set_color(palette[3])
# axs[1].set_yscale("log")
axs[1].set_title("\nUsers Anime Ratings Distribution [Category : ONA]\n",fontsize=20)
axs[1].set_xlabel("Rating")
axs[1].set_ylabel("Total")

sns.despine(left=True, bottom=True)
plt.show()
# -

# **Insights:**
#
# * Sebagian besar peringkat Anime tersebar antara 4.0 - 7.0
# * Sebagian besar peringkat pengguna tersebar antara 5.0 - 10.0
# * Modus dari distribusi peringkat pengguna adalah sekitar 7.0 - 8.0
# * Kedua distribusi tersebut miring ke kiri
# * Peringkat pengguna (-1) adalah outlier dalam peringkat pengguna yang dapat dibuang

# + _kg_hide-input=true
print("Let's explore the ratings distribution of MUSIC category :\n")

_, axs = plt.subplots(1,2,figsize=(20,8),sharex=False,sharey=False)
plt.tight_layout(pad=4.0)

sns.histplot(top_anime_temp2[top_anime_temp2["type"]=="Music"]["rating"],color=palette[5],kde=True,ax=axs[0],bins=20,alpha=1,fill=True,edgecolor=palette[12])
axs[0].lines[0].set_color(palette[11])
axs[0].set_title("\nAnime's Average Ratings Distribution [Category : Music]\n",fontsize=20)
axs[0].set_xlabel("Rating")
axs[0].set_ylabel("Total")

sns.histplot(fulldata[fulldata["type"]=="Music"]["user_rating"],color=palette[5],kde=True,ax=axs[1],bins="auto",alpha=1,fill=True,edgecolor=palette[12])
axs[1].lines[0].set_color(palette[11])
# axs[1].set_yscale("log")
axs[1].set_title("\nUsers Anime Ratings Distribution [Category : Music]\n",fontsize=20)
axs[1].set_xlabel("Rating")
axs[1].set_ylabel("Total")

sns.despine(left=True, bottom=True)
plt.show()
# -

# **Insights:**
#
# * Sebagian besar peringkat Anime tersebar antara 4.0 - 7.5
# * Sebagian besar peringkat pengguna tersebar antara 5.0 - 10.0
# * Modus dari distribusi peringkat pengguna adalah sekitar 6.5 - 8.0
# * Kedua distribusi tersebut miring ke kiri
# * Peringkat pengguna (-1) adalah outlier dalam peringkat pengguna yang dapat dibuang

# ***Genre Anime***
#

# + _kg_hide-input=true
top_anime_temp3 = top_anime[["genre"]]
top_anime_temp3["genre"] = top_anime_temp3["genre"].str.split(", | , | ,")
top_anime_temp3 = top_anime_temp3.explode("genre")
top_anime_temp3["genre"] = top_anime_temp3["genre"].str.title()

print(f'Total unique genres are {len(top_anime_temp3["genre"].unique())}')
print(f'Occurances of unique genres :')
top_anime_temp3["genre"].value_counts().to_frame().T.style.set_properties()

# + _kg_hide-input=true
from wordcloud import WordCloud

wordcloud = WordCloud(width = 800, height = 250, background_color ="black",colormap ="RdYlGn",
                      max_font_size=100, stopwords =None,repeat= True).generate(top_anime["genre"].str.cat(sep=", | , | ,"))

print("let's explore how genre's wordcloud looks like\n")
plt.figure(figsize = (20, 8),facecolor = "#ffd100") 
plt.imshow(wordcloud)
plt.axis("off")
plt.margins(x = 0, y = 0)
plt.tight_layout(pad = 0) 
plt.show()
# -

# ***Final Data Preprocessing***
#

# + _kg_hide-input=true
data = fulldata.copy()
data["user_rating"].replace(to_replace = -1 , value = np.nan ,inplace=True)
data = data.dropna(axis = 0)
print("Null values after final pre-processing :")
data.isna().sum().to_frame().T.style.set_properties()
# -

# Ada banyak pengguna yang hanya memberi rating sekali, bahkan jika mereka telah memberi rating 5 anime, hal itu tidak dapat dianggap sebagai catatan yang berharga untuk rekomendasi. Jadi, kami akan mempertimbangkan minimal 50 peringkat oleh pengguna sebagai nilai ambang batas.

# + _kg_hide-input=true
selected_users = data["user_id"].value_counts()
data = data[data["user_id"].isin(selected_users[selected_users >= 50].index)]
# -

# Kita akan membuat tabel pivot yang terdiri dari baris sebagai judul dan kolom sebagai id pengguna, hal ini akan membantu kita dalam membuat matriks jarang yang dapat sangat membantu dalam mencari kemiripan cosinus.

# + _kg_hide-input=true
data_pivot_temp = data.pivot_table(index="name",columns="user_id",values="user_rating").fillna(0)
data_pivot_temp.head()
# -

# Kami memiliki banyak simbol karakter Jepang atau karakter khusus dalam nama anime. Mari kita hapus semua itu dengan menggunakan fungsi ini

# + _kg_hide-input=true
import re
def text_cleaning(text):
    text = re.sub(r'&quot;', '', text)
    text = re.sub(r'.hack//', '', text)
    text = re.sub(r'&#039;', '', text)
    text = re.sub(r'A&#039;s', '', text)
    text = re.sub(r'I&#039;', 'I\'', text)
    text = re.sub(r'&amp;', 'and', text)
    
    return text

data["name"] = data["name"].apply(text_cleaning)

# + _kg_hide-input=true
data_pivot = data.pivot_table(index="name",columns="user_id",values="user_rating").fillna(0)
print("After Cleaning the animes names, let's see how it looks like.")
data_pivot.head()
# -

# ## Pengembangan Model Rekomendasi
#
# ***Teknik Matrix Factorization atau KNN***
#

# Collaborative Filtering adalah teknik yang dapat menyaring item yang mungkin disukai pengguna berdasarkan reaksi dari pengguna yang serupa. Teknik ini bekerja dengan mencari sekelompok besar orang dan menemukan sekumpulan kecil pengguna yang memiliki selera yang mirip dengan pengguna tertentu. Kami akan menggunakan Cosine similarity yang merupakan metrik yang digunakan untuk mengukur seberapa mirip dokumen terlepas dari ukurannya. Secara matematis, metrik ini mengukur kosinus sudut antara dua vektor yang diproyeksikan dalam ruang multi-dimensi. Kemiripan kosinus menguntungkan karena meskipun dua dokumen yang mirip berjauhan menurut jarak Euclidean (karena ukuran dokumen), kemungkinan besar dokumen-dokumen tersebut masih berorientasi lebih dekat satu sama lain. Semakin kecil sudutnya, semakin tinggi kemiripan kosinusnya.

# + _kg_hide-input=true
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

data_matrix = csr_matrix(data_pivot.values)

model_knn = NearestNeighbors(metric = "cosine", algorithm = "brute")
model_knn.fit(data_matrix)

query_no = np.random.choice(data_pivot.shape[0]) # random anime title and finding recommendation
print(f"We will find recommendation for {query_no} no anime which is {data_pivot.index[query_no]}.")
distances, indices = model_knn.kneighbors(data_pivot.iloc[query_no,:].values.reshape(1, -1), n_neighbors = 6)

# + _kg_hide-input=true
no = []
name = []
distance = []
rating = []

for i in range(0, len(distances.flatten())):
    if i == 0:
        print(f"Recommendations for {data_pivot.index[query_no]} viewers :\n")
    else:
        #  print(f"{i}: {data_pivot.index[indices.flatten()[i]]} , with distance of {distances.flatten()[i]}")        
        no.append(i)
        name.append(data_pivot.index[indices.flatten()[i]])
        distance.append(distances.flatten()[i])
        rating.append(*anime[anime["name"]==data_pivot.index[indices.flatten()[i]]]["rating"].values)

dic = {"No" : no, "Anime Name" : name, "Rating" : rating}
recommendation = pd.DataFrame(data = dic)
recommendation.set_index("No", inplace = True)
recommendation.style.set_properties()
# -

# ## Evaluate
#

# Content Based Filtering merekomendasikan item berdasarkan perbandingan antara konten item dan profil pengguna. Konten setiap item direpresentasikan sebagai sekumpulan deskriptor atau istilah, biasanya berupa kata-kata yang muncul dalam dokumen. Rekomendasi berbasis konten bekerja dengan data yang diberikan pengguna, baik secara eksplisit (peringkat) atau implisit (mengklik tautan). Berdasarkan data tersebut, profil pengguna dibuat yang kemudian digunakan untuk memberikan saran kepada pengguna. Ketika pengguna memberikan lebih banyak masukan atau melakukan tindakan atas rekomendasi, mesin akan semakin akurat.
#
# **Term Frequency(TF) & Inverse Document Frequency(IDF)**
#
# TF adalah frekuensi sebuah kata dalam sebuah dokumen. IDF adalah kebalikan dari frekuensi dokumen di antara seluruh korpus dokumen. TF-IDF digunakan terutama karena dua alasan: Misalkan kita mencari “kebangkitan analitik” di Google. Sudah pasti bahwa “the” akan muncul lebih sering daripada “analytics”, tetapi kepentingan relatif dari analytics lebih tinggi daripada sudut pandang kueri penelusuran. Dalam kasus seperti itu, pembobotan TF-IDF meniadakan efek dari kata-kata berfrekuensi tinggi dalam menentukan tingkat kepentingan suatu item (dokumen).
# Di sini kita akan menggunakannya pada genre anime sehingga kita dapat merekomendasikan konten kepada pengguna berdasarkan genre.
#

# + _kg_hide-input=true
from sklearn.feature_extraction.text import TfidfVectorizer

tfv = TfidfVectorizer(min_df=3, max_features=None, strip_accents="unicode", analyzer="word",
                      token_pattern=r"\w{1,}", ngram_range=(1, 3), stop_words = "english")

rec_data = fulldata.copy()
rec_data.drop_duplicates(subset ="name", keep = "first", inplace = True)
rec_data.reset_index(drop = True, inplace = True)
genres = rec_data["genre"].str.split(", | , | ,").astype(str)
tfv_matrix = tfv.fit_transform(genres)
# -

#
# Selain itu, scikit-learn sudah menyediakan metrik berpasangan (alias kernel dalam bahasa machine learning) yang dapat digunakan untuk representasi koleksi vektor yang padat maupun yang jarang. Di sini kita perlu menetapkan nilai 1 untuk anime yang direkomendasikan dan 0 untuk anime yang tidak direkomendasikan. Kita akan menggunakan kernel sigmoid di sini.

# + _kg_hide-input=true
from sklearn.metrics.pairwise import sigmoid_kernel

sig = sigmoid_kernel(tfv_matrix, tfv_matrix)      # Computing sigmoid kernel

rec_indices = pd.Series(rec_data.index, index = rec_data["name"]).drop_duplicates()


# Recommendation Function
def give_recommendation(title, sig = sig):
    
    idx = rec_indices[title] # Getting index corresponding to original_title

    sig_score = list(enumerate(sig[idx]))  # Getting pairwsie similarity scores 
    sig_score = sorted(sig_score, key=lambda x: x[1], reverse=True)
    sig_score = sig_score[1:11]
    anime_indices = [i[0] for i in sig_score]
     
    # Top 10 most similar movies
    rec_dic = {"No" : range(1,11), 
               "Anime Name" : anime["name"].iloc[anime_indices].values,
               "Rating" : anime["rating"].iloc[anime_indices].values,
               "Genre" : anime["genre"].iloc[anime_indices].values}
    dataframe = pd.DataFrame(data = rec_dic)
    dataframe.set_index("No", inplace = True)
    
    print(f"Recommendations for {title} viewers :\n")
    
    return dataframe.style.set_properties()




# +


