# Import library
import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples

# Load dataset
# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv("warmindo.csv")
    # Return a copy of the data to avoid mutation warning
    return data.copy()

data_warmindo = load_data()


# Sidebar untuk pemilihan atribut
st.sidebar.title("Pilih Atribut untuk Clustering")
selected_attributes = st.sidebar.multiselect("Pilih Atribut", data_warmindo.columns)

# Menampilkan dataset terpilih
# Menampilkan judul
st.title("Analisis Segmentasi Pelanggan dengan Metode K-Means Clustering")

# Center-align content
st.markdown(
    """
    <style>
        div.stButton > button {
            margin: 0 auto;
            display: block;
        }
        .stMarkdown {
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True
)
st.subheader("Dataset Terpilih")
st.write(data_warmindo[selected_attributes])

# Mengidentifikasi tipe data dan mengelompokkan kolom menjadi numerik atau kategorikal
numeric_cols = data_warmindo[selected_attributes].select_dtypes(include=['float64', 'int64']).columns
categorical_cols = data_warmindo[selected_attributes].select_dtypes(include=['object']).columns

# Menjalankan K-Means dengan jumlah kluster yang dipilih
num_clusters = st.slider("Pilih Jumlah Kluster", min_value=2, max_value=10, step=1, key="slider_key")

if st.button("Lakukan K-Means Clustering"):
    # Memilih kolom numerik
    selected_numeric_data = data_warmindo[numeric_cols]

    # Melakukan normalisasi
    scaler = StandardScaler()
    selected_numeric_data_normalized = scaler.fit_transform(selected_numeric_data)

    # Menggabungkan kolom kategorikal yang telah di-encode
    encoded_categorical_data = pd.get_dummies(data_warmindo[categorical_cols])
    selected_data = pd.concat([pd.DataFrame(selected_numeric_data_normalized, columns=numeric_cols), encoded_categorical_data], axis=1)

    # Menggunakan hanya kolom terpilih
    selected_data = selected_data[numeric_cols.tolist() + encoded_categorical_data.columns.tolist()]

    # Inisialisasi dan menjalankan K-Means
    kmeans = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
    data_warmindo['Cluster'] = kmeans.fit_predict(selected_data)

    # Menampilkan hasil clustering
    st.subheader("Hasil Clustering (Segmentasi Pola Pelanggan)")
    st.write(data_warmindo[['jenis_pembayaran', 'jenis_produk', 'Cluster']])

    # Visualisasi hasil clustering
    st.subheader("Visualisasi Hasil Clustering")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.scatterplot(x=selected_data.iloc[:, 0], y=selected_data.iloc[:, 1], hue=data_warmindo['Cluster'], palette='viridis', ax=ax)
    plt.title('Hasil Clustering - Scatter Plot')
    plt.xlabel(selected_data.columns[0])
    plt.ylabel(selected_data.columns[1])
    st.pyplot(fig)



    # Deskripsi hasil clustering
    st.subheader("Deskripsi Hasil Clustering (K-Means)")
    st.write("Visualisasi di atas menunjukkan hasil dari proses clustering menggunakan metode K-Means.")
    st.write("K-Means mengelompokkan data menjadi beberapa kluster berdasarkan kesamaan fitur.")
    st.write(f"Jumlah kluster yang dipilih: {num_clusters}")

    # Menampilkan statistik deskriptif untuk setiap kluster
    cluster_stats = data_warmindo.groupby('Cluster')[numeric_cols].describe()
    st.subheader("Statistik Deskriptif untuk Setiap Kluster")
    st.write(cluster_stats)

    # Grafik Perbandingan Jenis Pembayaran
    st.subheader("Grafik Perbandingan Jenis Pembayaran")
    payment_counts = data_warmindo['jenis_pembayaran'].value_counts()
    st.bar_chart(payment_counts)

    # Grafik Perbandingan Jenis Pesanan
    st.subheader("Grafik Perbandingan Jenis Pesanan")
    order_counts = data_warmindo['jenis_pesanan'].value_counts()
    st.bar_chart(order_counts)

    # Grafik Perbandingan Jenis Produk
    st.subheader("Grafik Perbandingan Jenis Produk")
    product_counts = data_warmindo['jenis_produk'].value_counts()
    st.bar_chart(product_counts)

    # Menghitung Silhouette Score untuk tiap kluster
    silhouette_avg = silhouette_score(selected_data, data_warmindo['Cluster'])
    silhouette_scores_per_cluster = silhouette_samples(selected_data, data_warmindo['Cluster'])

    st.subheader("Silhouette Score")
    st.write(f"Silhouette Score untuk seluruh kluster: {silhouette_avg:.3f}")

    st.subheader("Silhouette Score per Kluster")
    for cluster_num in range(num_clusters):
        cluster_score = silhouette_scores_per_cluster[data_warmindo['Cluster'] == cluster_num].mean()
        st.write(f"Kluster {cluster_num}: {cluster_score:.3f}")


