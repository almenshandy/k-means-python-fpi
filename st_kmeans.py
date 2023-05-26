import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import preprocessing

# Tambahkan tombol untuk mengupload file excel
uploaded_file = st.file_uploader("Upload File Excel")

# Jika file excel berhasil diupload
if uploaded_file is not None:
    # Baca file excel menggunakan Pandas
    data1 = pd.read_excel(uploaded_file, engine="openpyxl")

    # Tampilkan data awal
    st.write("Data Awal")
    st.write(data1)
    cluster = st.sidebar.number_input("Jumlah Cluster", min_value=0, step=1)

    # Jika data awal berhasil diunggah
    if data1 is not None:
        # Pilih centroid dari data yang diupload
        st.sidebar.write("")
        # titik_centroid = st.sidebar.multiselect("Pilih Centroid", data1.columns)
        titik_centroid = st.sidebar.multiselect("Pilih Centroid", data1.index.tolist())

        # Jika centroid dipilih dan jumlah titik centroid sesuai dengan jumlah cluster
        if titik_centroid and len(titik_centroid) == cluster:
            # Hapus kolom yang dipilih
            drop_kolom = st.sidebar.multiselect(
                "Pilih Kolom yang Ingin Dihapus", data1.columns.tolist()
            )
            data2 = data1.drop(drop_kolom, axis=1)
            titik_centroid = data2.loc[titik_centroid].values.tolist()

            st.write("Titik centroid")
            st.write(pd.DataFrame(titik_centroid, columns=data2.columns))

            # Tampilkan data yang sudah dihapus kolomnya
            st.write("Data Setelah Dihapus Kolom")
            st.write(data2)

            # Jika jumlah cluster valid
            if cluster >= 1:
                # Tampilkan informasi jumlah cluster
                st.write(f"Jumlah Cluster: {cluster}")

                # Membuat Variabel untuk Normalisasi MinMax
                minmax = preprocessing.MinMaxScaler().fit_transform(data2)

                # Masukkan hasil data minmax kedalam variabel 'data3'
                data3 = pd.DataFrame(minmax, index=data2.index, columns=data2.columns)

                st.write("Hasil Normalisasi: ")
                st.dataframe(data3)

                # Lakukan K-Means Clustering
                kmeans = KMeans(n_clusters=cluster, init=titik_centroid)
                kmeans.fit(data3)

                jarak = kmeans.transform(data3)
                data_jarak = pd.DataFrame(
                    jarak,
                    columns=[
                        "Jarak ke cluster {}".format(i)
                        for i in range(kmeans.n_clusters)
                    ],
                )
                data_jarak["Posisi Cluster"] = kmeans.labels_

                st.write("Tabel Perbandingan Jarak:")
                st.write(data_jarak)

                # Tampilkan hasil clustering
                st.write("Hasil Clustering")
                data3["cluster"] = kmeans.labels_
                data4 = pd.concat([data1[drop_kolom], data2], axis=1)
                st.write(data3)
