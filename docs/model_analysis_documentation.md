# Dokumentasi Analisis Model Deteksi Sarkasme

## 1. Perbandingan Rasio Split Data (70:30 vs 80:20)

| SplitRatio | F1-scores (%) | Recall | Precision |
|------------|--------------|--------|-----------|
| 70:30      | 75.04%       | 76.14% | 75.23%    |
| 80:20      | 76.88%       | 77.14% | 76.47%    |

Secara umum, rasio split data 80:20 menunjukkan performa yang lebih baik dibandingkan dengan rasio 70:30, dengan peningkatan pada semua metrik evaluasi (F1-score, Recall, dan Precision). Rasio 80:20 memberikan performa lebih baik karena menyediakan lebih banyak data untuk pelatihan, sehingga model dapat mempelajari pola yang lebih baik dari dataset. Peningkatan F1-score dari 75.04% menjadi 76.88% menunjukkan bahwa model dengan rasio 80:20 memiliki keseimbangan yang lebih baik antara precision dan recall, sehingga rasio ini lebih direkomendasikan untuk digunakan pada skenario berikutnya.

## 2. Perbandingan Teknik Stemming dan NonStemming

| Parameter    | F1-scores(%) | Recall  | Precision |
|--------------|--------------|---------|-----------|
| NonStemming  | 76.88%       | 76.14%  | 75.23%    |
| Stemming     | 75.04%       | 77.14%  | 76.47%    |

Berdasarkan tabel di atas, penerapan teknik Stemming menurunkan F1-score dari 76.88% menjadi 75.04%, namun meningkatkan Recall dari 76.14% menjadi 77.14% dan Precision dari 75.23% menjadi 76.47%. Terjadi trade-off di mana Stemming meningkatkan kemampuan model untuk mengidentifikasi lebih banyak kasus positif (Recall lebih tinggi) dan mengurangi false positive (Precision lebih tinggi), tetapi menurunkan keseimbangan keseluruhan antara kedua metrik tersebut (F1-score lebih rendah). Untuk deteksi sarkasme, Stemming dapat bermanfaat jika prioritas utama adalah meningkatkan Recall dan Precision, namun jika keseimbangan keseluruhan lebih penting, pendekatan NonStemming lebih direkomendasikan.

## 3. Parameter Model dan Pengaruhnya

Model deteksi sarkasme memerlukan konfigurasi parameter yang optimal untuk mencapai performa terbaik dalam mengidentifikasi konten sarkastik dalam teks. Hidden size menentukan kapasitas model untuk mengenali pola kompleks, sedangkan dropout mencegah overfitting dengan menonaktifkan neuron secara acak. Learning rate mengontrol seberapa cepat model belajar melalui pembaruan bobot, sementara batch size mengatur jumlah data yang diproses dalam satu iterasi. GRU layer meningkatkan kemampuan model untuk memahami struktur hierarkis dalam teks, dan jumlah epoch yang tepat memberikan waktu belajar yang cukup tanpa menyebabkan model terlalu menghapal data pelatihan.

### 3.1 GridSearch (Hidden Size 256)

| Accuracy | Hidden Size | Dropout | Learning Rate | Batch Size | GRU Layer | Epoch |
|----------|-------------|---------|--------------|------------|-----------|-------|
| 76.88%   | 256         | 0.2     | 0.0001       | 16         | 2         | 5     |

Hidden size 256 memberikan kapasitas yang optimal bagi model untuk menangkap pola kompleks dalam teks sarkastik tanpa terlalu besar yang dapat menyebabkan overfitting. Dropout 0.2 membantu model mempertahankan generalisasi dengan mencegah ketergantungan berlebihan pada fitur tertentu. Learning rate 0.0001 memungkinkan model belajar secara stabil dan bertahap, sehingga dapat menemukan bobot optimal tanpa melewatkan minimum global. Batch size 16 menyeimbangkan antara kecepatan pelatihan dan keakuratan pembaruan gradien, memungkinkan model beradaptasi dengan baik terhadap variasi dalam data. GRU layer sebanyak 2 memberikan kedalaman yang cukup untuk memahami konteks dan struktur temporal dalam teks sarkastik. Dengan 5 epoch, model memiliki waktu yang cukup untuk menyesuaikan parameter dan mencapai akurasi 76.88% yang menunjukkan keseimbangan optimal antara waktu pelatihan dan performa.

Grafik "80/20 without stemming - F1-Score Progress" menunjukkan peningkatan performa dari 72.44% hingga 76.88%, dengan kurva F1-Score mencapai titik tertinggi pada epoch ke-5. Precision meningkat dari 73.10% menjadi 76.47%, menunjukkan model semakin akurat dalam memprediksi kasus sarkasme. Terjadi fluktuasi pada epoch ke-4 dimana semua metrik mengalami penurunan, namun model berhasil pulih dan mencapai performa terbaik pada epoch terakhir. Model ini menunjukkan keseimbangan yang baik antara precision dan recall pada akhir pelatihan, terlihat dari gap minimal antara kedua metrik tersebut (76.47% dan 77.14%). Konvergensi ketiga metrik pada epoch ke-5 mengindikasikan model telah mencapai titik optimal dalam pembelajaran tanpa tanda-tanda overfitting.

### 3.2 RandomSearch (Hidden Size 512)

Grafik "Hidden Size 512 - Performance Metrics Progress" menunjukkan fluktuasi performa dengan F1-Score meningkat dari 72.95% hingga 75.04% pada epoch ke-5. Precision mencapai puncak 76.89% pada epoch ke-2 sebelum menurun bertahap hingga kembali ke nilai awal 75.23%. Recall membentuk pola U-shape yang dimulai dari 76.14%, turun hingga 73.84% pada epoch ke-4, lalu kembali ke 76.14% pada epoch terakhir. Terlihat adanya trade-off antara precision dan recall selama pelatihan, dimana saat precision tinggi, recall cenderung rendah dan sebaliknya. Model dengan hidden size 512 ini menunjukkan konvergensi yang baik pada epoch ke-5 dengan keseimbangan antara precision dan recall, meskipun F1-Score akhirnya tidak setinggi model dengan hidden size 256.

## 4. Kesimpulan

Berdasarkan analisis yang telah dilakukan, dapat disimpulkan bahwa:

1. **Rasio Split Data**: Rasio 80:20 memberikan performa yang lebih baik dibandingkan 70:30 untuk semua metrik evaluasi, sehingga direkomendasikan untuk digunakan dalam pengembangan model deteksi sarkasme.

2. **Teknik Stemming**: Meskipun teknik Stemming meningkatkan Recall dan Precision, namun menurunkan F1-score. Pemilihan teknik ini bergantung pada prioritas: jika keseimbangan keseluruhan lebih penting, NonStemming lebih direkomendasikan.

3. **Parameter Model**: Hidden size 256 dengan dropout 0.2, learning rate 0.0001, batch size 16, dan 2 GRU layer memberikan performa terbaik dengan F1-score 76.88%. Model dengan hidden size yang lebih besar (512) tidak menunjukkan peningkatan performa.

4. **Jumlah Epoch**: 5 epoch memberikan waktu yang cukup bagi model untuk belajar dan mencapai konvergensi tanpa tanda-tanda overfitting.

Secara keseluruhan, model deteksi sarkasme terbaik diperoleh dengan konfigurasi: rasio split data 80:20, tanpa stemming, hidden size 256, dropout 0.2, learning rate 0.0001, batch size 16, 2 GRU layer, dan 5 epoch pelatihan. Model ini mencapai F1-score 76.88%, Recall 77.14%, dan Precision 76.47%, menunjukkan keseimbangan yang baik antara kemampuan mengidentifikasi kasus sarkasme dan ketepatan prediksi. 