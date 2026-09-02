# Experimental Investigation Report: Tiny Memory Bank Architectures

## Daftar Pertanyaan dan Jawaban

1. **Kenapa terjadi Nan loss pada arsitektur awal?**
   Terjadi fenomena `NaN` (Not a Number) karena penggunaan *Softmax Cross-Entropy* pada *Auxiliary Loss* dengan label *soft* (`target_sim = jax.nn.softmax(labels)`). Kombinasi eksponensial di softmax pada vektor-vektor yang belum stabil menyebabkan gradien meledak. Ini diselesaikan dengan mengubah *Auxiliary Loss* menjadi *Mean Squared Error (MSE)*.

2. **Kenapa model awal selalu memberikan prediksi bypass/kosong di tahap awal decoder?**
   Model selalu menimpa fakta memori karena pengaturan ambang batas penulisan (`memory_write_threshold = -1.0`). Akibatnya, fakta-fakta selalu tertimpa di slot yang sama. Selain itu, nilai inisialisasi bobot kemiripan (`sim_weight`) yang terlalu kecil menyebabkan *Attention* tidak tajam (terlalu *blurry*), sehingga decoder tidak dapat mengisolasi satu fakta spesifik dan lebih memilih menebak `[EOS]` karena tidak ada informasi spesifik.

3. **Kenapa informasi hilang di tengah proses sequential decoder (GRU)?**
   Arsitektur Seq2Seq standar hanya memberikan informasi dari memori bank (`h_fused`) pada status awal (`initial_state`) GRU decoder. Karena sifat *vanishing gradients* pada sel berulang, informasi tersebut luntur di langkah-langkah berikutnya, menyebabkan model lupa fakta saat mencoba menyusun kata-kata.

4. **Apa solusi yang diimplementasikan untuk bottleneck informasi GRU tersebut?**
   Saya menerapkan teknik *Concatenation Bypass*. Di setiap langkah waktu (*timestep*), vektor `h_fused` tidak hanya digunakan di *initial_state*, tetapi digabungkan (dikatenasi) secara eksplisit dengan vektor *embedding* token saat itu. Dengan demikian, fakta selalu tersedia di *input* sel decoder pada setiap langkah generasi kalimat.

5. **Bagaimana mengatasi bentrok/overwriting pada kapasitas Memory Bank?**
   Saya mengembalikan nilai `memory_write_threshold` menjadi `0.9`. Dengan batas ambang yang tinggi, fakta baru tidak akan dengan mudah menimpa memori yang sudah ada kecuali mereka benar-benar sangat mirip. Jika tidak ada yang mirip, fakta baru akan diarahkan secara sekuensial ke ruang kosong (`STATE_EMPTY`) di dalam Memory Bank.

6. **Kenapa model membutuhkan peningkatan sim_weight (ketajaman softmax)?**
   Karena *Cosine Similarity* secara matematis dibatasi di rentang `[-1, 1]`. Jika dilewatkan ke fungsi Softmax tanpa dikalikan faktor skala yang besar, distribusinya akan sangat datar (maksimal 46% vs 17% probabilitas). Ini menyebabkan memori bank mengeluarkan vektor rata-rata yang sangat *blurry*. Menginisialisasi `sim_weight` ke angka `5.0` memaksa ketajaman distribusi mencapai lebih dari 98% untuk satu fakta yang dituju.

7. **Apa tantangan Information Theory terbesar pada ukuran model ini?**
   Dengan mengikuti aturan `1 parameter : 15 token`, model hanya boleh memiliki ~138.000 parameter (dimensi memori dan hidden GRU sebesar 32). Namun, dataset aslinya memuat 59.198 fakta unik. Sebuah vektor 32 dimensi (maksimal menampung 32 *floats*) tidak akan pernah bisa menyimpan kompresi dari puluhan ribu kalimat berbeda secara tanpa cacat (*lossless*). Ini adalah hukum batas kapasitas matematis.

8. **Bagaimana pembuktian teori keterbatasan kapasitas dilakukan?**
   Saya membuktikannya dengan melakukan *Overfitting Test*. Saya memangkas *dataset* menjadi hanya subset berisi 2.000 fakta (sehingga beban mengingat model berimbang dengan kapasitas 32 dimensinya). Pada jumlah data ini, model mulai berhasil belajar dan akurasi langsung menanjak drastis secara berurutan.

9. **Apakah model berhasil melewati akurasi stuck 41%?**
   Ya! Pada *dataset* yang disesuaikan secara proporsional dengan rasio parameternya, model tidak lagi *stuck* di 41% dan berhasil menanjak secara konsisten melewati 50%, 60%, hingga menyentuh angka 73.5% dan masih menunjukkan pola peningkatan (*loss* menurun) di epoch akhir.

10. **Apa kesimpulan mengenai performa parameter vs dataset size (1:15)?**
    Rasio 1:15 sangat ekstrem untuk model bahasa berarsitektur memori (*Memory Bank*) jika jumlah kosa katanya mencapai 2000 unik. Model dengan parameter sekecil ini (*Hidden Size 32*) sangat brilian untuk mengingat ratusan hingga ribuan fakta, namun mustahil digunakan untuk menghafal 59 ribu variasi kalimat berbeda tanpa terjadi korupsi informasi di dalam ruang vektor.

11. **Apa perbedaan efektivitas antara Contrastive Cross Entropy vs MSE?**
    *MSE (Mean Squared Error)* terbukti jauh lebih stabil karena loss diukur dengan selisih Euclidean, yang tidak meledak saat vektor belum selaras. *Cross Entropy* sangat agresif dan rentan meledakkan gradien pada awal masa orientasi vektor, khususnya di arsitektur yang menggunakan *Cosine Similarity* sebagai matriks pra-proses.

12. **Apakah struktur Memory Bank asli (kapasitas 128) dapat bertahan dengan Batch 32?**
    Ya, dapat bertahan dan bekerja dengan sempurna, dengan syarat mekanisme *write threshold* berfungsi dengan benar (tidak `-1.0`). Dengan begitu, data pada masing-masing iterasi akan dimasukkan ke sel acak kosong berikutnya tanpa pernah saling menimpa fakta unik pada *batch* bersangkutan.

13. **Apa rekomendasi ke depannya jika dataset kembali ke ukuran 59.000 baris?**
    Jika *dataset* dikembalikan ke 59.000 baris, arsitektur *Information Compression* (GRU tunggal) harus dibuang. Sebagai gantinya, model harus menggunakan arsitektur *Cross-Attention* di mana memori tidak disimpan sebagai 1 vektor per fakta, melainkan sebagai sekuens embedding *(sequence of tokens)*, sehingga *decoder* bisa langsung mengambil kata ("Penulis", "Manado") dari referensi kata-kata tersebut tanpa harus di-kompres-dekompres.
