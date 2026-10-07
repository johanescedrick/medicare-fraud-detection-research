# HANDOFF: Riset Fraud Detection Klaim Kesehatan
**Backbone:** Yoo et al. (2023) · **Metode baru (satu saja):** Degree-Aware Graph Attention Network (DA-GAT) · **Waktu: 2 bulan**
Versi 2, dibuat 6 Oktober 2026. Menggantikan versi sebelumnya (4 kontribusi, 4 bulan). Dokumen berdiri sendiri: Claude yang membacanya tidak punya akses ke percakapan sebelumnya.

---

## 0. Untuk Claude yang melanjutkan (baca dulu)

**Peran.** Pendamping riset. User menulis kodenya sendiri dan butuh penjelasan metode yang jelas, cepat, dan tepat. Jangan membuat notebook atau paket kode kecuali diminta.

**Gaya.** Bahasa Indonesia, istilah teknis Inggris boleh. Jelaskan dari dasar dengan contoh kecil. User ingin hasil yang cepat dan tepat dengan inovasi yang jelas.

**Aturan kerja:**
- Tandai mana yang **terverifikasi** (dibaca full-text) dan mana yang **belum**. Jangan mengarang rumus atau angka. Kalau perlu detail paper, minta user mengunggah PDF-nya.
- Kebaruan harus bisa dipertahankan di depan reviewer. Jangan menjual "memakai GAT" sebagai novelty (lihat 2.1).
- User sempat bingung karena penjelasan kebaruan berputar-putar. Kalau ditanya "apa yang baru", mulai dari tabel di **1.1**, jawab langsung, dan hindari menyodorkan opsi baru.

**Kesalahan yang pernah terjadi di sesi ini (jangan diulang):**
1. Menulis "~9% fraud di level klaim". Salah: 9% = level **provider**; level klaim ≈ 38%.
2. Merekomendasikan RDPGL sebagai backbone, padahal label *organized fraud*-nya dianotasi pakar NHSA dan tidak ada di Kaggle publik.
3. Menyebut detail MHAMFD (Lu et al. 2023) padahal hanya abstrak yang terbaca.
4. Menyebut Muhammad et al. "kontradiktif" dengan Yoo. Berlebihan: baseline ML mereka tanpa centrality, selisihnya ~1 poin.
5. Menyebut edge `CHARGE` untuk GNN sebagai "satu edge per pasangan unik". Tabel 3 Yoo menunjukkan jumlah edge `CHARGE` **sama dengan jumlah klaim** (330.288 / 122.543 / 103.872), jadi graf mereka kemungkinan **satu edge per klaim**. Belum pasti.
6. Mengusulkan "attention GNN" sebagai novelty, padahal Yoo sudah menguji GAT.

---

## 1. Konteks dan keputusan terkunci

- **Tujuan:** paper tentang health insurance fraud detection. Asal minat: hackathon BPJS Kesehatan 2021. Tugas akademik: minimal 3 paper relevan.
- **Waktu:** **2 bulan**. User ingin yang cepat, tepat, dan inovatif. **Graph Attention Network wajib ada.** Cukup **satu** metode baru.
- **Status user (tidak dikonfirmasi):** kemungkinan mahasiswa/peneliti pemula.

| Keputusan | Alasan |
|---|---|
| Dataset = **Kaggle** (bukan BPJS) | Punya ID relasional eksplisit; data BPJS tabular flat tanpa ID pasien/dokter/faskes |
| Backbone = **Yoo et al. 2023** (dikonfirmasi ulang) | Dataset sama, baseline ML dan GNN sudah ada. User memilih tetap Yoo karena IEEE Access berkuartil tinggi. Agregator pihak ketiga mencatat best quartile Q1 (Computer Science/Engineering), **bukan sumber resmi**: verifikasi di scimagojr.com untuk kategori dan tahun yang dipakai. Kuartil menilai jurnal, bukan tiap paper, dan tidak menjamin paper kita diterima |
| Alternatif backbone yang ditolak: Argade & Gandeeban 2026, *Benchmarking TabPFN...*, ISSF 2026 (Atlantis Press), atlantis-press.com/article/126025139.pdf | Lebih sederhana (fitur agregat level provider, satu graf "berbagi ≥1 pasien", GraphSAGE 2 layer, split 80/20 tunggal tanpa seed; LR ROC-AUC 0,953 / PR-AUC 0,742 vs GraphSAGE 0,954 / 0,740) dan punya kode (github.com/dipakml/healthcare-provider-fraud-benchmark, **belum dicek aktif**). Ditolak sebagai backbone, tetapi **berguna sebagai referensi implementasi level provider** (daftar fitur agregat, split, metrik cost-sensitive) dan sebagai bukti pendukung bahwa fitur volume saja sudah memberi ROC-AUC > 0,95. Catatan: selisih antar model di paper itu terlalu kecil untuk disimpulkan karena hanya satu split |
| Metode baru = **DA-GAT** (GAT sadar degree) | Menjawab kelemahan hasil Yoo; implementasi kecil di atas GATv2 |
| Graf = **2 edge milik Yoo saja** (`CHARGE`, `PROJECT_PROVIDER`) | Hemat waktu; perbandingan apple-to-apple |
| Level metrik utama = **provider** (+ klaim-terbobot sebagai pelengkap) | Label melekat di provider; klaim-level untuk membandingkan dengan Yoo |

**Dikeluarkan dari cakupan** (jadi future work): edge `DIAGNOSIS_SIM` dan `GEO`; relation-attention per node (`RelAttnGNN`); RE-GraphSAGE; GAT dengan fitur edge; explainability penuh dan validasi pakar; RDPGL penuh; studi kasus BPJS.

### 1.1 Ringkasan inti: Yoo vs yang kita tawarkan (mulai dari sini)

| | **Yoo et al. (yang sudah ada)** | **Yang ditawarkan (baru)** |
|---|---|---|
| Data | Kaggle Medicare | Sama |
| Graf | provider–pasien dan provider–provider (lewat physician yang sama) | Sama |
| Pertanyaan | GNN atau ML+centrality, mana yang lebih baik? | Kenapa GNN kalah, dan bisakah GNN-nya diperbaiki? |
| Hasil | ML+centrality menang, GNN kalah | Hasil itu diuji ulang dulu, lalu dicari sebabnya |
| Penyebab GNN kalah | Hanya dugaan penulis ("information loss"), tidak diuji | Hipotesis: GNN tidak tahu **jumlah** pasien/klaim provider (volume), padahal centrality menangkapnya. Diuji |
| Kontrol volume | Tidak ada | Ada: model ML dengan fitur jumlah klaim/pasien tanpa graf (M2) |
| Model GNN | GraphSAGE, GAT, HAN, HGT biasa | **DA-GAT** (GAT sadar degree) |
| Cara GAT bekerja | Rata-rata berbobot dari tetangga | Rata-rata berbobot dikali faktor degree; seberapa besar faktornya dipelajari model ($\gamma_r$) |
| Bukti | Hanya skor F1 | Skor F1, plus probe yang menunjukkan apakah embedding GAT biasa menyimpan informasi volume (D1) |

**Intuisi.** GAT itu seperti rata-rata nilai kelas: rata-rata tidak memberi tahu jumlah muridnya. Provider dengan 10 pasien dan provider dengan 1.000 pasien bisa terlihat sama bagi GAT, padahal volume besar sering menandakan fraud. DA-GAT menambahkan informasi "berapa banyak" itu ke GAT.

**Yang baru ada dua, bukan data/graf/GAT-nya:**
1. **Diagnosis:** apakah keunggulan centrality di Yoo sebenarnya sebagian besar efek volume provider?
2. **Perbaikan:** DA-GAT, yaitu GAT yang tahu jumlah tetangganya.

**Catatan jujur.** Ide pengali degree sudah ada di literatur GNN umum (PNA; lihat 5.4). Kebaruan ada pada diagnosis di data klaim kesehatan dan penerapannya ke GAT per-relasi. Jika cukup memberi degree sebagai fitur input (G1 ≈ G3), kontribusi bergeser menjadi diagnosisnya saja, dan itu tetap sah (lihat 5.2).

---

### 1.2 Ringkas: metode dan model baru yang ditawarkan

Versi satu halaman. Detail lengkap di 4.6 (DA-GAT), 4.7 (eksperimen), 5.2 (skenario hasil).

**Pertanyaan yang dijawab**
1. Apakah keunggulan centrality di Yoo sebenarnya sebagian besar efek **volume** provider?
2. Apakah GAT biasa gagal menangkap volume dari struktur graf?
3. Kalau iya, bisakah GAT diperbaiki agar tahu volume?

**Metode baru: DA-GAT (Degree-Aware GAT)**
- GAT menghitung **rata-rata berbobot** tetangga, jadi hasilnya tidak tergantung jumlah tetangga. Provider dengan 10 pasien dan 1.000 pasien bisa terlihat sama.
- DA-GAT mengalikan hasil agregasi per relasi dengan faktor degree $s_r(d)=\big(\tfrac{\log(d+1)}{\delta_r}\big)^{\gamma_r}$.
- $d$ = degree pada relasi itu; $\delta_r$ = rata-rata $\log(d+1)$; $\gamma_r$ = eksponen yang **dipelajari**, satu per jenis edge per layer.
- $\gamma_r=0$ berarti GAT biasa, $\gamma_r=1$ berarti faktor penuh. Nilai hasil belajar bisa dibaca: edge mana yang volumenya penting.

**Beda dengan Yoo**

| | Yoo et al. | Yang ditawarkan |
|---|---|---|
| Data, graf | Kaggle, 2 jenis edge | Sama |
| Model GNN | GraphSAGE, GAT, HAN, HGT biasa | **DA-GAT** |
| Agregasi GAT | rata-rata berbobot | rata-rata berbobot × faktor degree |
| Kontrol volume | tidak ada | ada (M2) |
| Sebab GNN kalah | hanya dugaan penulis ("information loss") | diuji lewat probe (D1) |
| Seleksi ambang | tidak dijelaskan | dari F1 maksimum di validation |
| Level metrik | tidak jelas | provider (utama) + klaim (pembanding) |

**Eksperimen inti** (fitur dipisah: X_mean tanpa volume, X_vol, X_cent; GNN hanya diberi X_mean)

| Kode | Model | Input | Untuk apa |
|---|---|---|---|
| M0 | LR, RF, XGB, LGBM, MLP | X_mean | Dasar, tanpa volume dan graf |
| M1 | sama | X_mean + X_cent | Replikasi "ML+centrality" Yoo |
| M2 | sama | X_mean + X_vol | **Kontrol volume** |
| M3 | sama | X_mean + X_vol + X_cent | Centrality menambah di atas volume? |
| G0 | GAT | X_mean | Replikasi GNN Yoo |
| G1 | GAT | X_mean + log-degree | Cukupkah degree sebagai fitur input? |
| G2 | DA-GAT, γ dikunci 1 | X_mean | Faktor penuh |
| G3 | **DA-GAT, γ dipelajari** | X_mean | **Model utama** |
| D1 | Probe linear | embedding G0 vs G3 | Apakah embedding menyimpan volume? (R² Ridge) |

**Kemungkinan hasil dan klaimnya**

| Hasil | Yang ditulis |
|---|---|
| M2 ≈ M1, G0 jauh di bawah, G3 menutup selisih | Keunggulan centrality di Yoo sebagian besar efek volume; DA-GAT memulihkannya |
| G1 ≈ G3 | Degree penting, faktor tidak perlu; kontribusi bergeser ke diagnosis |
| G0 ≈ M2 | GAT sudah menangkap volume; hipotesis ditolak, ditulis sebagai replikasi terkontrol |
| M1 ≫ M2 | Centrality membawa sinyal selain volume; temuan Yoo diperkuat |

**Yang perlu diwaspadai**
- Ide faktor degree **sudah ada** di literatur GNN umum (PNA, Corso et al. 2020). Kebaruan ada pada diagnosis di data klaim kesehatan dan penerapan ke GAT per-relasi. Belum dicek apakah ada yang sudah melakukan persis ini (lihat 5.4).
- Efeknya bisa kecil: provider train hanya ≈ 3.200. Pakai banyak seed, laporkan rentang, jangan hanya selisih rata-rata.
- Semua ini **belum diuji**: hipotesis volume dan DA-GAT belum dijalankan di data asli (lihat 7).

---

## 2. Paper

Hanya **tiga** paper di bawah yang dibaca full-text. Sisanya hanya abstrak/cuplikan.

### 2.1 Backbone: Yoo, Shin, Kyeong (2023) · *terverifikasi*

*Medicare Fraud Detection Using Graph Analysis: A Comparative Study of Machine Learning and Graph Neural Networks.* IEEE Access 11:88278–88294. DOI 10.1109/ACCESS.2023.3305962. **Kode resmi tidak ditemukan.**

**Masalah.** Fraud Medicare sering berupa kolusi provider–dokter–pasien. Apakah struktur graf membantu, dan lebih baik dipakai lewat GNN atau lewat fitur graph centrality pada ML biasa?

**Data.** Kaggle rohitrox. Gabungan akhir 556.703 klaim, 56 fitur, 212.232 berlabel fraud (≈38%; mereka sebut "hampir tanpa class imbalance"). Split **per provider** 6:2:2:

| Split | Klaim | Provider | Beneficiary | Edge PROJECT_PROVIDER |
|---|---|---|---|---|
| Train | 330.288 | 3.246 | 111.236 | 3.486 |
| Val | 122.543 | 1.082 | 59.570 | 342 |
| Test | 103.872 | 1.082 | 53.618 | 308 |

**Metode.**
- *GNN:* graf heterogen, node provider dan beneficiary. Edge `CHARGE` (provider–beneficiary) dan `PROJECT_PROVIDER` (provider–provider, proyeksi bipartit lewat physician bersama). **GraphSAGE, GAT, HAN, HGT.** Klasifikasi node provider. Dropout; mini-batch dengan neighbor sampling dan HGSampling. Dilatih 10 kali.
- *ML + centrality:* dua graf bipartit (provider–physician dan provider–beneficiary). Degree, eigenvector, closeness, PageRank sebagai fitur tambahan untuk LR (forward selection, p < 0,05), RF, XGBoost, LightGBM, MLP.
- *Metrik:* precision, recall, F1, AUROC. Model terbaik dipilih berdasarkan F1.

**Hasil.**

| Model | Precision | Recall | F1 | AUROC |
|---|---|---|---|---|
| ML tanpa centrality (rentang) | 0,42–0,52 | 0,11–0,33 | 0,19–0,39 | 0,53–0,58 |
| LR + centrality provider–beneficiary (terbaik) | 0,56 | 0,76 | 0,65 | 0,74 |
| GNN terbaik (HAN, heterogeneous mini-batch) | — | 0,52 | 0,51 | 0,74 |

- ML+centrality mengungguli GNN terbaik: +4 pp precision, +24 pp recall, +14 pp F1.
- Centrality dari graf provider–beneficiary jauh lebih berguna daripada provider–physician (LR: F1 0,19 → 0,37 dengan physician; → 0,65 dengan beneficiary).
- Di regresi logistik hanya **`Provider_PageRank`** signifikan (alasan penulis: multikolinearitas).
- Waktu training HAN vs LR: **abstrak 250–300×, Bab IV-C 250–350×** (inkonsistensi internal; kutip hati-hati).
- Estimasi penghematan 3,1 miliar euro / 5 miliar dolar memakai asumsi kasar; jangan direplikasi sebagai klaim.

**Keterbatasan yang diakui penulis:** dataset hampir seimbang; cara tabular→graf (agregasi jumlah node embedding) bukan satu-satunya; asal-usul data Kaggle tidak jelas. Penulis juga menduga GNN kalah karena **konversi tabular ke graf menyebabkan information loss**.

**Titik masuk untuk kita (analisis kita, bukan klaim paper):**
1. GNN kalah dari ML+centrality, dan yang menang terutama **PageRank/degree**.
2. Degree dan PageRank pada graf provider–pasien kemungkinan besar mencerminkan **volume provider**. Baseline ML klaim-level Yoo tidak punya informasi volume, jadi lonjakan F1 0,19 → 0,65 bisa sebagian besar efek volume. **Hipotesis, belum terbukti.**
3. Metrik tidak eksplisit di level mana (provider atau klaim). Dari Tabel 3 dan fitur klaim-level di Tabel 4–6, kemungkinan ML mereka dilatih per klaim dengan label diwariskan dari provider. **Belum pasti.**
4. GAT sudah mereka uji. Detail setup GAT pada graf dua tipe node tidak dijelaskan.

### 2.2 Pendukung: Ma et al. (2023), RDPGL · *terverifikasi*

*Fighting against Organized Fraudsters Using Risk Diffusion-based Parallel Graph Neural Network.* IJCAI-23, hlm. 6138–6145. Kode (dari teks paper, **tidak dicek aktif**): github.com/finint/antifraud.

Memakai Kaggle yang sama (38,1% fraud). Dua komponen paralel: GAT heterogen lokal pada graf beneficiary–provider (attention Eq. 1: $\alpha_{ij}=\mathrm{softmax}_j(\sigma(u^\top[W_bh_{b,i}\Vert W_ph_{p,j}]))$, pesan dari diri/edge/tetangga, embedding edge-klaim) dan risk diffusion global pada graf klaim–klaim (random walk + personalized PageRank + pre-training GCN). Label *organized fraud* dianotasi pakar NHSA, **tidak publik**. Hasil contoh (organized, outpatient): AUC 0,8713, recall 0,8302, F1 0,7982. Ablation: kedua komponen dibutuhkan.

**Peran untuk kita:** bukti bahwa attention pada graf provider–beneficiary dipakai di literatur terbaru (rujukan GAT), dan bahwa informasi di luar tetangga lokal membantu. Komponen risk diffusion dan labelnya **tidak** dipakai.

### 2.3 Pendukung: Muhammad et al. (2025) · *terverifikasi*

*Fraud detection and explanation in medical claims using GNN architectures.* Sci. Rep. 15:41734. DOI 10.1038/s41598-025-22910-6. Kode (dari teks paper, **tidak dicek aktif**): github.com/Tachy-Health/GNN.

Level **aktivitas** pada data proprietary MENA. Label = aktivitas **ditolak karena alasan medis** (bukan fraud terkonfirmasi), dataset diseimbangkan. Model: HINormer, HybridGNN, RE-GraphSAGE; explainer GNNExplainer dan PGExplainer.

| Macro-F1 | HINormer | HybridGNN | RE-GraphSAGE |
|---|---|---|---|
| Small (491k) | **0,84** | 0,71 | 0,83 |
| Medium (2 jt) | 0,81 | 0,58 | **0,84** |
| Large (6 jt) | **0,82** | 0,51 | 0,79 |

Baseline ML (small): SVM 0,81, RF 0,83, hampir setara GNN. Preprocessing HINormer 147–1.840 menit vs RE-GraphSAGE 1–31 menit. Validasi pakar medis: GNNExplainer+HINormer sesuai pakar; PGExplainer+HINormer **tidak** sesuai (kasus tes antibodi Rubella).

**Peran untuk kita:** rujukan bahwa model sederhana sering setara model kompleks, dan bahwa penjelasan dengan fidelity tinggi bisa salah secara medis (alasan interpretability dijadikan future work, bukan klaim).

### 2.4 Paper lain (hanya abstrak/cuplikan; untuk related work)

| Paper | Catatan |
|---|---|
| du Preez et al. 2024, *Artif. Intell. Med.*, DOI 10.1016/j.artmed.2024.103061 | SLR 137 studi; 96 memakai data AS; tantangan: label terbatas, tanpa benchmark |
| Nabrawi & Alanazi 2023, *Risks* 11(9):160 | RF/LR/ANN + SMOTE, data Arab Saudi, tanpa graf |
| Lu et al. 2023 (MHAMFD), *BMC MIDM* 23:62 | AHIN + attention tiga level. **Hanya abstrak dibaca** |
| Hong et al. 2024 (MHGSL), *Heliyon* 10 | Graf multi-view. Hanya abstrak/cuplikan |
| Chandola 2013, Branting 2016, Herland 2018 | Fitur graf + ML untuk fraud Medicare (dikutip Yoo), belum dibaca |

---

## 3. Data

### 3.1 Kaggle (dataset utama)

**https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis** (dari referensi Yoo [62] dan RDPGL; **belum dicek aktif**; unduhan butuh login). Pakai hanya 4 file `Train_*`.

| File | Isi | Kolom kunci |
|---|---|---|
| `Train_Inpatientdata` | rawat inap | `ClaimID`, `BeneID`, **`Provider`**, `Attending/Operating/OtherPhysician`, `AdmissionDt`, `DischargeDt`, `ClmDiagnosisCode_1..10`, `ClmProcedureCode_1..6`, `InscClaimAmtReimbursed` |
| `Train_Outpatientdata` | rawat jalan | sama, tanpa kolom admission/discharge |
| `Train_Beneficiarydata` | pasien | `BeneID`, `DOB`, `DOD`, `Gender`, `Race`, `State`, `County`, 11 `ChronicCond_*` (1 = ya, 2 = tidak), 4 kolom amount tahunan |
| `Train-*.csv` | label | `Provider`, `PotentialFraud` (Yes/No) |

Kolom provider bernama `Provider`, bukan `ProviderID`.

**Ukuran (dari ingatan, verifikasi setelah unduh):** ≈ 5.410 provider, ≈ 138.556 pasien, ≈ 40k klaim inpatient, ≈ 517k outpatient. Fraud rate **provider ≈ 9–10%**, **klaim ≈ 38%**.

**Prosedur (Yoo Fig. 1):** union inpatient+outpatient (flag `is_ip`) → dedup `ClaimID` → merge `BeneID` → merge `Provider`. Split **per provider**, stratified, seed tetap.

**Catatan:** label di level provider, klaim mewarisinya. Asal-usul data tidak jelas (Yoo juga mencatat), jadi generalisasi tidak bisa diklaim.

### 3.2 BPJS Hackathon 2021 (di luar cakupan 2 bulan)

`train.csv` (pemisah `;`): 160.174 baris × 53 kolom, label 50/50, satu baris = satu klaim. Fitur administratif (`kdkc` 126 nilai, `dati2` 486, `typeppk`, `jkpst`, `umur`, `los`) dan klinis (`diagprimer`, `dx2_*`, `proc*_*`). **Tidak ada ID pasien/dokter/faskes**, sehingga graf harus dikonstruksi buatan. Jadikan future work.

---

## 4. Metode

### 4.1 Urutan kerja

1. Gabung data, split provider, cetak ringkasan (cocokkan dengan Tabel 3 Yoo).
2. Fitur provider (4.2), graf bipartit, centrality (4.4).
3. Baseline ML, termasuk **kontrol volume**.
4. Baseline GNN (GAT + HAN) pada graf Yoo.
5. **DA-GAT** dan ablation (4.6).
6. Probe degree, tabel hasil, penulisan.

### 4.2 Definisi fitur: tiga kelompok (inti desain eksperimen)

Hipotesis kita adalah bahwa GAT tidak menangkap informasi **volume**. Supaya hipotesis ini bisa diuji, fitur input GNN **tidak boleh** sudah memuat volume. Jadi pisahkan:

| Kelompok | Isi (usulan; boleh disesuaikan) |
|---|---|
| **X_mean** (intensif, tanpa volume) | rata-rata/std per klaim: reimbursement, deductible, durasi klaim, LOS, jumlah diagnosis/prosedur/physician; usia rata-rata; % pria, meninggal, renal, rawat inap; rata-rata 11 kondisi kronis, 4 amount, 2 months-coverage |
| **X_vol** (volume) | `n_claims`, `n_ip`, `n_bene`, `n_physicians`, `reimb_sum`, `reimb_max`, `claims_per_bene`, `reimb_per_bene`, `state_nunique`, `county_nunique` |
| **X_cent** | 8 kolom: {degree, eigenvector, closeness, PageRank} × {graf provider–beneficiary, graf provider–physician} |

Catatan: karena Yoo tidak jelas soal edge, hitung degree dalam dua versi: jumlah **pasien unik** dan jumlah **klaim**.

### 4.3 Graf (persis milik Yoo)

Node `provider` (target) dan `beneficiary`. Edge:
- `CHARGE`: provider–beneficiary. **Default: satu edge per klaim** (sesuai Tabel 3 Yoo). Variasi robustness: satu edge per pasangan unik.
- `PROJECT_PROVIDER`: dua provider berbagi ≥ 1 physician (proyeksi bipartit). Hitung dari tiga kolom physician dan dari `AttendingPhysician` saja; pilih yang jumlah edge-nya paling dekat Yoo (3.486 pada graf train) dan laporkan. Beri batas jumlah provider per physician agar tidak terbentuk clique raksasa.

Pilihan desain yang harus ditulis sebagai deviasi: graf **transduktif** (satu graf, mask train/val/test) sementara Yoo membangun graf per split.

### 4.4 Centrality (Yoo Eq. 2–5)

| Ukuran | Rumus | Makna |
|---|---|---|
| Degree | $C_d(v_i)=d_i$ | jumlah koneksi (proksi volume) |
| Eigenvector | $C_e(v_i)=\frac1\lambda\sum_jA_{j,i}C_e(v_j)$ | penting jika tetangga penting |
| Closeness | $C_c(v_i)=1/\bar l_{v_i}$ | kedekatan rata-rata ke node lain |
| PageRank | $C_p(v_i)=\alpha\sum_jA_{j,i}\frac{C_p(v_j)}{d_j^{out}}+\beta$ | pengaruh diteruskan lewat tetangga |

Contoh graf kecil (P1–B1,B2,B3; P2–B3,B4; P3–B5), dihitung dengan networkx:

| | P1 | P2 | P3 |
|---|---|---|---|
| Degree | 3 | 2 | 1 |
| Eigenvector | 0,60 | 0,37 | 0,00 |
| Closeness (Wasserman–Faust) | 0,446 | 0,357 | 0,143 |
| PageRank (α = 0,85) | 0,215 | 0,150 | 0,125 |

Jebakan: pada graf dengan komponen terpisah, eigenvector terkonsentrasi di komponen terbesar (P3 = 0 adalah artefak), dan closeness sederhana memberi node terisolasi nilai tertinggi (gunakan koreksi Wasserman–Faust dan catat). Closeness eksak mahal di graf besar; aproksimasi dari ~300 pivot acak berkorelasi ≈ 0,98 dengan eksak pada data sintetis (belum diuji di data asli). Eigenvector di graf bipartit sebaiknya lewat `scipy eigsh` karena power-iteration biasa dapat gagal konvergen. Centrality dihitung dari seluruh graf tanpa memakai label (bukan kebocoran label; sebutkan di paper).

### 4.5 Baseline

- **ML:** LR, RF, XGBoost, LightGBM, MLP. Ambang keputusan dari F1 maksimum di validation.
- **GNN:** **GAT** (wajib) dan **HAN** (GNN terbaik Yoo) sebagai referensi. GraphSAGE dan HGT opsional bila waktu ada. PyG: `HeteroConv` + `GATv2Conv`/`GATConv`, `HANConv`, `HGTConv`, `SAGEConv`. Untuk GAT pada graf heterogen, pakai satu GAT per tipe edge lalu jumlahkan (pilihan desain; catat sebagai asumsi karena Yoo tidak merincinya).
- Kerangka message passing (Yoo Eq. 1): $H_u^{(l+1)}=\mathrm{Up}^{(l)}\big(H_u^{(l)},\mathrm{Agg}^{(l)}(\{H_v^{(l)}:v\in N(u)\})\big)$.
- Rumus GAT/HAN/HGT diambil dari paper aslinya, **bukan dari Yoo**.

### 4.6 Metode baru: DA-GAT (Degree-Aware GAT)

**Motivasi (hipotesis, belum terbukti).** Output GAT adalah rata-rata berbobot (softmax) dari tetangga, jadi besarnya agregasi tidak tergantung jumlah tetangga. Informasi "berapa banyak tetangga" (degree/volume) justru yang ditangkap centrality dan diduga mendorong keunggulan ML+centrality di Yoo. Argumen serupa ada di literatur (Xu et al. 2019 tentang aggregator rata-rata yang tidak bisa "menghitung"; Corso et al. 2020, PNA, yang menambahkan degree scaler).

**GAT standar** (untuk satu relasi $r$, $K$ head):
$$\alpha_{ij}=\frac{\exp\big(\mathrm{LeakyReLU}(a^\top[Wh_i\Vert Wh_j])\big)}{\sum_{k\in\mathcal N(i)}\exp\big(\mathrm{LeakyReLU}(a^\top[Wh_i\Vert Wh_k])\big)},\qquad z_i^{r}=\Big\Vert_{k=1}^{K}\sum_{j\in\mathcal N_r(i)}\alpha^{k}_{ij}W^{k}h_j$$

**DA-GAT** menambahkan scaler degree per relasi pada pesan terbobot sebelum relasi dijumlahkan:
$$h_i'=\sigma\Big(W_{self}h_i+\sum_{r}s_r(d_i^{r})\,z_i^{r}\Big),\qquad s_r(d)=\Big(\frac{\log(d+1)}{\delta_r}\Big)^{\gamma_r}$$

- $d_i^{r}$ = in-degree node $i$ pada relasi $r$ (bila `CHARGE` satu-edge-per-klaim, ini = jumlah klaim).
- $\delta_r$ = rata-rata $\log(d+1)$ pada node bertipe tujuan (dihitung dari struktur graf, tanpa label).
- $\gamma_r$ = **eksponen yang dipelajari**, satu skalar per relasi per layer, diinisialisasi 1. $\gamma_r=0$ berarti GAT biasa; $\gamma_r=1$ berarti scaler PNA. Nilai $\gamma_r$ yang dipelajari bisa dibaca: relasi mana yang degree-nya penting.
- Node dengan $d=0$ tidak punya pesan; mask agar log tidak tak terhingga.
- Implementasi: keluaran `GATv2Conv` per relasi dikali $s_r(d)$ (hitung degree dengan `torch_geometric.utils.degree`).

**Varian untuk ablation (satu kode, beda pengaturan):** $\gamma$ dikunci 0 (GAT biasa), $\gamma$ dikunci 1 (PNA-style), $\gamma$ dipelajari (DA-GAT penuh).

### 4.7 Matriks eksperimen

Semua GNN memakai input X_mean kecuali dinyatakan lain, supaya volume hanya bisa datang dari struktur.

| ID | Model | Input | Tujuan |
|---|---|---|---|
| E0 | LR dan XGBoost, **klaim-level** | fitur klaim ± centrality | Sanity check vs Yoo (lihat 6, gate 1) |
| M0 | LR, RF, XGB, LGBM, MLP | X_mean | Baseline tanpa volume dan tanpa graf |
| M1 | idem | X_mean + X_cent | Replikasi "ML+centrality" |
| M2 | idem | X_mean + X_vol | **Kontrol volume** |
| M3 | idem | X_mean + X_vol + X_cent | Apakah centrality menambah di atas volume? |
| G0 | GAT | X_mean | Replikasi GNN Yoo |
| H0 | HAN | X_mean | GNN terbaik Yoo |
| G1 | GAT | X_mean + log-degree | Jawaban sederhana: kasih saja degree sebagai input |
| G2 | DA-GAT, γ=1 tetap | X_mean | Scaler PNA-style |
| G3 | **DA-GAT, γ dipelajari** | X_mean | Metode utama |
| G4 | GAT dan DA-GAT | X_mean + X_cent | Fusi dengan centrality |
| D1 | Probe linear | embedding provider terakhir | Seberapa jelas embedding menyandi log(n_claims) dan log(n_bene)? (R² dengan Ridge, di provider test) pada G0 vs G3 |

### 4.8 Protokol evaluasi (dikunci)

- **Split:** per provider, stratified, 60/20/20, seed tetap untuk perbandingan utama. Untuk tabel akhir model kunci (M1, M2, G0, G1, G3), ulangi dengan 5 split acak.
- **Model:** hidden 64, 2 layer, 4 head, dropout 0,3, cross-entropy berbobot kelas ($w_c=N/(2N_c)$), AdamW, early stopping pada AUPRC validation. Nilai awal yang disarankan; tuning ringan di validation saja.
- **Seed:** 10 per konfigurasi GNN (seperti Yoo); laporkan rata-rata ± std.
- **Metrik:** precision, recall, F1 (kelas fraud), AUROC, **AUPRC**; threshold dari F1 maksimum di validation. Level provider (utama) dan klaim-terbobot (tiap klaim mewarisi prediksi provider, dibobot jumlah klaim).
- **Efisiensi:** waktu training dan inferensi (isu inti Yoo).
- Uji beda: bandingkan G3 vs G0/G1/M1 per seed berpasangan; laporkan interval, jangan hanya selisih rata-rata.

---

## 5. Inovasi dan klaim paper

### 5.1 Pertanyaan riset

- **RQ1:** Seberapa besar keunggulan ML+centrality (Yoo) sebenarnya efek **volume** provider? (M0 vs M1 vs M2 vs M3)
- **RQ2:** Apakah GAT standar gagal menyandi degree dari struktur (D1), dan apakah **agregasi sadar degree** menutup kesenjangan itu? (G0 vs G1 vs G2 vs G3)
- **RQ3:** Apakah DA-GAT menyamai atau melampaui ML+centrality tanpa perlu menghitung centrality terpisah? (G3 vs M1; G4)

### 5.2 Skenario hasil dan apa yang ditulis

| Hasil | Interpretasi | Klaim paper |
|---|---|---|
| M2 ≈ M1; G0 ≪ M2; G3 menutup selisih | Hipotesis didukung penuh | "Keunggulan centrality di Yoo sebagian besar efek volume; GAT sadar-degree memulihkannya" |
| G1 ≈ G3 | Degree penting, tapi scaler tidak diperlukan | Kontribusi = diagnosis; rekomendasi praktis: tambahkan degree sebagai fitur |
| G0 ≈ M2 | GAT sudah memulihkan volume | Hipotesis ditolak; ceritakan sebagai replikasi dengan kontrol |
| M1 ≫ M2 | Centrality membawa sinyal struktur selain volume | Temuan Yoo diperkuat dengan kontrol |

Hasil negatif yang rapi lebih sulit lolos di venue ber-impact tinggi, tetapi tetap bisa diterbitkan bila kontrolnya kuat. Siapkan framing untuk dua kemungkinan sejak awal.

### 5.3 Kontribusi (bersyarat pada hasil)

1. **Diagnosis:** dekomposisi keunggulan centrality di Yoo menjadi komponen volume dan struktur (kontrol M2, M3), sesuatu yang tidak dilakukan Yoo.
2. **Metode:** DA-GAT, GAT dengan scaler degree per relasi dan eksponen yang dipelajari ($\gamma_r$), diterapkan pada graf klaim kesehatan.
3. **Bukti mekanistik:** probe degree (D1) dan nilai $\gamma_r$ yang dipelajari.
4. **Replikasi terkontrol** hasil Yoo dengan pelaporan provider-level dan klaim-level.

### 5.4 Risiko kebaruan

- Ide scaler degree pada GNN **sudah ada** (PNA, Corso et al. 2020; degree encoding di Graphormer, Ying et al. 2021). Kebaruan DA-GAT bukan layer umum yang baru, melainkan (a) diagnosis pada data klaim kesehatan, (b) penerapan pada GAT per-relasi, (c) eksponen yang dipelajari. Saya **belum memverifikasi** bahwa tidak ada paper yang sudah melakukan persis ini. Cek Google Scholar ("degree-aware attention fraud detection", "degree scaler graph attention healthcare").
- Efeknya mungkin kecil. Provider train hanya ≈ 3.200, jadi gunakan banyak seed dan laporkan rentangnya.
- Jika G1 (degree sebagai input) sudah cukup, nilai tambah DA-GAT terhadap G1 mungkin nol. Itu bukan kegagalan, tetapi mengubah klaim (lihat 5.2).

### 5.5 Jangan diklaim

- "Attention GNN" sebagai novelty (Yoo sudah memakai GAT).
- DA-GAT sebagai arsitektur umum yang lebih baik di luar data ini.
- Bahwa bobot attention adalah penjelasan kausal.
- Generalisasi ke BPJS/Indonesia dari hasil Kaggle.
- Perbandingan langsung dengan angka RDPGL (label berbeda).
- Angka penghematan Yoo sebagai fakta.

### 5.6 Kerangka paper

1. **Pendahuluan:** masalah; celah (GNN kalah dari ML+centrality tanpa penjelasan; konfound volume); kontribusi.
2. **Related work:** SLR (du Preez) → fitur graf + ML (Chandola, Branting, Herland, Yoo) → GNN fraud kesehatan (RDPGL, Muhammad) → aggregator sadar degree (Xu, Corso).
3. **Data dan graf:** Kaggle, prosedur gabung, split, kelompok fitur, deviasi dari Yoo.
4. **Metode:** centrality, baseline, DA-GAT.
5. **Eksperimen dan hasil:** E0–D1, tabel utama, ablation γ, probe.
6. **Diskusi:** interpretasi, batasan, future work (edge klinis/geografis, explainability, BPJS).
7. **Kesimpulan.**

### 5.7 Calon venue (belum dicek scope/biaya/waktu review)

IEEE Access (tempat Yoo), Scientific Reports (tempat Muhammad), BMC Medical Informatics and Decision Making, Expert Systems with Applications. Untuk batas 2 bulan, konferensi regional atau jurnal open access dengan review cepat lebih realistis. **Verifikasi sendiri** sebelum memutuskan.

---

## 6. Rencana 8 minggu dengan titik keputusan

| Minggu | Pekerjaan | Gate |
|---|---|---|
| 1 | Unduh, gabung, split; ringkasan cocok dengan Tabel 3 Yoo; fitur X_mean/X_vol; graf; centrality | Jumlah klaim dan provider mendekati Yoo |
| 2 | M0–M3, E0 | **Gate 1:** pola Yoo muncul di E0 (LR F1 ≈ 0,19 tanpa centrality → ≈ 0,65 dengan centrality provider–beneficiary)? Jika tidak, periksa langkah gabung/split sebelum lanjut |
| 3 | G0, H0 | **Gate 2:** GNN (G0/H0) lebih lemah dari M1? Jika G0 sudah ≥ M1, motivasi DA-GAT berubah; kembali ke 5.2 |
| 4–5 | G1, G2, G3, G4, D1 (10 seed) | **Gate 3 (akhir minggu 5):** apakah G3 > G0 secara berpasangan? Jika tidak, beralih ke paper "diagnosis" (kontribusi 1 dan 3) |
| 6 | Robustness: edge per klaim vs unik, pilihan δ, 5 split | — |
| 7 | Tabel, gambar, tulis hasil dan metode | — |
| 8 | Tulis sisa, revisi, buffer | — |

**Pemangkasan bila waktu mepet** (tulis sebagai deviasi dari Yoo di paper): hitung hanya **degree dan PageRank**, karena di regresi logistik Yoo hanya PageRank yang signifikan; dan lewati HAN (H0), cukup GAT sebagai pembanding GNN.

---

## 7. Status dan hal yang belum terverifikasi

**Sudah:** pemetaan paper; penentuan backbone; penjelasan metode; keputusan cakupan. Di sesi sebelumnya dibuat paket kode referensi (`fraud_gnn/`) yang **hanya diuji pada data sintetis** (centrality cocok dengan networkx; model `RelAttnGNN`, bukan DA-GAT). **DA-GAT belum diimplementasikan atau diuji.** User tidak memerlukan kode itu dan menulis sendiri.

**Belum:**
- Dataset Kaggle asli belum diunduh dan diperiksa.
- Semua eksperimen (E0–D1) belum dijalankan; hipotesis volume belum teruji.
- Apakah graf Yoo satu edge per klaim: belum pasti.
- Level metrik Yoo: belum pasti.
- Rumus GAT/HAN/HGT dan referensi PNA/Xu/GATv2/Graphormer ditulis dari pengetahuan umum, **belum dibaca ulang** dalam sesi ini.
- Link GitHub RDPGL dan Muhammad belum dicek aktif.
- Keunikan DA-GAT belum diperiksa lewat pencarian literatur.

---

## 8. Referensi

- Y. Yoo, J. Shin, S. Kyeong, "Medicare Fraud Detection Using Graph Analysis: A Comparative Study of Machine Learning and Graph Neural Networks," *IEEE Access* 11:88278–88294, 2023. doi:10.1109/ACCESS.2023.3305962
- J. Ma et al., "Fighting against Organized Fraudsters Using Risk Diffusion-based Parallel Graph Neural Network," *IJCAI-23*, pp. 6138–6145.
- R. Muhammad et al., "Fraud detection and explanation in medical claims using GNN architectures," *Sci. Rep.* 15:41734, 2025. doi:10.1038/s41598-025-22910-6
- A. du Preez et al., "Fraud detection in healthcare claims using machine learning: A systematic review," *Artif. Intell. Med.*, 2024. doi:10.1016/j.artmed.2024.103061
- J. Lu et al., "Health insurance fraud detection by using an attributed heterogeneous information network with a hierarchical attention mechanism," *BMC Med. Inform. Decis. Mak.* 23:62, 2023. doi:10.1186/s12911-023-02152-0
- B. Hong et al., "Health insurance fraud detection based on multi-channel heterogeneous graph structure learning," *Heliyon* 10, 2024. doi:10.1016/j.heliyon.2024.e30045
- E. Nabrawi, A. Alanazi, *Risks* 11(9):160, 2023. doi:10.3390/risks11090160
- R. A. Gupta, "Healthcare Provider Fraud Detection Analysis," Kaggle (dataset).
- *Dari pengetahuan umum, verifikasi sebelum dikutip:* P. Veličković et al., "Graph Attention Networks," ICLR 2018; S. Brody et al., "How Attentive are Graph Attention Networks?" (GATv2), ICLR 2022; K. Xu et al., "How Powerful are Graph Neural Networks?" ICLR 2019; G. Corso et al., "Principal Neighbourhood Aggregation for Graph Nets," NeurIPS 2020; C. Ying et al., "Do Transformers Really Perform Bad for Graph Representation?" NeurIPS 2021; X. Wang et al., HAN, WWW 2019; Z. Hu et al., HGT, WWW 2020; W. Hamilton et al., GraphSAGE, NeurIPS 2017.
