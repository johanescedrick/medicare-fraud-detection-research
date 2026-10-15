# HANDOFF: Riset Fraud Detection Klaim Kesehatan
**Backbone:** Yoo et al. (2023) · **Jenis studi:** diagnostik berbasis hipotesis · **Intervensi:** DA-GAT + tangga atribut klaim di edge (G0 → GE-1 → GE) + backbone GNN lain · **Waktu: 2 bulan**
Versi 4, 6 Oktober 2026, setelah konsultasi dosen pembimbing dan pencarian literatur tambahan. Menggantikan versi 3. Dokumen berdiri sendiri: Claude yang membacanya tidak punya akses ke percakapan sebelumnya.

**Judul kerja (dipilih user):** *Are Graphs Worth the Trouble? A Diagnostic Study of Graph Neural Networks for Healthcare Provider Fraud Detection*

---

## 0. Untuk Claude yang melanjutkan (baca dulu)

**Peran.** Pendamping riset. User menulis kodenya sendiri dan butuh penjelasan metode yang jelas, cepat, dan tepat. Jangan membuat notebook atau paket kode kecuali diminta.

**Gaya.** Bahasa Indonesia, istilah teknis Inggris boleh. Jelaskan dari dasar dengan contoh kecil. User memakai preferensi "advisor, bukan asisten": sebut kelemahan di kalimat pertama, jangan membuka dengan pujian.

**Aturan kerja:**
- Tandai mana yang **terverifikasi** (dibaca full-text) dan mana yang **belum**. Jangan mengarang rumus atau angka. Kalau perlu detail paper, minta user mengunggah PDF-nya.
- **Framing studi = diagnostik, bukan komparatif.** Yoo sudah studi komparatif ("mana yang lebih baik"). Studi ini bertanya "kenapa GNN kalah" dan memakai model baru sebagai **intervensi** untuk menguji tiap penjelasan. Jangan kembali ke framing "kami mengusulkan model yang lebih baik".
- Jangan menjual "memakai GAT" sebagai novelty (Yoo sudah memakai GAT).
- Jangan menjual DA-GAT, atribut klaim di edge, atau "GNN lain" sebagai kebaruan arsitektur: semuanya punya preseden (lihat 2.5.3). Kebaruan = **diagnosis** atas klaim Yoo.
- Arahan dosen (1.2) sejalan dengan desain, tetapi kata "atasi" tidak boleh mengubah studi menjadi adu banyak GNN atau paper "model yang lebih baik". Pegang framing diagnostik.
- Kalau ditanya "apa yang baru", mulai dari **1.1**, jawab langsung, dan jangan menyodorkan opsi baru. User sempat bingung karena penjelasan berputar-putar.
- Kode eksperimen HAN adalah **"HAN"**, bukan "H0". **H0, H1, H2 adalah nama hipotesis.**

**Kesalahan yang pernah terjadi di sesi ini (jangan diulang):**
1. Menulis "~9% fraud di level klaim". Salah: 9% = level **provider**; level klaim ≈ 38%.
2. Merekomendasikan RDPGL sebagai backbone, padahal label *organized fraud*-nya dianotasi pakar NHSA dan tidak ada di Kaggle publik.
3. Menyebut detail MHAMFD (Lu et al. 2023) padahal hanya abstrak yang terbaca.
4. Menyebut Muhammad et al. "kontradiktif" dengan Yoo. Berlebihan: baseline ML mereka tanpa centrality, selisihnya ~1 poin.
5. Menyebut edge `CHARGE` sebagai "satu edge per pasangan unik". Tabel 3 Yoo menunjukkan jumlah edge `CHARGE` **sama dengan jumlah klaim** (330.288 / 122.543 / 103.872), jadi kemungkinan **satu edge per klaim**. Belum pasti.
6. Mengusulkan "attention GNN" sebagai novelty, padahal Yoo sudah menguji GAT.
7. Desain v2.1 memasukkan degree ke X_cent, sehingga "M1 ≈ M2" nyaris tautologis. Diperbaiki dengan M1b dan C1 (lihat 4.9).
8. Menyebut GAT sadar-degree dan atribut klaim di edge belum punya preseden. Ada: GRIT (attention tidak peka degree + scaler log(1+d) berbobot-belajar), PNA, MBGNN (fitur ber-skala degree, fraud kesehatan), RDPGL (embedding edge-klaim pada Kaggle yang sama), Egressy dkk./GRANDE/DIAM (edge features pada multigraf transaksi). Lihat 2.5.3.
9. Memperlakukan RDPGL sebagai "pendukung" saja. RDPGL memakai **dataset Kaggle yang sama** dan embedding edge-klaim; bedanya: tugas klasifikasi klaim, label organized fraud (tidak publik), split temporal.
10. Menyebut "pencarian literatur belum dilakukan". Sudah dilakukan (2.5) tetapi terbatas.
11. Menyatakan judul lama "Why Do GNNs Underperform..." tetap benar apa pun hasilnya. Salah: judul itu mengasumsikan GNN kalah. Diganti (5.1).

---

## 1. Konteks dan keputusan terkunci

- **Tujuan:** paper tentang deteksi fraud klaim kesehatan dengan GNN. Asal minat: hackathon BPJS Kesehatan 2021. Tugas akademik: minimal 3 paper relevan.
- **Waktu:** **2 bulan**. **Graph Attention Network wajib ada.**
- **Tujuan utama studi (dikonfirmasi user):** mengungkap **kenapa GNN kalah dari ML+centrality** pada studi Yoo.

| Keputusan | Alasan |
|---|---|
| Dataset = **Kaggle** (bukan BPJS) | Punya ID relasional eksplisit; data BPJS tabular flat tanpa ID pasien/dokter/faskes |
| Backbone = **Yoo et al. 2023** | Hasilnya berlawanan intuisi (GNN kalah), dataset publik, dan penulis sendiri menulis dugaan penyebab serta keterbatasan yang tidak diuji. User memilih tetap Yoo; IEEE Access tercatat Q1 menurut agregator pihak ketiga (**bukan sumber resmi**, verifikasi di scimagojr.com) |
| Jenis studi = **diagnostik berbasis hipotesis** | Pertanyaan "kenapa" lebih kuat dan lebih baru daripada mengulang perbandingan |
| Penjelasan yang diuji = **H0, H1, H2** (lihat 4.6) | Masing-masing terhubung ke teks Yoo |
| Intervensi = **DA-GAT** (H1) dan **tangga atribut klaim** G0 → GE-1 → GE (H2), desain **2 × 2** | Tiap faktor menguji satu penjelasan; tangga memberi pola dose-response (makin banyak informasi klaim, selisih menyempit?) |
| Backbone GNN lain (arahan dosen) = **GINEConv, PNAConv, TransformerConv, GraphSAGE** di samping GATv2 | Uji H1 "alami" dan uji ketahanan lintas arsitektur, **bukan** ajang adu model (lihat 4.7b) |
| Graf = **2 edge milik Yoo** (`CHARGE`, `PROJECT_PROVIDER`); `CHARGE` kini membawa atribut klaim | Perbandingan tetap apple-to-apple |
| Level metrik utama = **provider** (+ klaim-terbobot sebagai pembanding) | Label melekat di provider |
| Alternatif backbone yang ditolak: Argade & Gandeeban 2026 (ISSF 2026, atlantis-press.com/article/126025139.pdf) | Lebih sederhana tetapi prosiding dengan satu split. Tetap berguna sebagai **referensi implementasi level provider** (fitur agregat, split, metrik) dan bukti pendukung bahwa fitur volume saja memberi ROC-AUC > 0,95. Kode: github.com/dipakml/healthcare-provider-fraud-benchmark, **belum dicek aktif** |

**Dikeluarkan dari cakupan** (future work): G4 (fusi GAT + centrality); edge `DIAGNOSIS_SIM` dan `GEO`; `RelAttnGNN`; RE-GraphSAGE; explainability penuh dan validasi pakar; RDPGL penuh; studi kasus BPJS.

### 1.1 Ringkasan inti (mulai dari sini)

| | **Yoo et al. (sudah ada)** | **Studi ini (baru)** |
|---|---|---|
| Jenis studi | Komparatif: mana yang lebih baik? | **Diagnostik: kenapa GNN kalah?** |
| Data, graf | Kaggle Medicare, 2 jenis edge | Sama; edge klaim membawa atribut klaim |
| Hasil | ML+centrality menang, GNN kalah | Diuji ulang dulu (premis), lalu dicari sebabnya |
| Penyebab GNN kalah | Hanya dugaan ("information loss", Bab V-B) | Tiga penjelasan diuji: **H0** artefak evaluasi, **H1** volume hilang, **H2** atribut klaim hilang |
| Model GNN | GraphSAGE, GAT, HAN, HGT biasa | GATv2 + dua **intervensi** (DA-GAT; tangga atribut klaim G0 → GE-1 → GE) dalam 2 × 2, lalu backbone lain (GINEConv, PNAConv, TransformerConv, GraphSAGE) sebagai uji ketahanan |
| Kontrol | Tidak ada | Kontrol volume (M2), centrality tanpa degree (M1b), korelasi (C1), probe (D1) |
| Evaluasi | Level dan ambang tidak jelas | Level provider, ambang dari validasi, 5 split × 10 seed |

**Intuisi H1.** GAT itu seperti rata-rata nilai kelas: rata-rata tidak memberi tahu jumlah muridnya. Provider dengan 10 klaim dan 1.000 klaim bisa terlihat sama bagi GAT, padahal volume besar sering menandakan fraud.

**Intuisi H2.** Tiap klaim punya nilai, diagnosis, dan lama rawat. Kalau semua itu diringkas ke node, GAT hanya melihat rangkuman, bukan klaim-klaim yang janggal.

**Desain 2 × 2 (inti eksperimen GNN):**

| | Tanpa atribut klaim di edge | Dengan atribut klaim di edge |
|---|---|---|
| **Agregasi biasa** | G0: GAT (replikasi Yoo) | GE: GAT + atribut klaim (uji H2) |
| **Sadar degree** | G3: DA-GAT (uji H1) | **GE3: DA-GAT + atribut klaim (H1 + H2)** |

**Yang baru bukan data, graf, atau GAT-nya, melainkan:**
1. Pengujian terkontrol atas penjelasan yang ditulis Yoo sendiri tetapi tidak diuji.
2. Pemisahan kontribusi volume vs struktur graf vs atribut klaim.
3. Protokol evaluasi yang adil (H0).

**Catatan jujur.** Semua komponen punya preseden (2.5.3): scaler degree (PNA, GRIT, MBGNN), edge features pada GAT (PyG `edge_dim`, RDPGL pada dataset yang sama, GNN multigraf untuk AML), dan temuan bahwa model berbasis pohon/fitur graf setara atau mengungguli GNN (G-GBM, Deprez, StGNN, Weber). Kebaruan ada pada **diagnosis**: menguji penjelasan atas klaim Yoo secara terkontrol pada dataset ini. Tulis "menguji penjelasan", bukan "membuktikan penyebab".

### 1.2 Arahan dosen pembimbing dan keselarasannya dengan desain

Dosen menyarankan dua hal. Keduanya sejalan dengan desain, dengan satu penyesuaian.

| Arahan dosen | Pemetaan ke desain | Penyesuaian |
|---|---|---|
| 1. Atasi *information loss* yang diklaim Yoo. Langkah pertama: embedding atribut klaim atau pembobotan edge | H2, dijalankan sebagai **tangga informasi**: G0 (tanpa atribut) → **GE-1** (satu atribut, nilai klaim, setara bobot edge) → **GE** (semua atribut) | "Atasi" dibaca sebagai "uji apakah menambahkan informasi yang hilang menutup selisih". Hasil negatif tetap kontribusi |
| 2. Cari algoritma GNN lain (GAT, DA-GAT, dst.) | Backbone lain dipakai untuk uji H1 alami dan uji ketahanan (4.7b) | Bukan "adu banyak GNN": Yoo sudah studi komparatif, dan 2 bulan tidak cukup |

**Selaraskan dengan dosen di minggu 1:**
- Target paper = paper metode ("model yang mengatasi information loss") atau paper analisis ("menguji penjelasan")? Ini menentukan apakah hasil negatif diterima dan jurnal mana yang cocok.
- Kriteria sukses: pola tangga atribut dan uji H1 alami dianggap bukti, walau tidak ada model yang mengungguli ML+centrality.

**Skema ringkas eksperimen GNN:**

```
Input node: X_mean (tanpa volume)        Graf: CHARGE (1 edge/klaim), PROJECT_PROVIDER
        │
        ├─ Sumbu atribut klaim (H2):   G0 ──► GE-1 ──► GE          (tidak ada → 1 atribut → semua)
        ├─ Sumbu degree (H1):          GATv2 ──► DA-GAT             (tanpa → dengan faktor degree)
        └─ Sumbu arsitektur (H1+RQ4):  SAGE-mean, GATv2, TransformerConv │ GINEConv, PNAConv
                                       (tidak peka degree)               (peka degree bawaan)
Inti 2 × 2:  G0 | GE  /  G3 | GE3      Probe: D1      Dinilai: 2 protokol (H0), 5 split
```

---

## 2. Paper

Empat paper dibaca full-text: tiga di bawah (2.1–2.3) dan Argade & Gandeeban (lihat tabel keputusan di bagian 1). Sisanya hanya abstrak/cuplikan.

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

**Pernyataan penulis yang dijadikan pertanyaan riset (dibaca dari full-text):**

| Pernyataan Yoo (parafrase) | Lokasi | Diuji sebagai |
|---|---|---|
| GNN kalah karena konversi tabular→graf menyebabkan kehilangan informasi | Bab V-B | H1 (volume) dan H2 (atribut klaim) |
| Representasi node dibuat dengan menjumlahkan embedding antar-relasi; metode lain mungkin lebih baik | Bab V-D, keterbatasan 2 | H2: atribut klaim disimpan di edge |
| Data hampir seimbang (6:4); class imbalance dijadikan kerja lanjutan | Bab V-D, keterbatasan 1 | Evaluasi level provider (fraud ≈ 9%) |
| Eksplorasi metode node embedding lain untuk memperbaiki GNN | Bab VI | H1: agregasi sadar degree |
| Ambang keputusan dan level metrik tidak dijelaskan | (tidak ada) | H0: artefak evaluasi |

### 2.2 Pendukung: Ma et al. (2023), RDPGL · *terverifikasi*

*Fighting against Organized Fraudsters Using Risk Diffusion-based Parallel Graph Neural Network.* IJCAI-23, hlm. 6138–6145. Kode (dari teks paper, **tidak dicek aktif**): github.com/finint/antifraud.

Memakai Kaggle yang sama (38,1% fraud). Dua komponen paralel: GAT heterogen lokal pada graf beneficiary–provider (attention Eq. 1: $\alpha_{ij}=\mathrm{softmax}_j(\sigma(u^\top[W_bh_{b,i}\Vert W_ph_{p,j}]))$, pesan dari diri/edge/tetangga, embedding edge-klaim) dan risk diffusion global pada graf klaim–klaim (random walk + personalized PageRank + pre-training GCN). Label *organized fraud* dianotasi pakar NHSA, **tidak publik**. Hasil contoh (organized, outpatient): AUC 0,8713, recall 0,8302, F1 0,7982. Ablation: kedua komponen dibutuhkan.

**Peran untuk kita:** bukti bahwa attention pada graf provider–beneficiary dipakai di literatur terbaru. **Penting untuk novelty:** RDPGL sudah memakai embedding edge-klaim (Eq. 2 dan 4) pada dataset Kaggle yang sama, sehingga intervensi H2 (atribut klaim di edge) **bukan** kebaruan arsitektur; sitasi RDPGL di bagian metode. Komponen risk diffusion dan label organized fraud **tidak** dipakai.

**RDPGL vs risetmu (dataset sama, tugas berbeda):**

| | RDPGL | Risetmu |
|---|---|---|
| Dataset | Kaggle (Gupta 2020), > 0,55 juta klaim, 38,1% fraud | Sama |
| Unit dan tugas | klasifikasi edge/klaim (individual dan organized) | klasifikasi node provider |
| Label | label individual + label organized yang dianotasi pakar NHSA (tidak ada di Kaggle publik) | `PotentialFraud` per provider |
| Split | temporal (Jan–Sep, lalu Okt–Des) | provider, stratified |
| Atribut klaim di edge | ya (embedding edge-klaim) | ya (GE-1, GE, GE3) |

Angka kedua studi **tidak bisa dibandingkan langsung**.

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

### 2.5 Posisi terhadap literatur (pencarian tambahan, 6 Oktober 2026)

Legenda: ● = teks penuh dibaca; ○ = hanya abstrak/cuplikan. Kolom gap: "penulis" = ditulis penulis paper; "saya" = inferensi saya.

#### 2.5.1 Lima belas paper tambahan

**A. Memakai dataset Kaggle yang sama**

| # | Paper | Yang dilakukan | Gap / catatan |
|---|---|---|---|
| 1 ● | Vandervorst, Deprez, Verbeke, Verdonck (2026), *Inductive inference of gradient-boosted decision trees on graphs for insurance fraud detection*, Data Min. Knowl. Discov. 40(5):78; arXiv 2510.05676; doi 10.1007/s10618-026-01249-0. Kode (dari paper, belum dicek): github.com/VerbekeLab/GBDT_Graphs | G-GBM: GBDT memakai fitur sepanjang metapath. Pada Kaggle provider (2 graf: 2.600 dan 2.781 provider; fraud 11% dan 8%): empat fitur provider (jumlah klaim, rata-rata dan simpangan baku nominal, jumlah pasien) + LightGBM sudah AUC-ROC > 0,90; fitur tetangga tidak menambah berarti. Split: dua graf terpisah (klaster Louvain) | Penulis: attention/SHAP per metapath; dimensi fitur membengkak pada graf padat; ketergantungan train/test pada graf yang tumpang tindih sering diabaikan. Saya: tidak menjelaskan kenapa GNN lemah dan tidak membandingkan dengan centrality Yoo. **Versi arXiv v1 menyebut HINSage gagal; versi lebih baru melaporkan HinSAGE/HAN "moderat" pada AUC-PR: cek versi yang dikutip** |
| 2 ○ | Deprez, Vandervorst, Verbeke, Verdonck, Baesens (2024), *Network analytics for insurance fraud detection: a critical case study*, Eur. Actuar. J. 14:965–990; doi 10.1007/s13385-024-00384-6 | BiRank, metapath2vec, GraphSAGE, fitur jaringan sederhana pada data asuransi mobil **dan** Kaggle provider. Kesimpulan: metode kompleks belum tentu mengungguli fitur jaringan dasar | Saya: attention (GAT/HAN) tidak ada di daftar; penyebab tidak dibahas. Teks penuh diblokir |
| 3 ● | Argade & Gandeeban (2026), *Benchmarking TabPFN Against Traditional and Relational Models for Provider Fraud Detection*, ISSF 2026 | LR, LightGBM, CatBoost, TabPFN, GraphSAGE pada fitur agregat provider | Penulis: graf multi-entitas lebih kaya, hibrida tabular-graf. Saya: satu split tanpa seed, edge biner |
| 4 ● | Farahmandazad & Danesh (2025), arXiv 2502.15898 | Lima model ML + SMOTE pada 556.703 record; Random Forest: akurasi validasi 98,8%, F1 98,4% | Penulis: XAI, hibrida graf + DL, real-time, unsupervised. Saya: split tidak dijelaskan di level provider sementara fitur diagregasi per provider, **kemungkinan kebocoran (belum pasti)**. Mengutip Yoo sebagai "GNN mengungguli ML tradisional", padahal temuan utama Yoo 2023 sebaliknya |
| 5 ○ | Rakesh & Shetty (2024), *Enhanced Medicare Fraud Detection Using GCN*, AISP; doi 10.1109/AISP61711.2024.10870744 | Graf provider–pasien (power edge) dan provider–provider (shared-physician); GCN dilaporkan lebih baik dari LR dasar | Saya: pembandingnya hanya LR dasar tanpa centrality (yang justru mengalahkan GNN di Yoo) |
| 6 ○ | Almeida (2024), tesis USP, *Detecção de Fraudes no Setor de Saúde: Métodos de IA* (Portugis; URL penuh tidak tersedia) | ML (LR, RF, XGBoost, LightGBM, MLP) vs GNN (GraphSAGE, GAT, HAN, HGT) pada Medicare; menguji apakah fitur turunan graf menambah performa | Hasil **belum dibaca**. Tampak mereplikasi Yoo: paling mungkin tumpang tindih dengan risetmu |

**B. GNN untuk asuransi kesehatan (dataset lain)**

| # | Paper | Yang dilakukan | Gap / catatan |
|---|---|---|---|
| 7 ● | Chen, Lu, Yang, Chen, Lin (2022), StGNN, J. Netw. Intell. 7(2):480–498 | Metapath + attention dua tingkat + Bi-LSTM; data biro asuransi kota (Tiongkok), 440 dan 10.647 pasien; Medical-2 rasio fraud 1:70. **LR dan RF dengan fitur buatan (termasuk jumlah kunjungan dan total biaya) setara HAN (F1 0,77–0,84 vs 0,75–0,80) dan jauh di atas GCN (0,44–0,58)**; penulis: GCN memperlakukan semua tetangga sama sehingga terkena noise | Penulis: saring tetangga bermirip rendah; subgraf per siklus pengobatan; graf dinamis. Data tidak publik |
| 8 ● | Lu, Lin, Chen, Chen (2023), MBGNN, J. Netw. Intell. 8(3):895–912 | Tiga strategi sampling + attention dua langkah + modul topologi. Pembanding hanya model graf (GCN, GAT, HAN, StGNN, Metapath2Vec), **tanpa ML**. **Fitur node dikalikan fungsi degree (λ^(d/n), Eq. 2)** agar node padat mendapat perhatian lebih | Penulis: modul topologi memakai GNN paling dasar; waktu hanya sebagai entitas. Saya: preseden langsung penskalaan degree pada fraud kesehatan |
| 9 ○ | Lu dkk. (2023), MHAMFD, BMC MIDM 23:62 | Attention multi-level pada perilaku kunjungan | Bagian limitation tidak terbaca (PMC diblokir) |
| 10 ○ | Hong dkk. (2024), MHGSL, Heliyon 10 | GNN heterogen + GCN, multi-channel dan fusi fitur | Limitation tidak terverifikasi; Muhammad dkk. mencatat studi sejenis kurang penjelasan |
| 11 ○ | Zhang dkk. (2024), POCL, AAAI-24; doi 10.1609/AAAI.V38I20.30259; kode github.com/finint/POCL | Pre-training kontrastif + pembaruan online untuk pola fraud yang berevolusi | Fokus non-stasioner, di luar cakupan |
| 12 ○ | ConFVG, ICLR 2026 (poster) | Autoencoder graf berpemandu Fiedler vector + subgraph attention fusion; label langka dan non-stasioner | Dataset spesifik tidak disebut di abstrak |

**C. Latar: kelemahan GNN untuk fraud dan data tabular**

| # | Paper | Yang dilakukan | Gap / catatan |
|---|---|---|---|
| 13 ● | Pereira & Murai (2021), arXiv 2105.14568 | Mengevaluasi deteksi fraud jaringan; representasi node dipengaruhi tetangga dan hub sehingga memperburuk ketimpangan kelas; mencatat Weber dkk.: RF mengungguli GNN dengan fitur yang sama (Elliptic) | Penulis: imbalance + concept drift belum terpecahkan. Bukan data kesehatan |
| 14 ○ | Grinsztajn dkk. (2022), NeurIPS Datasets & Benchmarks; arXiv 2207.08815 | Pohon tetap terbaik pada data tabular menengah (~10 ribu sampel). Tiga tantangan NN: tahan fitur tak informatif, pertahankan orientasi data, belajar fungsi tak beraturan | Saya: ukuran data kita (≈5.400 provider) sebanding; dasar H3 opsional |
| 15 ○ | du Preez dkk. (2024), Artif. Intell. Med.; doi 10.1016/j.artmed.2024.103061 | SLR 137 studi (abstrak dari user) | Data tak konsisten/tak terstandar, label terbatas; saran: transparansi persiapan data dan dataset benchmark → mendukung kontribusi protokol evaluasi |

#### 2.5.2 Hasil yang dilaporkan pada dataset Kaggle yang sama (bahan H0)

| Paper | Level dan split | Model terbaik | Hasil |
|---|---|---|---|
| Yoo (2023) | split provider 6:2:2; level metrik tidak eksplisit | LR + centrality | F1 0,65; AUROC 0,74 |
| Farahmandazad (2025) | split tidak dijelaskan | Random Forest | akurasi 98,8%; F1 98,4% |
| Argade (2026) | provider, 80/20 stratified | LR / GraphSAGE | ROC-AUC 0,953 / 0,954; PR-AUC 0,742 / 0,740 |
| Vandervorst (2026) | provider, dua graf terpisah (Louvain) | LightGBM, G-GBM | AUC-ROC > 0,90 |

Metrik berbeda dan **tidak sebanding**; justru itu poinnya: selisih sebesar ini kemungkinan besar berasal dari level dan cara split, bukan modelnya. Kandidat **Tabel 1** paper.

#### 2.5.3 Preseden tiap komponen (tidak ada yang baru)

| Komponen | Preseden | Domain | Status |
|---|---|---|---|
| Atribut/fitur di edge pada GAT | PyG `GATv2Conv(edge_dim=...)`; **RDPGL** (Kaggle sama); PU GNN (arXiv 2211.08604; GATv2 dengan fitur link, chargeback fraud); Egressy dkk., AAAI-24 (GNN multigraf berarah; F1 naik hingga 30% pada AML; doi 10.1609/aaai.v38i10.29069); GRANDE (arXiv 2302.02101); DIAM (Edge2Seq) | umum, keuangan, kesehatan | Bukan kebaruan |
| Attention sadar degree | **GRIT** (arXiv 2305.17589, *Graph Inductive Biases in Transformers without Message Passing*: attention bawaan tidak peka degree; scaler log(1+d) berbobot-belajar); **PNA** (arXiv 2004.05718); **MBGNN** (Eq. 2); LSGAT (koefisien attention untuk node berderajat besar); DEMO-Net (disebut sekunder) | umum, kesehatan (data lain) | Bukan kebaruan |
| GNN lemah pada fitur tabular | BGNN (arXiv 2101.08543, ICLR 2021); Grinsztajn; Weber 2019 (arXiv 1908.02591: RF unggul atas GCN pada Elliptic); G-GBM | umum, keuangan, asuransi | Dasar H3 |
| Fitur graf + ML ≥ GNN | Yoo; Deprez; StGNN (LR/RF ≈ HAN); Weber | kesehatan, keuangan | Dasar premis |
| Konversi tabular → graf | survei GNN4TDL (arXiv 2401.02143); RF-GNN (arXiv 2602.24224) | umum | Bukan masalah kita: Kaggle punya relasi eksplisit |

#### 2.5.4 Yang tidak ditemukan dan kalimat novelty

- **Tidak ditemukan:** studi yang menguji penjelasan atas klaim Yoo dengan eksperimen terkontrol. Yoo disitir sebagai latar belakang dan beberapa kali keliru (Farahmandazad; versi di *Risks* 2025, doi 10.3390/risks13100198, memuat uraian serupa). Waspadai sitasi sekunder.
- **Kalimat aman:** "Penelitian terdahulu mengusulkan perbaikan terhadap keterbatasan agregasi GNN (PNA, GRIT), pemanfaatan fitur edge (RDPGL, GNN multigraf), dan model hibrida GBDT–graf (G-GBM, BGNN), tetapi sejauh yang kami ketahui belum ada yang menguji penyebab kekalahan GNN pada temuan Yoo et al. (2023)."
- **Kalimat tidak aman:** "Kami pertama yang mengatasi information loss"; "volume cukup" sebagai temuan baru (G-GBM sudah menunjukkannya pada dataset yang sama); DA-GAT atau GE sebagai arsitektur baru.

#### 2.5.5 Batas pencarian dan kata kunci

- Google Scholar tidak bisa diakses; beberapa teks penuh diblokir (Deprez, MHAMFD, MHGSL); beberapa paper hanya abstrak. Tidak terbaca sama sekali: Xiao dkk. 2023 (*Front. Comput. Sci.*, hanya referensi), Liu dkk. 2025 (*Data Anal. Knowl. Discov.*, abstrak tidak tampil). Hasil Almeida belum dibuka.
- **Kueri yang sudah dijalankan:** kehilangan informasi saat konversi tabular ke graf; GNN multigraf dengan edge features; GBDT + GNN pada fitur tabular; degree-scaler dan attention sadar degree; GAT dengan edge features untuk fraud; kutipan dan penjelasan atas Yoo.
- **Jalankan sendiri di Scholar sebelum minggu 1:** "why graph neural networks underperform fraud detection tabular"; "healthcare provider fraud GNN volume features"; "edge attributes claims graph health insurance fraud"; "degree-aware graph attention" + "edge features" + "provider fraud".


---

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

**Ukuran (dari ingatan, verifikasi setelah unduh):** ≈ 5.410 provider, ≈ 138.556 pasien, ≈ 40k klaim inpatient, ≈ 517k outpatient. Fraud rate **provider ≈ 9–10%**, **klaim ≈ 38%**. Pemeriksaan silang dari paper lain: G-GBM (Vandervorst dkk.) melaporkan 278 dan 228 provider fraud pada dua graf berisi 2.600 dan 2.781 provider (≈ 9,4%), konsisten dengan angka ini; angka klaim 38% konsisten dengan Yoo dan Farahmandazad (0,381). Tetap verifikasi di data asli.

**Prosedur (Yoo Fig. 1):** union inpatient+outpatient (flag `is_ip`) → dedup `ClaimID` → merge `BeneID` → merge `Provider`. Split **per provider**, stratified, seed tetap.

**Catatan:** label di level provider, klaim mewarisinya. Asal-usul data tidak jelas (Yoo juga mencatat), jadi generalisasi tidak bisa diklaim.

### 3.2 BPJS Hackathon 2021 (di luar cakupan 2 bulan)

`train.csv` (pemisah `;`): 160.174 baris × 53 kolom, label 50/50, satu baris = satu klaim. Fitur administratif (`kdkc` 126 nilai, `dati2` 486, `typeppk`, `jkpst`, `umur`, `los`) dan klinis (`diagprimer`, `dx2_*`, `proc*_*`). **Tidak ada ID pasien/dokter/faskes**, sehingga graf harus dikonstruksi buatan. Jadikan future work.

---

---

## 4. Metode

### 4.1 Urutan kerja

1. Gabung data, split provider, cetak ringkasan (cocokkan dengan Tabel 3 Yoo).
2. Susun fitur (4.2), graf dengan atribut klaim di edge (4.3), centrality (4.4).
3. **Replikasi premis (E0):** apakah GNN memang kalah?
4. Baseline ML termasuk kontrol volume (M0–M3, M1b) dan korelasi C1.
5. Baseline GNN (G0, HAN opsional), lalu uji H0 dengan dua protokol evaluasi.
6. Intervensi 2 × 2 dan tangga atribut (G1, G3, GE-1, GE, GE3), probe D1, lalu backbone lain (4.7b).
7. Robustness, tabel, penulisan.

### 4.2 Definisi fitur: tiga kelompok (inti desain eksperimen)

Hipotesis kita adalah bahwa GAT tidak menangkap informasi **volume**. Supaya hipotesis ini bisa diuji, fitur input GNN **tidak boleh** sudah memuat volume. Jadi pisahkan:

| Kelompok | Isi (usulan; boleh disesuaikan) |
|---|---|
| **X_mean** (intensif, tanpa volume) | rata-rata/std per klaim: reimbursement, deductible, durasi klaim, LOS, jumlah diagnosis/prosedur/physician; usia rata-rata; % pria, meninggal, renal, rawat inap; rata-rata 11 kondisi kronis, 4 amount, 2 months-coverage |
| **X_vol** (volume) | `n_claims`, `n_ip`, `n_bene`, `n_physicians`, `reimb_sum`, `reimb_max`, `claims_per_bene`, `reimb_per_bene`, `state_nunique`, `county_nunique` |
| **X_cent** | 8 kolom: {degree, eigenvector, closeness, PageRank} × {graf provider–beneficiary, graf provider–physician} |

Catatan: karena Yoo tidak jelas soal edge, hitung degree dalam dua versi: jumlah **pasien unik** dan jumlah **klaim**.

**Atribut klaim di edge `CHARGE` (untuk H2):** satu vektor per klaim, dinormalisasi (fit di train): `InscClaimAmtReimbursed`, `DeductibleAmtPaid`, durasi klaim (hari), lama rawat (hari, 0 untuk rawat jalan), flag rawat inap, jumlah kode diagnosis, jumlah kode prosedur, jumlah physician tercatat. Kode diagnosis mentah **tidak** dimasukkan agar tetap sederhana (bisa jadi future work).

### 4.3 Graf (struktur milik Yoo; `CHARGE` membawa atribut klaim)

Node `provider` (target) dan `beneficiary`. Edge:
- `CHARGE`: provider–beneficiary. **Default: satu edge per klaim** (sesuai Tabel 3 Yoo). Setiap edge membawa atribut klaimnya (4.2) untuk intervensi H2. Variasi robustness: satu edge per pasangan unik (atribut dirata-rata).
- `PROJECT_PROVIDER`: dua provider berbagi ≥ 1 physician (proyeksi bipartit). Hitung dari tiga kolom physician dan dari `AttendingPhysician` saja; pilih yang jumlah edge-nya paling dekat Yoo (3.486 pada graf train) dan laporkan. Bandingkan setelah graf dibatasi ke provider train saja, karena Yoo membangun graf terpisah per split. Beri batas jumlah provider per physician agar tidak terbentuk clique raksasa.

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
- **GNN:** **GAT** (wajib) dan **HAN** (GNN terbaik Yoo) sebagai referensi (kode eksperimen **"HAN"**, bukan "H0", karena H0 adalah nama hipotesis). GraphSAGE masuk sebagai backbone replikasi (lihat 4.7b); HGT opsional bila waktu ada. PyG: `HeteroConv` + `GATv2Conv`/`GATConv`, `HANConv`, `HGTConv`, `SAGEConv`. Untuk GAT pada graf heterogen, pakai satu GAT per tipe edge lalu jumlahkan (pilihan desain; catat sebagai asumsi karena Yoo tidak merincinya).
- Kerangka message passing (Yoo Eq. 1): $H_u^{(l+1)}=\mathrm{Up}^{(l)}\big(H_u^{(l)},\mathrm{Agg}^{(l)}(\{H_v^{(l)}:v\in N(u)\})\big)$.
- Rumus GAT/HAN/HGT diambil dari paper aslinya, **bukan dari Yoo**.

### 4.6 Hipotesis dan prediksi

| Hipotesis | Isi | Prediksi bila benar | Eksperimen |
|---|---|---|---|
| **H0** artefak evaluasi | Selisih GNN–ML di Yoo sebagian berasal dari cara evaluasi (level metrik, ambang keputusan) | Selisih mengecil saat semua model dinilai dengan protokol adil | Semua model dinilai dua kali (4.10) |
| **H1** volume hilang | Attention merata-ratakan tetangga sehingga jumlah klaim/pasien tidak terbawa | M2 ≈ M1; C1 tinggi; D1 rendah untuk G0; G3 (atau G1) menutup selisih | M0–M3, M1b, C1, D1, G0 vs G1 vs G3 |
| **H2** atribut klaim hilang | Nilai, diagnosis, lama rawat diringkas ke node | Selisih menyempit bertahap G0 → GE-1 → GE (dose-response) | G0 vs GE-1 vs GE |
| H1 + H2 | Keduanya diperlukan | GE3 > G3 dan GE3 > GE | 2 × 2 |
| **H3** (opsional) bias induktif | GNN kurang cocok dengan fitur tabular heterogen (BGNN, Grinsztajn) | "Embedding GNN → XGBoost" atau GBDT pada fitur graf menutup selisih | satu eksperimen tambahan (H3x, 4.9) |

**Premis harus terbukti dulu.** Jika di replikasi GNN tidak kalah (gate minggu 3), pertanyaan "kenapa" hilang; laporkan bahwa kekalahan GNN di Yoo tidak bertahan dengan protokol yang adil.

### 4.7 Intervensi

#### Intervensi 1 (H1): DA-GAT (Degree-Aware GAT)

**Motivasi (hipotesis, belum terbukti).** Output GAT adalah rata-rata berbobot (softmax) dari tetangga, jadi besarnya agregasi tidak tergantung jumlah tetangga. Informasi "berapa banyak tetangga" (degree/volume) justru yang ditangkap centrality dan diduga mendorong keunggulan ML+centrality di Yoo. Argumen serupa ada di literatur: Xu et al. 2019 (aggregator rata-rata tidak bisa "menghitung"); Corso et al. 2020 (PNA: degree-scaler); dan **GRIT** (arXiv 2305.17589), yang menyatakan attention secara bawaan tidak peka terhadap degree dan menambahkan scaler log(1 + d) dengan bobot yang dipelajari. Bentuk faktor DA-GAT hampir sama dengan GRIT; bedanya hanya eksponen per relasi, jadi tulis sebagai **adaptasi** GRIT/PNA, bukan kontribusi arsitektur.

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

#### Intervensi 2 (H2): tangga atribut klaim di edge

**Arahan dosen:** langkah pertama = embedding atribut klaim atau pembobotan edge. Dijalankan sebagai tangga informasi:

| Tingkat | Model | Isi edge `CHARGE` |
|---|---|---|
| 0 | G0 | tidak ada |
| 1 | **GE-1** | satu atribut: `InscClaimAmtReimbursed` (nilai klaim); setara bobot edge, `edge_dim=1` |
| 2 | **GE** | semua atribut klaim (vektor, lihat 4.2) |

Pola tangga adalah bukti diagnostik H2: jika selisih terhadap ML+centrality menyempit seiring bertambahnya informasi klaim, penjelasan H2 makin kuat; jika tidak berubah, H2 melemah.

Skor attention ikut memakai vektor atribut klaim $x_{ij}$ pada edge $j \to i$:
$$e_{ij}=a^\top\,\mathrm{LeakyReLU}\big(W[h_i\Vert h_j\Vert W_e x_{ij}]\big),\qquad \alpha_{ij}=\mathrm{softmax}_{j}(e_{ij})$$
- Di PyG cukup dengan parameter `edge_dim` pada `GATv2Conv`, lalu kirim `edge_attr` untuk relasi `CHARGE` (dan arah baliknya). `PROJECT_PROVIDER` tidak punya atribut klaim (beri vektor nol atau pakai konv tanpa `edge_dim`).
- Karena `CHARGE` satu edge per klaim, setiap klaim membawa atributnya sendiri, tidak diringkas ke node.
- **Novelty:** bukan arsitektur baru. Preseden: RDPGL (embedding edge-klaim, Kaggle yang sama), PyG `GATv2Conv(edge_dim)`, PU GNN (GATv2 dengan fitur link, fraud chargeback), Egressy dkk./GRANDE/DIAM (edge features pada multigraf transaksi). Perannya di sini sebagai alat uji H2; pola tangga (dose-response) adalah bukti diagnostiknya.

**Model gabungan GE3** = DA-GAT + atribut klaim: pesan relasi `CHARGE` dihitung dengan attention ber-atribut, lalu dikali $s_r(d)$.

### 4.7b Backbone GNN lain (arahan dosen)

Arahan dosen: cari algoritma GNN lain. Di studi diagnostik ini, backbone lain dipakai untuk dua hal: (a) **uji H1 "alami"**: arsitektur yang secara bawaan peka degree; (b) **uji ketahanan**: apakah efek tangga atribut dan faktor degree konsisten lintas arsitektur (RQ4). **Bukan** ajang adu model.

| Backbone | Layer PyG | Atribut edge | Peka degree | Peran |
|---|---|---|---|---|
| GraphSAGE (mean) | `SAGEConv` | tidak | tidak | replikasi Yoo |
| GATv2 | `GATv2Conv` | ya (`edge_dim`) | tidak (softmax) | basis 2 × 2 |
| DA-GAT | `GATv2Conv` + faktor | ya | ya (faktor) | uji H1 |
| TransformerConv | `TransformerConv` | ya, di dalam attention | tidak | ketahanan H2 pada attention lain |
| GINEConv | `GINEConv` | ya, ditambahkan ke pesan | ya (agregasi sum, GIN) | uji H1 alami |
| PNAConv | `PNAConv` | ya (`edge_dim`) | ya (scaler degree) | uji H1 alami tanpa kode custom |

**Prediksi di bawah H1** (input tanpa volume): arsitektur peka-degree (GINE, PNA, DA-GAT) setara atau lebih baik dari yang tidak (SAGE-mean, GATv2, TransformerConv). Jika tidak terlihat, H1 melemah.

**Prioritas (2 bulan):**
- **P1:** GATv2 untuk 2 × 2 dan GE-1 (5 split × 10 seed).
- **P2:** GINEConv dan PNAConv, dengan dan tanpa atribut edge (5 split × 3 seed).
- **P3:** TransformerConv dan GraphSAGE (5 split × 3 seed), bila waktu ada.

**Catatan teknis (belum diverifikasi):** dokumentasi `PNAConv` memakai satu angka untuk `in_channels` (bukan pasangan), sehingga kemungkinan tidak mendukung graf bipartit (provider dan pasien beda dimensi). `GINEConv` dan `TransformerConv` belum dicek. Jika tidak mendukung, proyeksikan provider dan pasien ke dimensi yang sama dan pakai graf homogen (`to_homogeneous`), atau bungkus per relasi dengan `HeteroConv` bila didukung. Cek dokumentasi PyG sebelum minggu 3.

### 4.8 Diagnostik mekanisme

- **C1 — korelasi Spearman** tiap kolom X_cent terhadap log(n_claims) dan log(n_bene), di semua provider. Bukti langsung dan hampir gratis bahwa centrality mencerminkan volume.
- **D1 — probe linear:** Ridge dari embedding provider (lapis terakhir sebelum classifier) ke log(n_claims) dan log(n_bene), R² di provider test, untuk G0, G3, GE3.
- **Catatan D1.** Pada DA-GAT, faktor degree membuat embedding menyandi volume *secara konstruksi*, jadi D1 hanya bukti mekanisme (G0 gagal menyandi volume, G3 berhasil). Bukti **manfaat** adalah selisih performa G3 vs G0 dan G1.
- **Nilai $\gamma_r$ yang dipelajari** (per relasi per layer) dilaporkan sebagai indikator relasi mana yang informasi volumenya dipakai model.

### 4.9 Matriks eksperimen

Semua GNN memakai input node **X_mean** kecuali dinyatakan lain, supaya volume hanya bisa datang dari struktur graf.

| Kode | Model | Input | Menguji |
|---|---|---|---|
| E0 | LR dan XGBoost, **level klaim** | fitur klaim ± centrality | **Premis** (replikasi pola Yoo) |
| M0 | LR, RF, XGB, LGBM, MLP | X_mean | Dasar tanpa volume dan graf |
| M1 | idem | X_mean + X_cent | H1 (replikasi "ML+centrality") |
| M1b | idem | X_mean + X_cent **tanpa kolom degree** | H1: apakah PageRank dkk. hanya proksi volume? |
| M2 | idem | X_mean + X_vol | H1: **kontrol volume** |
| M3 | idem | X_mean + X_vol + X_cent | H1: centrality menambah di atas volume? |
| C1 | korelasi Spearman | X_cent vs log volume | H1 |
| G0 | GAT | X_mean | Sel dasar 2 × 2; replikasi GNN Yoo |
| G1 | GAT | X_mean + log-degree | H1: cukupkah degree sebagai fitur input? |
| G3 | DA-GAT, γ dipelajari | X_mean | H1 (sel 2 × 2) |
| GE-1 | GAT + 1 atribut edge (nilai klaim) | X_mean + `InscClaimAmtReimbursed` di edge | H2: tangga langkah 1 (setara bobot edge) |
| GE | GAT + atribut klaim | X_mean + atribut edge | H2: tangga langkah 2 (sel 2 × 2) |
| **GE3** | **DA-GAT + atribut klaim** | X_mean + atribut edge | **H1 + H2 (sel 2 × 2)** |
| D1 | probe Ridge | embedding G0, G3, GE3 | Mekanisme H1 |
| G2 | DA-GAT, γ dikunci 1 | X_mean | Opsional: scaler PNA-style |
| HAN | HAN | X_mean | Opsional: GNN terbaik Yoo |
| SAGE | GraphSAGE (mean) | X_mean (± atribut edge: tidak didukung) | Replikasi Yoo (P3) |
| GINE | GINEConv, ± atribut edge | X_mean (+ atribut edge) | H1 alami (P2) |
| PNA | PNAConv, ± atribut edge | X_mean (+ atribut edge) | H1 alami (P2) |
| TCONV | TransformerConv, ± atribut edge | X_mean (+ atribut edge) | Ketahanan H2 (P3) |
| H3x | embedding GNN terbaik → XGBoost | X_mean + embedding | H3 opsional |

**Uji H0 bukan model baru:** semua model di atas dinilai dengan dua protokol (4.10).

### 4.10 Protokol evaluasi (dikunci)

**Dua protokol untuk uji H0:**

| | Protokol "ala Yoo" | Protokol adil (utama) |
|---|---|---|
| Level metrik | klaim | provider (+ klaim-terbobot sebagai pembanding) |
| Ambang keputusan | 0,5 (**asumsi**; Yoo tidak menyebutkannya) | F1 tertinggi di validation |
| Yang dilaporkan | selisih GNN–ML | selisih GNN–ML |

Jika selisih GNN–ML jauh lebih kecil di protokol adil, sebagian kekalahan GNN di Yoo adalah artefak evaluasi (H0 didukung).

**Aturan umum:**
- **Split:** per provider, stratified, 60/20/20. Satu split tetap untuk pengembangan; tabel akhir model kunci (M1, M2, G0, G1, G3, GE-1, GE, GE3) memakai **5 split acak**.
- **Model GNN:** hidden 64, 2 layer, 4 head, dropout 0,3, cross-entropy berbobot kelas ($w_c=N/(2N_c)$), AdamW, early stopping pada AUPRC validation. Nilai awal; tuning ringan di validation.
- **Adil:** G0, G1, G3, GE-1, GE, GE3 (dan backbone lain pada 4.7b) diberi anggaran tuning yang sama.
- **Seed:** 10 per konfigurasi prioritas 1; 3 untuk prioritas 2 dan 3 (4.7b); rata-ratakan di dalam split.
- **Metrik:** precision, recall, F1 (kelas fraud), AUROC, **AUPRC**.
- **Uji beda:** berpasangan **per split** (n = 5), bukan per seed (seed hanya mengukur keragaman inisialisasi). Laporkan interval, jangan hanya selisih rata-rata.
- **Efisiensi:** waktu training dan inferensi per model.
- **Komputasi:** ukur waktu satu run di minggu 1 sebelum menetapkan jumlah seed/split. Perkiraan: P1 (G0, G1, G3, GE-1, GE, GE3) 5 split × 10 seed × 6 = 300 run; P2 dan P3 (4 backbone × dengan/tanpa atribut edge) 5 split × 3 seed × 8 = 120 run. Kurangi seed bila perlu dan catat.

---

## 5. Paper: klaim, outline, dan risiko

### 5.1 Judul

*Are Graphs Worth the Trouble? A Diagnostic Study of Graph Neural Networks for Healthcare Provider Fraud Detection*

**Alasan pemilihan.** Pola judul: pengait yang mudah dipahami ("apakah graf sepadan dengan repotnya?") + subjudul deskriptif yang memuat kata kunci untuk pencarian (*graph neural networks, healthcare, provider fraud detection*). Judul ini tetap benar apa pun hasilnya, termasuk jika GNN ternyata tidak kalah dari ML+centrality, karena tidak mengandung asumsi hasil dan tidak menyebut mekanisme tertentu (volume, atribut klaim, information loss).

**Aturan:** hindari kata "improving", "outperforming", dan "mengatasi" sampai hasil ada. Pakai "healthcare", bukan "Medicare", karena asal-usul dataset tidak jelas. Panjang ≈ 17 kata.

**Judul lama yang dibuang:** *Why Do Graph Neural Networks Underperform Centrality-Based Machine Learning in Healthcare Provider Fraud Detection? Testing Volume and Claim-Attribute Explanations*. Alasan: mengasumsikan GNN pasti kalah (salah jika premis gagal di gate minggu 3), tidak menyebut H0, terlalu panjang (≈ 26 kata), dan terlalu teknis untuk pembaca umum.

**Alternatif yang ditawarkan (tidak dipilih):**
- *What Do Graph Neural Networks Actually See in Healthcare Provider Fraud? A Controlled Diagnostic Study*
- *Can Graph Neural Networks See What Simple Features See? Healthcare Provider Fraud Detection Revisited*
- *Graphs Versus Features: Diagnosing When Graph Neural Networks Help in Healthcare Fraud Detection*

**Judul bersyarat (pertimbangkan di minggu 7, hanya jika hasilnya mendukung):**

| Jika hasilnya | Judul |
|---|---|
| Premis gagal (GNN tidak kalah saat dinilai adil) | *Do Graph Neural Networks Really Underperform Feature-Based Models in Healthcare Provider Fraud Detection? A Controlled Re-Evaluation* |
| H1 didukung (volume penjelasnya) | *Counting Matters: Why Graph Attention Misses Provider Volume in Healthcare Fraud Detection* |
| Tidak ada intervensi yang membantu | *Why Simple Graph Features Still Win: A Controlled Look at Graph Neural Networks in Healthcare Fraud Detection* |

Judul final boleh disesuaikan di akhir sesuai hasil; itu wajar dalam penulisan paper.

### 5.2 Pertanyaan riset

- **RQ0 (premis):** Apakah kekalahan GNN dari ML+centrality di Yoo dapat direplikasi?
- **RQ1 (H0):** Berapa bagian selisih yang berasal dari cara evaluasi?
- **RQ2 (H1):** Seberapa besar keunggulan centrality adalah efek volume, dan apakah agregasi sadar degree menutup selisih?
- **RQ3 (H2):** Apakah menambahkan atribut klaim di edge (tangga GE-1 → GE) menutup selisih, dan makin banyak informasi klaim makin kecil selisihnya?
- **RQ4 (ketahanan):** Apakah efek tangga atribut dan faktor degree konsisten lintas arsitektur (GATv2, GINE, PNA, TransformerConv, GraphSAGE)?

### 5.3 Skenario hasil dan isi laporan

| Pola hasil | Interpretasi | Klaim paper |
|---|---|---|
| GNN tidak kalah di replikasi (E0/G0 ≈ M1) | Premis tidak bertahan | "Kekalahan GNN di Yoo tidak bertahan dengan protokol adil" |
| Selisih mengecil drastis di protokol adil | H0 didukung | Sebagian besar selisih adalah artefak evaluasi |
| M2 ≈ M1, C1 tinggi, G3 menutup selisih | H1 didukung | Keunggulan centrality sebagian besar efek volume; agregasi sadar degree memulihkannya |
| G1 ≈ G3 | H1 didukung, scaler tidak perlu | Rekomendasi praktis: beri degree sebagai fitur |
| GE menutup selisih | H2 didukung | Atribut klaim yang hilang menjelaskan kekalahan |
| Selisih menyempit bertahap G0 → GE-1 → GE | H2 didukung dengan pola dose-response | Makin banyak informasi klaim, makin kecil selisih |
| GINE/PNA ≥ GATv2 tanpa intervensi | H1 didukung lewat desain arsitektur | Agregasi peka-degree memulihkan informasi volume, tanpa kode custom |
| Efek intervensi hanya pada GAT, tidak pada backbone lain | Efek spesifik arsitektur | Laporkan sebagai batas generalisasi |
| GE3 > G3 dan GE3 > GE | H1 dan H2 saling melengkapi | Kedua informasi diperlukan |
| Tidak ada intervensi yang membantu, M1 ≫ M2 | H1 dan H2 ditolak | Centrality membawa sinyal struktur yang tidak tertangkap GAT; temuan Yoo diperkuat dengan kontrol |

### 5.4 Kontribusi (bersyarat pada hasil)

1. Studi diagnostik pertama (sejauh yang kami ketahui; pencarian literatur tambahan di 2.5 tidak menemukan studi serupa, tetapi terbatas) yang menguji penjelasan atas kekalahan GNN dari ML+centrality pada data klaim kesehatan.
2. Dekomposisi keunggulan centrality menjadi komponen volume dan struktur (M1b, M2, M3, C1).
3. Dua intervensi (DA-GAT; tangga atribut klaim) dalam desain faktorial 2 × 2, diuji lintas beberapa backbone GNN (4.7b). Kalimat novelty yang aman: lihat 2.5.4.
4. Protokol evaluasi adil dan replikasi terkontrol dengan pelaporan level provider dan klaim.

### 5.5 Risiko

- **Kebaruan komponen:** semua komponen punya preseden (2.5.3). Kebaruan ada di diagnosis. Overlap parsial: **G-GBM** (fitur volume sudah cukup pada Kaggle yang sama), **Almeida 2024** (replikasi Yoo; hasil belum dibaca), **RDPGL** (atribut klaim di edge, dataset sama). Ulangi pencarian di Google Scholar sebelum minggu 1 (kata kunci di 2.5.5).
- **Ekspektasi dosen:** kata "atasi" bisa berarti beliau mengharapkan hasil positif. Selaraskan di minggu 1 (1.2): paper analisis vs paper metode.
- **Premis bisa gagal** (lihat 5.3 baris pertama).
- **Ukuran efek kecil:** provider train ≈ 3.200; pakai 5 split, laporkan interval.
- **Klaim kausal:** ablasi menunjukkan kontribusi, bukan bukti kausal mutlak.

### 5.6 Jangan diklaim

- "Studi komparatif" sebagai framing utama, atau "model baru yang lebih unggul".
- "Membuktikan penyebab"; gunakan "menguji penjelasan".
- DA-GAT atau GE3 sebagai arsitektur umum yang lebih baik di luar data ini.
- Edge-klaim atau scaler degree sebagai temuan arsitektur baru.
- "GNN lain" (GINE, PNA, dst.) sebagai kontribusi: itu uji ketahanan dan uji H1 alami.
- "Kami pertama yang mengatasi information loss"; "volume cukup" sebagai temuan baru (G-GBM sudah menunjukkannya pada dataset yang sama).
- Generalisasi ke BPJS/Indonesia; perbandingan langsung dengan angka RDPGL; angka penghematan Yoo sebagai fakta.

### 5.7 Outline paper

1. **Abstrak (5 kalimat):** temuan Yoo → dugaan yang tidak diuji → tiga penjelasan → desain intervensi → hasil.
2. **Pendahuluan:** fraud kolusif dan daya tarik GNN; temuan Yoo yang berlawanan intuisi; pertanyaan "kenapa"; empat kontribusi.
3. **Kajian pustaka:** (a) ML untuk fraud klaim (du Preez, Nabrawi, Farahmandazad); (b) fitur graf + ML (Chandola, Branting, Herland, Yoo, Deprez); (c) GNN untuk fraud kesehatan (Yoo 2022, RDPGL, StGNN, MBGNN, Muhammad, MHAMFD); (d) keterbatasan agregator GNN dan penskalaan degree (Xu, PNA, GRIT, MBGNN) sebagai landasan H1; (e) fitur edge dan multigraf (RDPGL, Egressy, GRANDE) sebagai landasan H2; (f) GNN vs model tabular (BGNN, Grinsztajn, G-GBM, Weber) sebagai landasan H3 dan premis.
4. **Data dan graf:** dataset, penggabungan, split, tiga kelompok fitur, atribut klaim di edge, deviasi dari Yoo.
5. **Hipotesis dan metode:** H0–H2 dan prediksinya; baseline; dua intervensi; diagnostik; protokol.
6. **Hasil:** (1) replikasi premis; (2) H0; (3) H1; (4) H2; (5) gabungan 2 × 2 dan γ; (6) efisiensi.
7. **Diskusi:** penjelasan mana yang terbukti, kapan GNN layak dipakai, keterbatasan, kerja lanjutan (imbalance, BPJS tanpa ID relasional, explainability).
8. **Kesimpulan.**

### 5.8 Gambar dan tabel yang direncanakan

- **Gambar:** (1) alur eksperimen; (2) GAT vs DA-GAT vs edge beratribut; (3) tangga atribut G0 → GE-1 → GE; (4) hasil 2 × 2; (5) perbandingan backbone (P1–P3); (6) R² probe D1; (7) nilai γ yang dipelajari.
- **Tabel:** (1) statistik data vs Tabel 3 Yoo; (2) hipotesis → eksperimen; (3) hasil ML M0–M3, M1b; (4) korelasi C1; (5) hasil GNN dan 2 × 2 di dua protokol; (6) waktu training.

### 5.9 Calon venue (belum dicek scope/biaya/waktu review)

IEEE Access (tempat Yoo; framing "revisiting" cocok), Scientific Reports, BMC Medical Informatics and Decision Making, Expert Systems with Applications. Untuk batas 2 bulan, konferensi regional atau jurnal open access dengan review cepat lebih realistis. **Verifikasi sendiri.**

---

## 6. Rencana 8 minggu dengan titik keputusan

| Minggu | Pekerjaan | Gate |
|---|---|---|
| 1 | Unduh, gabung, split; ringkasan vs Tabel 3 Yoo; X_mean/X_vol/X_cent; graf + atribut edge; centrality; ukur waktu satu run GNN; **konsultasi dosen (1.2)**; **cek dukungan bipartit PyG untuk GINE/PNA/TransformerConv** | Jumlah klaim dan provider mendekati Yoo |
| 2 | E0, M0–M3, M1b, C1 | **Gate 1 (premis ML):** pola kualitatif Yoo muncul di E0 (baseline rendah, centrality provider–beneficiary melonjak dan > provider–physician)? Bukan angka persis. Jika tidak, periksa gabung/split |
| 3 | G0, GraphSAGE (HAN opsional); semua model dinilai dua protokol | **Gate 2 (premis GNN + H0):** di protokol adil, apakah G0 masih lebih lemah dari M1? Jika tidak, H0/premis menjelaskan selisih; laporan bergeser (5.3) |
| 4 | G1, G3, GE-1, GE, GE3 (+ G2 opsional); 5 split × 10 seed | — |
| 5 | Backbone lain: GINEConv, PNAConv (P2), TransformerConv (P3); D1; H3x opsional | **Gate 3 (akhir minggu 5):** apakah ada intervensi yang menutup selisih secara berpasangan per split? Jika tidak, laporan fokus pada H0/H1 lewat M2, M1b, C1 dan uji H1 alami (GINE/PNA) |
| 6 | Robustness: edge per klaim vs pasangan unik, pilihan δ, definisi `PROJECT_PROVIDER` | — |
| 7 | Tabel dan gambar; tulis metode dan hasil | — |
| 8 | Tulis sisa, revisi, buffer | — |

**Pemangkasan bila waktu mepet** (tulis sebagai deviasi): hitung hanya degree dan PageRank (di regresi logistik Yoo hanya PageRank yang signifikan); lewati HAN, G2, dan backbone P3 (TransformerConv, GraphSAGE); kurangi seed menjadi 5 (P1) dan 2 (P2).

---

## 7. Status dan hal yang belum terverifikasi

**Sudah:** pemetaan paper; backbone; framing diagnostik; desain 2 × 2 dan tangga atribut; backbone lain; outline paper; judul; pencarian literatur tambahan (2.5). File pendukung dari sesi sebelumnya:
- `draft-proposal-dagat.pdf` (6 halaman) **sudah usang**: masih berdesain DA-GAT saja, belum memuat H0/H2, GE, GE3, tangga GE-1, backbone lain, dan 2.5. Perlu diperbarui.
- Paket kode referensi `fraud_gnn/` hanya diuji pada data sintetis dan berisi `RelAttnGNN`, **bukan** DA-GAT atau GE3. User menulis kodenya sendiri.

**Belum:**
- Dataset Kaggle asli belum diunduh dan diperiksa; fraud rate ≈ 9% dan jumlah provider konsisten dengan G-GBM (3.1), tetapi belum dicek langsung.
- Semua eksperimen belum dijalankan; semua hipotesis belum teruji; premis belum direplikasi.
- Apakah graf Yoo satu edge per klaim, dan apakah atribut klaim mereka dipakai di edge: belum pasti.
- Level metrik dan ambang keputusan Yoo: tidak dinyatakan di paper.
- Rumus GAT/GATv2/HAN/HGT dan referensi Xu/Graphormer ditulis dari pengetahuan umum, **belum dibaca ulang**. Abstrak PNA, GRIT, BGNN, Grinsztajn, Egressy, dan dokumentasi PyG (`GATv2Conv`, `PNAConv`) sudah dicek, tetapi bukan teks penuh.
- Link GitHub RDPGL, Muhammad, Argade, POCL, dan G-GBM belum dicek aktif (beberapa situs memblokir akses otomatis).
- Pencarian literatur tambahan **sudah dilakukan tetapi terbatas** (2.5.5); ulangi di Google Scholar sebelum minggu 1. Hasil Almeida (2024) dan teks penuh Deprez, MHAMFD, MHGSL belum dibaca.
- Dukungan graf bipartit di PyG untuk `PNAConv`, `GINEConv`, `TransformerConv` belum dicek (4.7b).
- Arahan dosen (1.2) belum diselaraskan soal target paper metode vs analisis.

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
- F. Vandervorst, B. Deprez, W. Verbeke, T. Verdonck, "Inductive inference of gradient-boosted decision trees on graphs for insurance fraud detection," *Data Min. Knowl. Discov.* 40(5):78, 2026. doi:10.1007/s10618-026-01249-0; arXiv:2510.05676
- B. Deprez, F. Vandervorst, W. Verbeke, T. Verdonck, B. Baesens, "Network analytics for insurance fraud detection: a critical case study," *Eur. Actuar. J.* 14:965–990, 2024. doi:10.1007/s13385-024-00384-6
- J.-P. Chen, P. Lu, F. Yang, R. Chen, K. Lin, "Medical insurance fraud detection using graph neural networks with spatio-temporal constraints," *J. Netw. Intell.* 7(2):480–498, 2022.
- J.-T. Lu, K.-B. Lin, X. Chen, J.-P. Chen, "Multibehavioral pattern-based graph neural network with topology awareness to detect healthcare fraud," *J. Netw. Intell.* 8(3):895–912, 2023.
- D. Farahmandazad, K. Danesh, "ML-driven approaches to combat Medicare fraud," arXiv:2502.15898, 2025.
- D. Argade, K. Gandeeban, "Benchmarking TabPFN against traditional and relational models for provider fraud detection," *ISSF 2026* (Atlantis Press), 2026.
- Rakesh, D. P. Shetty, "Enhanced Medicare fraud detection using graph convolutional networks," *AISP 2024*. doi:10.1109/AISP61711.2024.10870744
- R. Zhang et al., "Pre-trained online contrastive learning for insurance fraud detection," *AAAI-24*. doi:10.1609/AAAI.V38I20.30259
- R. D. R. Pereira, F. Murai, "How effective are graph neural networks in fraud detection for network data?" arXiv:2105.14568, 2021.
- L. Grinsztajn, E. Oyallon, G. Varoquaux, "Why do tree-based models still outperform deep learning on typical tabular data?" *NeurIPS Datasets & Benchmarks*, 2022. arXiv:2207.08815
- G. Corso, L. Cavalleri, D. Beaini, P. Liò, P. Veličković, "Principal neighbourhood aggregation for graph nets," *NeurIPS* 2020. arXiv:2004.05718
- "Graph inductive biases in transformers without message passing" (GRIT), arXiv:2305.17589 (penulis dan venue belum dicek).
- B. Egressy et al., "Provably powerful graph neural networks for directed multigraphs," *AAAI-24*. doi:10.1609/aaai.v38i10.29069
- R. Wu, B. Ma, "GRANDE: a neural model over directed multigraphs with application to anti-money laundering," arXiv:2302.02101.
- S. Ivanov, L. Prokhorenkova, "Boost then convolve: gradient boosting meets graph neural networks," *ICLR* 2021. arXiv:2101.08543
- M. Weber et al., "Anti-money laundering in Bitcoin: experimenting with graph convolutional networks for financial forensics," arXiv:1908.02591, 2019.
- F. M. Almeida, "Detecção de fraudes no setor de saúde: métodos de inteligência artificial," tesis, USP, 2024 (hasil belum dibaca).
- ConFVG, "Healthcare insurance fraud detection via continual Fiedler vector graph model," ICLR 2026 (poster; penulis belum dicek).
- *Dari pengetahuan umum, verifikasi sebelum dikutip:* P. Veličković et al., "Graph Attention Networks," ICLR 2018; S. Brody et al., "How Attentive are Graph Attention Networks?" (GATv2), ICLR 2022; K. Xu et al., "How Powerful are Graph Neural Networks?" ICLR 2019; G. Corso et al., "Principal Neighbourhood Aggregation for Graph Nets," NeurIPS 2020; C. Ying et al., "Do Transformers Really Perform Bad for Graph Representation?" NeurIPS 2021; X. Wang et al., HAN, WWW 2019; Z. Hu et al., HGT, WWW 2020; W. Hamilton et al., GraphSAGE, NeurIPS 2017.
