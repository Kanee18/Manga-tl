<p align="center">
  <img src="docs/images/logo.png" alt="MangaTL" width="140">
</p>

<h1 align="center">MangaTL</h1>

<p align="center">
  <b>Penerjemah manga otomatis untuk Windows: baca, hapus, terjemahkan, dan tulis ulang balon teks seperti hasil scanlation.</b><br>
  Pilih folder chapter, tekan satu tombol, dan dapatkan halaman yang balonnya sudah berisi terjemahan dengan font komik, bukan sekadar subtitle yang ditempel di atas gambar.
</p>

<p align="center">
  <a href="https://github.com/Kanee18/Manga-tl/releases/latest"><img alt="Download" src="https://img.shields.io/github/v/release/Kanee18/Manga-tl?label=download&color=ffd23f"></a>
  <img alt="Windows 10/11" src="https://img.shields.io/badge/Windows-10%20%7C%2011%20(64--bit)-5cc9f2">
  <img alt="Offline" src="https://img.shields.io/badge/bisa-100%25%20offline-86dfac">
  <img alt="Freeware" src="https://img.shields.io/badge/lisensi-freeware-ff82b2">
</p>

<p align="center">
  <img src="docs/images/preview-app.png" alt="Tampilan MangaTL" width="900">
</p>

---

## Daftar isi

- [Apa itu MangaTL](#apa-itu-mangatl)
- [Fitur](#fitur)
- [Spesifikasi komputer](#spesifikasi-komputer)
- [Download dan install](#download-dan-install)
- [Pilih penerjemah: offline atau online](#pilih-penerjemah-offline-atau-online)
- [Cara pakai](#cara-pakai)
- [Privasi](#privasi)
- [Kalau ada masalah](#kalau-ada-masalah)
- [Update dan uninstall](#update-dan-uninstall)
- [Melaporkan bug](#melaporkan-bug)
- [Lisensi](#lisensi)
- [Kredit](#kredit)

---

## Apa itu MangaTL

MangaTL mengerjakan semua langkah yang biasanya dikerjakan tim scanlation, langsung di komputer Anda:

1. **Menemukan** setiap balon dan teks di halaman.
2. **Membaca** teksnya (OCR), termasuk tulisan vertikal.
3. **Menerjemahkan** satu halaman sekaligus, supaya percakapan antarbalon tetap nyambung.
4. **Menghapus** teks asli. Balon polos diisi warna latarnya, teks di atas gambar direkonstruksi dengan AI inpainting.
5. **Menulis ulang** terjemahannya dengan font komik, disesuaikan dengan bentuk dan ukuran balonnya.

<p align="center">
  <img src="docs/images/before-after.png" alt="Sebelum dan sesudah diterjemahkan" width="900"><br>
  <sub>Kiri: halaman asli. Kanan: hasil MangaTL (Inggris → Indonesia), tanpa diedit manual.</sub>
</p>

---

## Fitur

- **Satu klik untuk satu chapter.** Buka folder gambar atau arsip (`.cbz`, `.cbr`, `.zip`, `.rar`, `.7z`), centang halaman yang mau diterjemahkan, lalu jalankan.
- **Bisa offline sepenuhnya** dengan model AI lokal lewat [Ollama](https://ollama.com), atau **online** lewat API: OpenAI, Anthropic Claude, Google Gemini, OpenRouter, Groq, DeepSeek, Mistral, atau server lain yang kompatibel dengan OpenAI.
- **Efek suara dan tulisan di gambar dibiarkan**, begitu juga nomor halaman dan tulisan yang memang tidak perlu diterjemahkan.
- **Bisa diedit langsung di aplikasi.** Ubah terjemahan, ukuran, posisi, atau tebal huruf per balon, hapus noda, lalu bandingkan dengan aslinya. Undo tersedia.
- **Series sheet**: catat nama karakter, istilah, dan gaya bicara tiap karakter supaya terjemahan konsisten dari chapter ke chapter.
- **Glosarium**: istilah yang harus selalu diterjemahkan dengan cara tertentu.
- **Lanjutkan kapan saja**: halaman yang sudah diterjemahkan disimpan, jadi bisa dikoreksi lagi nanti.
- **Export ke CBZ** untuk dibaca di aplikasi pembaca komik.
- **Tur interaktif** saat pertama dibuka. Tekan **F1** kapan saja untuk melihatnya lagi.
- **Tombol Report** untuk menyimpan semua data yang dibutuhkan saat melaporkan hasil terjemahan yang salah.

### Bahasa yang didukung

| Dari \ Ke | Indonesia | Inggris |
|---|:---:|:---:|
| **Inggris** | ✅ (paling banyak dioptimalkan) | — |
| **Jepang** | ✅ | ✅ |
| **Mandarin** | ✅ | ✅ |

---

## Spesifikasi komputer

| | Minimum | Rekomendasi |
|---|---|---|
| **Sistem operasi** | Windows 10 64-bit | Windows 11 64-bit |
| **Prosesor** | 64-bit (x64) | Prosesor modern 64-bit |
| **RAM** | 8 GB | 16 GB atau lebih |
| **Kartu grafis** | Tidak wajib, bisa jalan di CPU | NVIDIA dengan **VRAM 8 GB** (seri GTX 16 / RTX ke atas) |
| **Ruang disk** | ±3 GB (pakai penerjemah online) | ±11 GB (pakai penerjemah offline) |
| **Internet** | Untuk download saja | Hanya kalau memakai penerjemah online |

**Rincian ruang disk:** installer berukuran ±1,3 GB dan terpasang jadi ±2,8 GB. Mode offline butuh tambahan ±7,2 GB untuk model bahasa `gemma4:e2b`.

**Kecepatan** (diukur di RTX 3060 Ti 8 GB, RAM 24 GB, model `gemma4:e2b`):

| | |
|---|---|
| Halaman pertama setiap kali aplikasi dibuka | ±2 menit (memuat semua model) |
| Halaman berikutnya | **±30 detik** per halaman |
| Satu chapter 32 halaman | ±20 menit, bisa ditinggal |
| Pemakaian VRAM | ±3 GB |

Tanpa kartu grafis NVIDIA, aplikasi tetap berjalan, tapi satu halaman bisa butuh beberapa menit.

---

## Download dan install

1. Buka halaman **[Releases](https://github.com/Kanee18/Manga-tl/releases/latest)**.
2. Download **`MangaTL-Setup-1.0.0.exe`** (±1,3 GB).
3. Jalankan file tersebut.
   > Kalau muncul layar biru **"Windows protected your PC"**, klik **More info → Run anyway**. Peringatan ini muncul karena installer belum memiliki sertifikat digital berbayar, bukan karena ada virus.
4. Ikuti wizard-nya. Tidak perlu hak admin. Aplikasi dipasang di `%LOCALAPPDATA%\Programs\MangaTL`.
5. Biarkan pilihan **Create a desktop shortcut** tercentang kalau ingin ada ikon di Desktop.
6. Selesai. MangaTL ada di **Start Menu** dan Desktop.

Setelah itu, pilih penerjemahnya (langkah di bawah).

---

## Pilih penerjemah: offline atau online

MangaTL sendiri yang mendeteksi, membaca, menghapus, dan menulis teks. Bagian **menerjemahkan** dikerjakan model bahasa (AI), dan Anda bisa memilih salah satu dari dua cara ini.

### A. Offline dengan Ollama (gratis, privat)

Cocok kalau Anda punya kartu grafis NVIDIA dan tidak ingin ada data yang keluar dari komputer.

1. Install **[Ollama](https://ollama.com/download)** (versi 0.20 atau lebih baru).
2. Buka **Command Prompt**, lalu jalankan perintah ini (download ±7,2 GB, cukup sekali):
   ```
   ollama pull gemma4:e2b
   ```
3. Buka MangaTL. Penerjemah bawaannya sudah Ollama dengan model `gemma4:e2b`, jadi langsung bisa dipakai.

### B. Online lewat API (tanpa kartu grafis kuat)

Cocok untuk laptop atau komputer tanpa GPU NVIDIA. Biasanya hasilnya juga lebih bagus, tapi **layanan API berbayar sesuai pemakaian** (beberapa punya kuota gratis).

1. Buka **Settings** di MangaTL.
2. Di **Translator**, pilih layanan, misalnya **Google Gemini**, **OpenAI**, atau **Anthropic Claude**.
3. Klik link **Get a key from …** di bawah kolom API key, buat key di situs layanan tersebut, lalu tempel di kolom **API key**.
4. Tekan **Refresh** untuk memuat daftar model, pilih modelnya, lalu klik **OK**.

Setiap layanan mengingat key dan modelnya sendiri, jadi Anda bisa berpindah layanan kapan saja.

---

## Cara pakai

1. **Siapkan chapter** dalam satu folder berisi gambar halaman (`001.jpg`, `002.jpg`, …), atau satu file arsip `.cbz` / `.zip` / `.rar` / `.7z`.
2. Klik **Open Chapter** (`Ctrl+O`) atau **Open Archive**. Folder juga bisa langsung di-drag ke jendela aplikasi.
3. **Hilangkan centang** pada halaman yang tidak perlu diterjemahkan, seperti cover, halaman kredit, atau halaman yang isinya gambar saja.
4. Coba dulu satu halaman dengan **Translate Page** (`Ctrl+T`).
5. Periksa hasilnya. **Compare with original** memperlihatkan halaman sebelum dan sesudah. Tab **Lines** menampilkan teks asli dan terjemahannya.
6. Kalau sudah sesuai, klik **Translate Chapter** (`Ctrl+Shift+T`).
7. Ada balon yang kurang pas? Klik balonnya, ubah teks atau ukurannya di panel kanan, lalu **Save page**.

Hasil disimpan di sebelah folder chapter Anda, dalam folder berakhiran **`_ID`**. Klik **Export CBZ** untuk menjadikannya satu file komik.

> **Tips:** atur **Reading order** di Settings sebelum menerjemahkan satu chapter. Manga yang belum di-mirror memakai `rtl` (kanan ke kiri, bawaannya), sedangkan komik barat atau yang sudah di-mirror memakai `ltr`. Kalau urutannya salah, konteks percakapan bisa kacau.

---

## Privasi

- **Mode offline (Ollama):** tidak ada gambar, teks, atau data apa pun yang dikirim ke internet. Semuanya diproses di komputer Anda.
- **Mode online (API):** yang dikirim **hanya teks di dalam balon** (beserta catatan series dan glosarium). **Gambar halaman tidak pernah dikirim.** Teks tersebut diproses oleh layanan yang Anda pilih, sesuai kebijakan privasi masing-masing layanan.
- **API key** disimpan di komputer Anda sendiri (`%LOCALAPPDATA%\MangaTL\api_keys.json`) dan tidak pernah ikut di laporan bug.
- MangaTL tidak memerlukan akun, tidak ada iklan, dan tidak ada pelacakan.

---

## Kalau ada masalah

| Masalah | Solusi |
|---|---|
| Muncul **"Windows protected your PC"** saat install | Klik **More info → Run anyway**. |
| Aplikasi tidak mau terbuka | Buka `%LOCALAPPDATA%\MangaTL\startup-error.log` (tempel alamat itu di Explorer) dan kirimkan isinya saat melapor. |
| Status **translator: NOT READY** | **Mode offline:** Ollama belum berjalan atau model belum di-download. Jalankan `ollama pull gemma4:e2b`, atau buka aplikasi Ollama. **Mode online:** periksa API key dan koneksi internet. |
| Satu halaman butuh beberapa menit | Model tidak muat di VRAM. Jalankan `ollama ps` di Command Prompt: kolom PROCESSOR harus `100% GPU`. Tutup aplikasi lain yang memakai GPU, atau pakai penerjemah online. |
| Tidak yakin bagian mana yang bermasalah | Jalankan **self-test**: buka Command Prompt, lalu `"%LOCALAPPDATA%\Programs\MangaTL\MangaTL.exe" --selftest`. Aplikasi memeriksa 6 hal penting dan menandai yang gagal. |
| Satu balon masih berbahasa asli | Lihat statusnya di tab **Lines**. `skipped_sfx` artinya efek suara yang sengaja dibiarkan. `failed_fit` artinya terjemahannya tidak muat; ketik ulang yang lebih pendek di panel kanan. |
| Nama karakter berubah-ubah antarhalaman | Tambahkan namanya di **Series** atau **Glossary**. |

---

## Update dan uninstall

- **Update:** download installer versi baru dan jalankan. Versi lama akan langsung diganti.
- **Uninstall:** buka **Settings Windows → Apps → MangaTL → Uninstall**.
- Pengaturan, API key, catatan series, dan proyek Anda **tidak ikut terhapus** saat uninstall atau update. Semuanya ada di `%LOCALAPPDATA%\MangaTL`. Hapus folder itu kalau ingin benar-benar bersih.

---

## Melaporkan bug

1. Saat hasil terjemahan salah, pilih balonnya, lalu klik tombol **Report** di aplikasi dan tulis apa yang salah.
2. Aplikasi menyimpan laporannya di `%LOCALAPPDATA%\MangaTL\reports\`.
3. Buat laporan di halaman **[Issues](https://github.com/Kanee18/Manga-tl/issues)**, lalu lampirkan folder laporan tersebut (di-zip) beserta penjelasan singkat.

---

## Lisensi

MangaTL adalah **freeware**: gratis untuk dipakai, tetapi **bukan open source**. Kode sumbernya tidak dipublikasikan.

- ✅ Boleh dipakai gratis untuk keperluan pribadi.
- ❌ Tidak boleh dijual, dimodifikasi, di-reverse engineer, atau disebarkan ulang tanpa izin. Bagikan link ke halaman ini, bukan file installer-nya.

Ketentuan lengkap ada di [LICENSE](LICENSE).

**Tentang konten:** MangaTL hanya menerjemahkan file yang sudah Anda miliki. Aplikasi ini tidak mendownload komik dari mana pun. Hak untuk menerjemahkan dan menyebarkan suatu komik adalah tanggung jawab penggunanya.

---

## Kredit

MangaTL dibuat oleh **[Kanee18](https://github.com/Kanee18)**, dengan memakai karya hebat dari banyak pihak:

| Bagian | Karya |
|---|---|
| Deteksi teks dan balon | [ogkalu/comic-text-and-bubble-detector](https://huggingface.co/ogkalu/comic-text-and-bubble-detector) |
| Masker huruf dan baris | [dmMaze/comic-text-detector](https://github.com/dmMaze/comic-text-detector) |
| OCR (pembaca teks komik) | [genshiai-daichi/baberu-ocr](https://huggingface.co/genshiai-daichi/baberu-ocr) |
| Penghapus teks (inpainting) | [LaMa](https://github.com/advimman/lama), via [mayocream/lama-manga-onnx](https://huggingface.co/mayocream/lama-manga-onnx) dan [Carve/LaMa-ONNX](https://huggingface.co/Carve/LaMa-ONNX) |
| Model penerjemah offline | Gemma (`gemma4:e2b`) oleh Google, dijalankan lewat [Ollama](https://ollama.com) |
| Font | Anime Ace 2 oleh [Blambot](https://blambot.com), [Comic Neue](https://comicneue.com) |
| Antarmuka | [Qt](https://www.qt.io) dan [PyQt6](https://www.riverbankcomputing.com/software/pyqt/) |
| Mesin AI dan gambar | [ONNX Runtime](https://onnxruntime.ai), [OpenCV](https://opencv.org), [Skia](https://skia.org) via skia-python, [NumPy](https://numpy.org) |
| GPU | NVIDIA CUDA dan cuDNN |
| Installer | [Inno Setup](https://jrsoftware.org/isinfo.php) |

Daftar lengkap beserta lisensi masing-masing ada di [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).

Terima kasih kepada semua pembuat model dan pustaka di atas. Tanpa karya mereka, MangaTL tidak akan ada. 🙏
