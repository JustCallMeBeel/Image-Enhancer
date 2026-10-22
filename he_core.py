"""he_core.py - inti perhitungan Histogram Equalization (tanpa tampilan).

Dasar: bagian "Histogram Equalization" pada kode praktikum dosen,
https://github.com/marsani/Operasi_TitikCitra (lisensi GPL-3.0). Dari sana diambil:
  - pilihan algoritma Global (cv2.equalizeHist) dan CLAHE (cv2.createCLAHE),
  - cara memproses citra berwarna lewat kanal kecerahan saja (ruang warna YCrCb),
  - grafik histogram + CDF sebelum/sesudah.
Bagian yang ditambahkan di sini: perhitungan "dari nol" dengan NumPy (build_lut,
equalize_manual) supaya setiap langkah bisa dilihat, citra contoh sintetis, dan statistik.

File ini hanya berisi fungsi. Tampilannya ada di app_he.py.
"""
import cv2
import numpy as np

LEVELS = 256  # citra 8-bit: nilai piksel 0..255


# ---------------------------------------------------------------------------
# Langkah-langkah dasar (dipakai untuk penjelasan)
# ---------------------------------------------------------------------------
def histogram(channel, levels=LEVELS):
    """Langkah 1: hitung berapa piksel yang bernilai 0, 1, 2, ..., levels-1."""
    return np.bincount(channel.ravel(), minlength=levels)


def cdf_of(hist):
    """Langkah 2: jumlah kumulatif histogram (CDF) = berapa piksel yang nilainya <= k."""
    return np.cumsum(hist)


def build_lut(hist, levels=LEVELS):
    """Langkah 3: tabel pemetaan nilai lama -> nilai baru.

        s_k = round( (CDF(k) - CDF_min) / (M*N - CDF_min) * (L - 1) )

    M*N = jumlah seluruh piksel, L = jumlah tingkat keabuan, CDF_min = CDF pada nilai
    terkecil yang benar-benar muncul. Rumus ini sama dengan yang dipakai OpenCV.
    """
    cdf = cdf_of(hist)
    total = int(cdf[-1])
    first = int(np.nonzero(hist)[0][0])
    cdf_min = int(cdf[first])
    if total == cdf_min:  # semua piksel bernilai sama: tidak ada yang bisa diratakan
        return np.arange(levels, dtype=np.int64)
    lut = np.round((cdf - cdf_min) / (total - cdf_min) * (levels - 1))
    return np.clip(lut, 0, levels - 1).astype(np.int64)


def equalize_manual(channel, levels=LEVELS):
    """Langkah 4: terapkan tabel ke setiap piksel. Mengembalikan (hasil, histogram, tabel)."""
    hist = histogram(channel, levels)
    lut = build_lut(hist, levels)
    return lut.astype(np.uint8)[channel], hist, lut


# ---------------------------------------------------------------------------
# Algoritma siap pakai dari OpenCV (seperti pada kode dosen)
# ---------------------------------------------------------------------------
def equalize_opencv(channel):
    return cv2.equalizeHist(channel)


def equalize_clahe(channel, clip=2.0, grid=8):
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(channel)


ALGORITHMS = {
    "manual": "Global - dihitung sendiri (NumPy)",
    "opencv": "Global - OpenCV (cv2.equalizeHist)",
    "clahe": "CLAHE - lokal, kontras dibatasi",
}


def process(rgb, algo="manual", color_mode="brightness", clip=2.0, grid=8):
    """Jalankan equalization pada citra RGB uint8.

    color_mode "brightness": hanya kanal kecerahan (Y) yang diratakan, warna (Cr, Cb) tetap.
    color_mode "gray": citra diubah ke abu-abu dulu.
    Mengembalikan (citra_hasil_rgb, kecerahan_sebelum, kecerahan_sesudah).
    """
    if algo == "manual":
        fn = lambda ch: equalize_manual(ch)[0]
    elif algo == "opencv":
        fn = equalize_opencv
    elif algo == "clahe":
        fn = lambda ch: equalize_clahe(ch, clip, grid)
    else:
        raise ValueError(f"algoritma tidak dikenal: {algo}")

    if color_mode == "gray":
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        out_gray = fn(gray)
        return cv2.cvtColor(out_gray, cv2.COLOR_GRAY2RGB), gray, out_gray

    y, cr, cb = cv2.split(cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb))
    y2 = fn(y)
    out = cv2.cvtColor(cv2.merge((y2, cr, cb)), cv2.COLOR_YCrCb2RGB)
    return out, y, y2


def stats(channel):
    """Ringkasan angka: nilai terkecil/terbesar, rata-rata, simpangan baku (std)."""
    return {
        "Nilai terkecil": int(channel.min()),
        "Nilai terbesar": int(channel.max()),
        "Rata-rata": round(float(channel.mean()), 2),
        "Std (ukuran kontras)": round(float(channel.std()), 2),
    }


# ---------------------------------------------------------------------------
# Citra contoh sintetis (BUKAN data APOD) - hanya agar aplikasi bisa dicoba tanpa unggah
# ---------------------------------------------------------------------------
def make_demo_image(height=360, width=480, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(18, 4, (height, width))  # langit gelap + sedikit noise
    yy, xx = np.mgrid[0:height, 0:width]
    for cx, cy, spread, amp in [(150, 140, 60, 38), (330, 220, 80, 28), (250, 90, 35, 45)]:
        base += amp * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * spread**2)))  # kabut redup
    for _ in range(60):  # bintang
        x, y = rng.integers(1, width - 1), rng.integers(1, height - 1)
        base[y - 1 : y + 2, x - 1 : x + 2] += rng.uniform(60, 160)
    gray = np.clip(base, 0, 255)
    rgb = np.stack([gray * 0.85, gray * 0.92, gray], axis=-1)  # sedikit kebiruan
    return np.clip(rgb, 0, 255).astype(np.uint8)
