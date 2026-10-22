"""uji_he.py - pemeriksaan cepat hitungan Histogram Equalization.

Jalankan:  python uji_he.py
Isinya: (1) contoh hitung tangan pada citra kecil 4x4 dengan 8 tingkat keabuan (dipakai di laporan),
(2) perbandingan hitungan sendiri dengan cv2.equalizeHist pada beberapa citra.
"""
import numpy as np

import he_core as he

# --- 1. Contoh kecil: 4 x 4 piksel, nilai 0..7 (L = 8) -----------------------------------
kecil = np.array(
    [[1, 1, 2, 2],
     [1, 2, 2, 3],
     [2, 3, 3, 3],
     [3, 3, 3, 4]],
    dtype=np.uint8,
)
hasil, hist, lut = he.equalize_manual(kecil, levels=8)
cdf = he.cdf_of(hist)

print("Citra awal:\n", kecil)
print("\nNilai lama | jumlah piksel | CDF | nilai baru")
for k in np.nonzero(hist)[0]:
    print(f"{k:^10} | {hist[k]:^13} | {cdf[k]:^3} | {lut[k]:^10}")
print("\nCitra hasil:\n", hasil)

# --- 2. Bandingkan dengan OpenCV ----------------------------------------------------------
print("\nSelisih terbesar hitungan sendiri vs cv2.equalizeHist:")
rng = np.random.default_rng(1)
uji = {
    "citra contoh sintetis": he.make_demo_image(),
    "acak merata": rng.integers(0, 256, (200, 300), dtype=np.uint8),
    "acak gelap": rng.integers(0, 60, (200, 300), dtype=np.uint8),
    "acak sempit 100-140": rng.integers(100, 141, (200, 300), dtype=np.uint8),
}
for nama, data in uji.items():
    kanal = data[..., 0] if data.ndim == 3 else data
    sendiri = he.equalize_manual(kanal)[0]
    opencv = he.equalize_opencv(kanal)
    print(f"  {nama:<24}: {int(np.abs(sendiri.astype(int) - opencv.astype(int)).max())}")
