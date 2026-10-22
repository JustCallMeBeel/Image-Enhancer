"""app_he.py - Aplikasi Histogram Equalization untuk citra astronomi (tugas Pengolahan Citra Digital).

Dasar: kode praktikum dosen, https://github.com/marsani/Operasi_TitikCitra (lisensi GPL-3.0).
Diambil dari sana: pilihan Global vs CLAHE, pemrosesan kanal kecerahan untuk citra berwarna,
dan grafik histogram + CDF. Ditambahkan di sini: perhitungan dari nol dengan NumPy, tab
"Langkah demi langkah", tabel statistik, dan tombol unduh hasil.

Jalankan:  streamlit run app_he.py
"""
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import he_core as he

st.set_page_config(page_title="Histogram Equalization", layout="wide")
st.title("Histogram Equalization untuk Citra Astronomi")
st.caption("Tugas Pengolahan Citra Digital - metode terpilih: Histogram Equalization")


# ---------------------------------------------------------------------------
# Fungsi gambar grafik
# ---------------------------------------------------------------------------
def plot_hist_cdf(channel_before, channel_after):
    """Histogram (batang) + CDF (garis oranye, skala 0-100%) sebelum dan sesudah."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    for ax, ch, title, color in [
        (axes[0], channel_before, "Sebelum", "#4F46E5"),
        (axes[1], channel_after, "Sesudah", "#10B981"),
    ]:
        h = he.histogram(ch)
        ax.bar(np.arange(256), h, width=1.0, color=color, alpha=0.85)
        ax.set_xlim(0, 255)
        ax.set_xlabel("Nilai piksel (0 = gelap, 255 = terang)")
        ax.set_ylabel("Jumlah piksel")
        ax.set_title(f"Histogram dan CDF - {title}", fontweight="bold")
        ax2 = ax.twinx()
        ax2.plot(np.arange(256), np.cumsum(h) / h.sum() * 100, color="#F59E0B", linewidth=2)
        ax2.set_ylim(0, 100)
        ax2.set_ylabel("CDF (%)")
    fig.tight_layout()
    return fig


def plot_single_hist(channel):
    h = he.histogram(channel)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(np.arange(256), h, width=1.0, color="#4F46E5", alpha=0.85)
    ax.set_xlim(0, 255)
    ax.set_xlabel("Nilai piksel")
    ax.set_ylabel("Jumlah piksel")
    fig.tight_layout()
    return fig


def plot_cdf(channel):
    cdf = he.cdf_of(he.histogram(channel))
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(np.arange(256), cdf / cdf[-1] * 100, color="#F59E0B", linewidth=2)
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Nilai piksel")
    ax.set_ylabel("CDF (%)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_lut(lut):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot([0, 255], [0, 255], color="#9CA3AF", linestyle="--", label="tanpa perubahan")
    ax.plot(np.arange(256), lut, color="#10B981", linewidth=2, label="tabel pemetaan")
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.set_xlabel("Nilai lama")
    ax.set_ylabel("Nilai baru")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Sidebar: pilihan pengguna
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("1. Pilih citra")
    upload = st.file_uploader("Unggah citra", type=["png", "jpg", "jpeg", "bmp", "webp"])
    if upload is None:
        st.info("Belum ada citra diunggah, jadi dipakai citra contoh sintetis (bukan dari APOD).")

    st.header("2. Pilih algoritma")
    algo = st.radio("Algoritma", list(he.ALGORITHMS.keys()), format_func=he.ALGORITHMS.get)
    clip, grid = 2.0, 8
    if algo == "clahe":
        clip = st.slider("Clip limit", 1.0, 8.0, 2.0, 0.5, help="Makin besar, makin kuat kontrasnya (dan noise ikut naik).")
        grid = st.select_slider("Ukuran blok", options=[4, 8, 16], value=8, format_func=lambda v: f"{v} x {v}")

    st.header("3. Citra berwarna")
    color_mode = st.radio(
        "Diproses sebagai",
        ["brightness", "gray"],
        format_func=lambda v: "Kecerahan saja (warna dipertahankan)" if v == "brightness" else "Abu-abu",
    )

# ---------------------------------------------------------------------------
# Muat citra dan proses
# ---------------------------------------------------------------------------
if upload is not None:
    rgb = np.array(Image.open(upload).convert("RGB"))
    source_name = upload.name
else:
    rgb = he.make_demo_image()
    source_name = "contoh sintetis"

out_rgb, before_ch, after_ch = he.process(rgb, algo, color_mode, clip, grid)
st.write(f"Citra: **{source_name}** - ukuran {rgb.shape[1]} x {rgb.shape[0]} piksel")

tab_hasil, tab_langkah, tab_teori = st.tabs(["Hasil", "Langkah demi langkah", "Tentang metode"])

# ---------------------------------------------------------------------------
# Tab 1: Hasil
# ---------------------------------------------------------------------------
with tab_hasil:
    c1, c2 = st.columns(2)
    with c1:
        st.image(rgb, caption="Sebelum", clamp=True)
    with c2:
        st.image(out_rgb, caption=f"Sesudah - {he.ALGORITHMS[algo]}", clamp=True)

    st.subheader("Histogram dan CDF")
    st.caption("Sesudah equalization, batang histogram menyebar ke seluruh rentang dan garis CDF mendekati diagonal lurus.")
    fig = plot_hist_cdf(before_ch, after_ch)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Angka sebelum dan sesudah")
    s_before, s_after = he.stats(before_ch), he.stats(after_ch)
    st.dataframe(pd.DataFrame({"Sebelum": s_before, "Sesudah": s_after}))

    buf = io.BytesIO()
    Image.fromarray(out_rgb).save(buf, format="PNG")
    st.download_button("Unduh hasil (PNG)", buf.getvalue(), file_name="hasil_equalization.png", mime="image/png")

# ---------------------------------------------------------------------------
# Tab 2: Langkah demi langkah (selalu memakai perhitungan dari nol, algoritma Global)
# ---------------------------------------------------------------------------
with tab_langkah:
    st.write("Empat langkah Global Histogram Equalization pada kanal kecerahan citra di atas.")
    hist = he.histogram(before_ch)
    cdf = he.cdf_of(hist)
    manual_out, _, lut = he.equalize_manual(before_ch)

    st.markdown("**Langkah 1 - Hitung histogram.** Berapa banyak piksel untuk tiap nilai 0 sampai 255.")
    fig = plot_single_hist(before_ch)
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("**Langkah 2 - Hitung CDF.** Jumlah kumulatif: berapa persen piksel yang nilainya sama dengan atau lebih kecil dari k.")
    fig = plot_cdf(before_ch)
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("**Langkah 3 - Buat tabel pemetaan.** Setiap nilai lama dipetakan ke nilai baru dengan rumus:")
    st.latex(r"s_k = \mathrm{round}\left(\frac{\mathrm{CDF}(k) - \mathrm{CDF}_{min}}{M \times N - \mathrm{CDF}_{min}} \times (L-1)\right)")
    st.caption("M x N = jumlah semua piksel, L = 256 tingkat keabuan, CDF_min = CDF pada nilai terkecil yang muncul.")
    present = np.nonzero(hist)[0]
    sample = present[:10]
    st.dataframe(
        pd.DataFrame(
            {"Nilai lama": sample, "Jumlah piksel": hist[sample], "CDF": cdf[sample], "Nilai baru": lut[sample]}
        ).set_index("Nilai lama")
    )
    st.caption("Contoh 10 nilai tergelap yang muncul di citra ini.")
    fig = plot_lut(lut)
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("**Langkah 4 - Ganti setiap piksel** dengan nilai barunya dari tabel.")
    cv_out = he.equalize_opencv(before_ch)
    diff = int(np.abs(manual_out.astype(int) - cv_out.astype(int)).max())
    st.success(f"Pemeriksaan: selisih terbesar antara hitungan sendiri dan OpenCV = {diff} tingkat keabuan (0 berarti sama persis).")

# ---------------------------------------------------------------------------
# Tab 3: Tentang metode
# ---------------------------------------------------------------------------
with tab_teori:
    st.markdown(
        """
**Histogram Equalization** meratakan sebaran nilai keabuan supaya memakai seluruh rentang gelap sampai terang.
Pada citra yang gelap, hampir semua piksel menumpuk di nilai rendah, sehingga detail redup tidak terlihat.
Metode ini menarik nilai-nilai itu menyebar, jadi perbedaan kecil menjadi lebih jelas.

**Global** memakai satu tabel pemetaan untuk seluruh citra. Cepat dan mudah dijelaskan, tetapi bisa memperkuat noise
di latar yang gelap. **CLAHE** membagi citra menjadi blok-blok kecil dan membatasi penguatan kontras (clip limit),
sehingga noise lebih terkendali.

**Dipakai untuk:** memperjelas nebula dan galaksi redup pada foto astronomi, citra rontgen yang kontrasnya rendah,
dan sebagai tahap awal sebelum analisis citra (misalnya deteksi tepi atau segmentasi).
"""
    )
