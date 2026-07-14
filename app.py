"""
app.py — Prediksi Kelayakan Pinjaman (Indonesia)
=================================================
Cukup panggil `pipeline.predict()` / `pipeline.predict_proba()` pada data
mentah (raw). SEMUA imputasi, feature engineering, log-transform, encoding,
dan scaling sudah dibungkus di dalam `loan_pipeline.joblib` -- app ini TIDAK
menduplikasi logika preprocessing apa pun secara manual, sehingga tidak
mungkin terjadi selisih antara notebook training dan aplikasi ini.
(Ini sengaja dipertahankan dari desain v12 -- lihat loan_utils.py.)
"""
import json
import re

import joblib
import pandas as pd
import streamlit as st

from loan_utils import CATEGORY_OPTIONS, build_raw_input, predict_loan

st.set_page_config(
    page_title="Prediksi Kelayakan Pinjaman",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════
#  THEME
# ══════════════════════════════════════════════════════════════════════════
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True
D = st.session_state.dark_mode

P = {
    "bg": "#0d1117" if D else "#f0f4f8",
    "bg2": "#161b22" if D else "#ffffff",
    "bg3": "#21262d" if D else "#f6f8fa",
    "tx": "#e6edf3" if D else "#1c2526",
    "tx2": "#8b949e" if D else "#57606a",
    "bdr": "#30363d" if D else "#d0d7de",
    "gold": "#FFD700" if D else "#9a6700",
    "green": "#3fb950" if D else "#1a7f37",
    "red": "#f78166" if D else "#cf222e",
    "blue": "#58a6ff" if D else "#0969da",
    "orange": "#e3a000" if D else "#9a5700",
    "s_bg": "#1a4d2e" if D else "#dafbe1",
    "e_bg": "#4d1a1a" if D else "#ffebe9",
    "w_bg": "#3d2e00" if D else "#fff3cd",
    "grad": "linear-gradient(135deg,#0f2027,#203a43,#2c5364)" if D
            else "linear-gradient(135deg,#e8f4fd,#c9e6fa,#a8d5f0)",
    "ht": "#ffffff" if D else "#1c3d5a",
    "hs": "#a8d8f0" if D else "#2d6a9f",
}

st.markdown(f"""<style>
.stApp, .main .block-container {{background:{P['bg']} !important; color:{P['tx']} !important}}
section[data-testid="stSidebar"] {{background:{P['bg2']} !important; border-right:1px solid {P['bdr']} !important}}
section[data-testid="stSidebar"] * {{color:{P['tx']} !important}}
h1,h2,h3,h4,p,label,span,div {{color:{P['tx']} !important}}
hr {{border-color:{P['bdr']} !important}}
input, textarea, [data-baseweb="input"] input {{
    background:{P['bg3']} !important; color:{P['tx']} !important;
    border:1px solid {P['bdr']} !important; border-radius:8px !important}}
[data-baseweb="select"]>div {{background:{P['bg3']} !important; border:1px solid {P['bdr']} !important;
    color:{P['tx']} !important; border-radius:8px !important}}
[data-testid="stMetric"] {{background:{P['bg2']} !important; border:1px solid {P['bdr']} !important;
    border-radius:12px !important; padding:14px 18px !important}}
[data-testid="stMetricValue"] {{color:{P['gold']} !important; font-weight:900 !important}}
[data-testid="stDataFrame"] table {{background:{P['bg2']} !important; color:{P['tx']} !important}}
[data-testid="stExpander"] {{background:{P['bg2']} !important; border:1px solid {P['bdr']} !important; border-radius:10px !important}}
.warn-box {{background:{P['w_bg']}; border-left:3px solid {P['orange']}; border-radius:0 8px 8px 0;
    padding:9px 14px; margin:4px 0; font-size:.88em}}
.tip-row {{display:flex; justify-content:space-between; padding:5px 0;
    border-bottom:1px solid {P['bdr']}; font-size:.87em}}
</style>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════
def fmt(v):
    """Format angka dengan titik per 3 digit: 1000000 -> 1.000.000"""
    return f"{int(round(v)):,}".replace(",", ".")


def _reformat_idr(key):
    """Callback on_change: bersihkan input, tulis ulang dgn titik setiap 3 digit."""
    digits = re.sub(r"[^\d]", "", st.session_state.get(key, ""))
    st.session_state[key] = fmt(int(digits)) if digits else ""


def idr_field(label, default, key, help_text=None, max_val=9_999_999_999):
    """Input Rupiah dengan auto-format titik (1.000.000.000) setiap kali
    pengguna selesai mengetik (Enter / pindah fokus). Mendukung nilai hingga miliaran."""
    if key not in st.session_state:
        st.session_state[key] = fmt(default)
    st.text_input(label, key=key, on_change=_reformat_idr, args=(key,),
                  help=help_text or f"Contoh: {fmt(default)}")
    digits = re.sub(r"[^\d]", "", st.session_state[key])
    val = int(digits) if digits else 0
    return max(0, min(max_val, val))


def star(n, total=5):
    return "★" * n + "☆" * (total - n)


def pretty_model_name(name):
    """'DecisionTreeClassifier' -> 'Decision Tree Classifier' (kosmetik saja;
    model_info.json menyimpan type(model).__name__ dari notebook training)."""
    return re.sub(r"(?<!^)(?=[A-Z])", " ", name) if name else "-"


def render_table(headers, rows):
    """Tabel HTML ringan (bukan st.dataframe/st.table) -- konsisten dengan tema
    dark/light kustom, dan menghindari lapisan serialisasi pyarrow yang tidak
    perlu untuk tabel statis kecil seperti ini."""
    thead = "".join(f'<th style="text-align:left;padding:8px 12px;background:{P["bg3"]};'
                     f'color:{P["gold"]};border-bottom:2px solid {P["bdr"]}">{h}</th>' for h in headers)
    trs = ""
    for row in rows:
        tds = "".join(f'<td style="padding:7px 12px;border-bottom:1px solid {P["bdr"]};'
                       f'color:{P["tx"]}">{v}</td>' for v in row)
        trs += f"<tr>{tds}</tr>"
    return (f'<table style="width:100%;border-collapse:collapse;font-size:.88em;'
            f'margin:6px 0 14px">{thead}{trs}</table>')


# ══════════════════════════════════════════════════════════════════════════
#  LOAD MODEL
# ══════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_pipeline():
    return joblib.load("loan_pipeline.joblib")


@st.cache_data
def load_model_info():
    try:
        with open("model_info.json", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


try:
    pipeline = load_pipeline()
except FileNotFoundError:
    st.error(
        "File `loan_pipeline.joblib` tidak ditemukan di folder yang sama dengan `app.py`. "
        "Pastikan file ini (beserta `loan_utils.py` dan `model_info.json`) ikut di-push/di-upload."
    )
    st.stop()

info = load_model_info()

# Statistik training (fallback ke nilai wajar kalau model_info lama / belum ada field ini)
INCOME_MIN_TRAIN = info.get("income_min_train", 200_000)
INCOME_MED_TRAIN = info.get("income_median_train", 3_800_000)
LOAN_MAX_TRAIN = info.get("loan_max_train", 130_000_000)
TOP_FACTORS = info.get("top_factors") or [
    {"label": "Riwayat Kredit", "weight": 1.0},
    {"label": "Total Pendapatan", "weight": 0.8},
    {"label": "Rasio Beban Pinjaman", "weight": 0.6},
    {"label": "Cicilan Bulanan (EMI)", "weight": 0.5},
    {"label": "Tenor Pinjaman", "weight": 0.3},
]

# Tenor: nilai standar dataset (bulan), diurutkan TERLAMA -> TERCEPAT
TENOR_OPTIONS = [480, 360, 300, 240, 180, 120, 84, 60, 36, 12]

# ══════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════
bc = "#f0c040" if D else "#2d6a9f"
st.markdown(f"""
<div style="background:{P['grad']};padding:36px 40px 30px;border-radius:18px;
     text-align:center;margin-bottom:22px;box-shadow:0 8px 32px rgba(0,0,0,.3)">
  <div style="border:2px solid {bc};border-radius:14px;padding:22px 32px">
    <p style="color:{P['gold']};font-size:.85em;letter-spacing:4px;margin:0 0 6px">
      🏦 MACHINE LEARNING PROJECT</p>
    <h1 style="color:{P['ht']};font-size:1.9em;font-weight:900;margin:8px 0 6px">
      Prediksi Kelayakan Pinjaman</h1>
    <p style="color:{P['hs']};font-size:1em;margin:0 0 14px">
      {pretty_model_name(info.get('model_family', 'Model'))} · Dataset Rupiah Indonesia (IDR)</p>
  </div>
</div>""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
c1.metric("🤖 Model", pretty_model_name(info.get("model_family", "-")))
c2.metric("✅ OOF Accuracy", f"{info.get('oof_accuracy', 0) * 100:.1f}%")
c3.metric("📊 OOF Macro-F1", f"{info.get('oof_macro_f1', 0):.4f}")
st.markdown("<hr style='margin:8px 0 20px'>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    if st.button("☀️ Mode Terang" if D else "🌙 Mode Gelap", key="theme_btn"):
        st.session_state.dark_mode = not D
        st.rerun()

    st.markdown(f"""
    <div style="background:{P['bg3']};border:1px solid {P['bdr']};border-radius:10px;
         padding:14px 16px;margin:10px 0">
      <p style="color:{P['gold']};font-weight:700;font-size:.82em;letter-spacing:2px;margin:0 0 8px">
        📖 PANDUAN</p>
      <p style="font-size:.86em;color:{P['tx2']};margin:0;line-height:1.6">
        Isi semua kolom, lalu klik<br>
        <b style="color:{P['gold']}">🔮 Prediksi Kelayakan</b>
      </p>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"<p style='color:{P['gold']};font-weight:700;font-size:.82em;"
                f"letter-spacing:2px;margin-top:16px'>🏆 FAKTOR TERPENTING</p>", unsafe_allow_html=True)
    for n, factor in enumerate(TOP_FACTORS, 1):
        n_stars = max(1, round(factor["weight"] * 5))
        st.markdown(
            f'<div class="tip-row"><span>{n}. {factor["label"]}</span>'
            f'<span style="color:{P["gold"]}">{star(n_stars)}</span></div>',
            unsafe_allow_html=True)

    st.markdown(f"<p style='color:{P['gold']};font-weight:700;font-size:.82em;"
                f"letter-spacing:2px;margin-top:16px'>🎯 INTERPRETASI HASIL</p>", unsafe_allow_html=True)
    for col, ic, msg in [
        (P["green"], "✅", "Prob ≥ 60% → DISETUJUI"),
        (P["orange"], "⚠️", "Prob 40–59% → BORDERLINE"),
        (P["red"], "❌", "Prob < 40% → DITOLAK"),
    ]:
        st.markdown(
            f'<div style="background:{P["bg3"]};border-left:3px solid {col};'
            f'border-radius:0 6px 6px 0;padding:7px 12px;margin:5px 0;font-size:.86em">'
            f'{ic} {msg}</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div style="margin-top:16px;padding:11px 13px;background:{P['bg3']};
         border-radius:8px;font-size:.8em;color:{P['tx2']};line-height:1.65">
      💡 <b style="color:{P['gold']}">Format angka:</b><br>
      • Ketik <b>5000000</b> atau <b>5.000.000</b><br>
      • Keduanya diterima otomatis
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
#  INPUT FORM
# ══════════════════════════════════════════════════════════════════════════
st.markdown(f"<p style='color:{P['gold']};font-weight:700;font-size:.82em;"
            f"letter-spacing:2px;margin-bottom:16px'>📋 DATA PEMOHON PINJAMAN</p>", unsafe_allow_html=True)

col_l, col_r = st.columns(2, gap="large")

with col_l:
    st.markdown(f"<p style='color:{P['tx2']};font-size:.78em;letter-spacing:2px;"
                f"font-weight:700;margin-bottom:10px'>👤 DATA PERSONAL</p>", unsafe_allow_html=True)
    jenis_kelamin = st.selectbox("Jenis Kelamin", CATEGORY_OPTIONS["Jenis_Kelamin"])
    status_pernikahan = st.selectbox("Status Pernikahan", CATEGORY_OPTIONS["Status_Pernikahan"])
    jumlah_tanggungan = st.selectbox("Jumlah Tanggungan", CATEGORY_OPTIONS["Jumlah_Tanggungan"],
                                      help="Anggota keluarga yang ditanggung")
    pendidikan = st.selectbox("Pendidikan", CATEGORY_OPTIONS["Pendidikan"])
    wiraswasta = st.selectbox("Status Pekerjaan", CATEGORY_OPTIONS["Wiraswasta"],
                               help="'Ya' = wiraswasta/usaha sendiri, 'Tidak' = karyawan/PNS")
    wilayah_properti = st.selectbox("Lokasi Properti", CATEGORY_OPTIONS["Wilayah_Properti"],
                                     help="Wilayah lokasi properti yang diajukan")

with col_r:
    st.markdown(f"<p style='color:{P['tx2']};font-size:.78em;letter-spacing:2px;"
                f"font-weight:700;margin-bottom:10px'>💰 DATA FINANSIAL</p>", unsafe_allow_html=True)
    pendapatan = idr_field("Pendapatan Pemohon (Rp/bulan)", default=5_000_000,
                            key="pendapatan_input", help_text="Gaji/penghasilan bulanan pemohon utama")
    pendapatan_co = idr_field("Pendapatan Co-Pemohon (Rp/bulan)", default=0,
                               key="pendapatan_co_input", help_text="Isi 0 jika tidak ada co-pemohon")
    jumlah_pinjaman = idr_field("Jumlah Pinjaman Diajukan (Rp)", default=100_000_000,
                                 key="jumlah_pinjaman_input", help_text="Total pinjaman yang diajukan")
    tenor_pinjaman = st.selectbox(
        "Tenor Pinjaman", TENOR_OPTIONS,
        index=TENOR_OPTIONS.index(360),
        format_func=lambda x: f"{x} Hari  (~{x // 12} Bulan)",
    )
    riwayat_kredit = st.selectbox("Riwayat Kredit", CATEGORY_OPTIONS["Riwayat_Kredit"],
                                   help="Riwayat pembayaran kredit sebelumnya")

# ── Validasi & peringatan input di luar rentang wajar ──────────────────────
total_income = pendapatan + pendapatan_co
loan_income_ratio = jumlah_pinjaman / (total_income + 1)
emi_bulanan = jumlah_pinjaman / tenor_pinjaman

warnings_input = []
if pendapatan < INCOME_MIN_TRAIN:
    warnings_input.append(
        f"⚠️ Pendapatan Rp {fmt(pendapatan)}/bln sangat rendah (di bawah data training). "
        "Prediksi mungkin kurang akurat.")
if loan_income_ratio > 5:
    warnings_input.append(
        f"⚠️ Rasio Beban Pinjaman = {loan_income_ratio:.1f} — sangat tidak wajar. "
        "Pinjaman jauh melebihi kemampuan bayar.")
if jumlah_pinjaman > LOAN_MAX_TRAIN:
    warnings_input.append(
        f"⚠️ Jumlah pinjaman Rp {fmt(jumlah_pinjaman)} melebihi rentang data training "
        f"(maks ≈ Rp {fmt(LOAN_MAX_TRAIN)}).")
if total_income > 0 and emi_bulanan > total_income * 0.8:
    warnings_input.append(
        f"⚠️ Cicilan bulanan ≈ Rp {fmt(emi_bulanan)} ({emi_bulanan / total_income * 100:.0f}% "
        "dari pendapatan) — sangat berat.")

if warnings_input:
    st.markdown(
        f'<div class="warn-box" style="border-radius:10px;padding:12px 16px;margin:8px 0">'
        f'<p style="color:{P["orange"]};font-weight:700;font-size:.85em;margin:0 0 6px">'
        f'⚠️ PERHATIAN — INPUT DI LUAR RENTANG NORMAL</p>'
        + "".join(f'<p style="color:{P["orange"]};font-size:.85em;margin:3px 0">{w}</p>'
                   for w in warnings_input)
        + "</div>", unsafe_allow_html=True)

st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
predict_btn = st.button("🔮  Prediksi Kelayakan Pinjaman", type="primary", use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
#  PREDICTION
# ══════════════════════════════════════════════════════════════════════════
if predict_btn:
    raw_input = build_raw_input(
        jenis_kelamin=jenis_kelamin, status_pernikahan=status_pernikahan,
        jumlah_tanggungan=jumlah_tanggungan, pendidikan=pendidikan, wiraswasta=wiraswasta,
        pendapatan_pemohon=pendapatan, pendapatan_pendamping=pendapatan_co,
        jumlah_pinjaman=jumlah_pinjaman, tenor_pinjaman=tenor_pinjaman,
        riwayat_kredit=riwayat_kredit, wilayah_properti=wilayah_properti,
    )
    _, p_yes = predict_loan(pipeline, raw_input)

    if p_yes >= 0.60:
        tier, tier_icon = "DISETUJUI", "✅"
        tier_bg, tier_bdr, tier_tx = P["s_bg"], P["green"], P["green"]
        tier_msg = ("Profil sangat kuat — kelayakan sangat tinggi." if p_yes >= 0.80
                    else "Pemohon memenuhi kriteria kelayakan pinjaman.")
    elif p_yes >= 0.40:
        tier, tier_icon = "BORDERLINE — Perlu Pertimbangan", "⚠️"
        tier_bg, tier_bdr, tier_tx = P["w_bg"], P["orange"], P["orange"]
        tier_msg = "Persetujuan tidak pasti. Perkuat profil keuangan sebelum mengajukan."
    else:
        tier, tier_icon = "DITOLAK", "❌"
        tier_bg, tier_bdr, tier_tx = P["e_bg"], P["red"], P["red"]
        tier_msg = "Profil belum memenuhi syarat. Lihat rekomendasi di bawah."

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{P['gold']};font-weight:700;font-size:.82em;"
                f"letter-spacing:2px;margin-bottom:10px'>🎯 HASIL PREDIKSI</p>", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:{tier_bg};border:2px solid {tier_bdr};border-radius:14px;
         padding:20px 28px;margin-bottom:14px">
      <div style="display:flex;align-items:center;gap:12px">
        <span style="font-size:1.8em">{tier_icon}</span>
        <div>
          <p style="color:{tier_tx};font-size:1.3em;font-weight:900;margin:0">{tier}</p>
          <p style="color:{tier_tx};font-size:.9em;margin:0;opacity:.85">{tier_msg}</p>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    bar_color = P["green"] if p_yes >= 0.60 else (P["orange"] if p_yes >= 0.40 else P["red"])
    st.markdown(f"""
    <div style="margin-bottom:4px;display:flex;justify-content:space-between;font-size:.83em">
      <span style="color:{P['red']}">0% Ditolak</span>
      <span style="color:{P['orange']}">40% – Borderline – 60%</span>
      <span style="color:{P['green']}">100% Disetujui</span>
    </div>
    <div style="background:{P['bg3']};border-radius:8px;height:18px;position:relative;
         overflow:hidden;margin-bottom:16px;border:1px solid {P['bdr']}">
      <div style="width:{p_yes * 100:.1f}%;background:{bar_color};height:100%;border-radius:8px;
           transition:width .4s"></div>
    </div>""", unsafe_allow_html=True)

    pa, pb = st.columns(2)
    pa.metric("Probabilitas Disetujui", f"{p_yes * 100:.1f}%", delta=f"{(p_yes - 0.5) * 100:+.1f}% dari batas 50%")
    pb.metric("Probabilitas Ditolak", f"{(1 - p_yes) * 100:.1f}%")

    # ── Tabel analisis faktor (semua nilai string agar aman utk st.dataframe) ──
    st.markdown(f"<p style='color:{P['gold']};font-weight:700;font-size:.82em;"
                f"letter-spacing:2px;margin-top:18px;margin-bottom:8px'>📊 ANALISIS FAKTOR</p>",
                unsafe_allow_html=True)

    ch_ok = "Baik" in riwayat_kredit
    lr_label = "aman ✅" if loan_income_ratio < 0.5 else ("perhatian ⚠️" if loan_income_ratio < 1 else "berbahaya ❌")
    fact_rows = [
        ("Riwayat Kredit", "Baik ✅" if ch_ok else "Buruk ❌",
         "✅ Mendukung" if ch_ok else "❌ Risiko Tinggi"),
        ("Total Pendapatan", f"Rp {fmt(total_income)}/bln",
         "✅ Mencukupi" if total_income > 5_000_000 else "⚠️ Perlu Ditingkatkan"),
        ("Jumlah Pinjaman", f"Rp {fmt(jumlah_pinjaman)}",
         "✅ Wajar" if jumlah_pinjaman < 50_000_000 else "⚠️ Besar"),
        ("Cicilan Bulanan (EMI)", f"Rp {fmt(emi_bulanan)}/bln",
         "✅ Ringan" if emi_bulanan < total_income * 0.3 else "⚠️ Berat"),
        ("Rasio Beban Pinjaman", f"{loan_income_ratio:.3f}  ({lr_label})",
         "✅ Aman" if loan_income_ratio < 0.5 else
         "⚠️ Perlu Perhatian" if loan_income_ratio < 1 else "❌ Terlalu Tinggi"),
        ("Tenor", f"{tenor_pinjaman} bulan (~{tenor_pinjaman // 12} thn)",
         "✅ Standar" if tenor_pinjaman == 360 else "◇ Non-Standar"),
    ]
    st.markdown(render_table(["Faktor", "Nilai", "Status"], fact_rows), unsafe_allow_html=True)

    # ── Rekomendasi ─────────────────────────────────────────────────────────
    st.markdown(f"<p style='color:{P['gold']};font-weight:700;font-size:.82em;"
                f"letter-spacing:2px;margin-top:20px;margin-bottom:8px'>💡 REKOMENDASI</p>",
                unsafe_allow_html=True)

    recs = []
    if not ch_ok:
        recs.append((P["red"], "❌ Perbaiki Riwayat Kredit (Prioritas Utama)",
                     "Riwayat kredit buruk adalah penolak terbesar. Langkah: lunasi/negosiasikan "
                     "ulang tunggakan dengan kreditur, lalu ajukan kembali setelah 6–12 bulan bersih."))
    if loan_income_ratio > 0.5:
        safe_loan = int(total_income * 0.4)
        recs.append((P["orange"] if loan_income_ratio < 1 else P["red"],
                     f"⚠️ Kurangi Jumlah Pinjaman (saat ini Rp {fmt(jumlah_pinjaman)})",
                     f"Rasio {loan_income_ratio:.2f} terlalu tinggi. Dengan pendapatan Rp {fmt(total_income)}/bln, "
                     f"pinjaman ideal ≈ Rp {fmt(safe_loan)} (rasio ≈ 0.4), atau tambahkan co-pemohon."))
    if total_income > 0 and emi_bulanan > total_income * 0.5:
        recs.append((P["orange"], f"⚠️ Cicilan Terlalu Berat ({emi_bulanan / total_income * 100:.0f}% pendapatan)",
                     f"Pertimbangkan tenor lebih panjang (360–480 bulan) untuk menurunkan cicilan bulanan."))
    if total_income < INCOME_MED_TRAIN * 0.5:
        recs.append((P["orange"], f"⚠️ Pendapatan Relatif Rendah (Rp {fmt(total_income)}/bln)",
                     f"Di bawah median pemohon tipikal (≈ Rp {fmt(INCOME_MED_TRAIN)}/bln). "
                     "Tambahkan co-pemohon atau tunggu kenaikan gaji sebelum mengajukan."))
    if 0.40 <= p_yes < 0.65:
        recs.append((P["blue"], "◇ Approval Borderline — Perkuat Dokumen",
                     f"Probabilitas {p_yes * 100:.0f}% tergolong tipis. Siapkan slip gaji 3 bulan, "
                     "rekening koran 6 bulan, dan surat keterangan kerja untuk memperkuat pengajuan."))
    if not recs:
        recs.append((P["green"], "✅ Profil Kuat — Tips Mempertahankan Kelayakan",
                     "Tetap jaga riwayat kredit bersih, hindari utang baru sebelum pinjaman cair, "
                     "dan siapkan DP minimal 20% jika ini KPR."))

    for color, title, desc in recs:
        st.markdown(f"""
        <div style="background:{P['bg3']};border-left:4px solid {color};border-radius:0 10px 10px 0;
             padding:12px 16px;margin:7px 0">
          <p style="color:{color};font-weight:700;font-size:.9em;margin:0 0 5px">{title}</p>
          <p style="color:{P['tx2']};font-size:.86em;margin:0;line-height:1.6">{desc}</p>
        </div>""", unsafe_allow_html=True)

    # ── Ringkasan input yang dikirim ke model ───────────────────────────────
    with st.expander("Lihat data yang dikirim ke model"):
        summary_rows = [(col, str(raw_input.iloc[0][col])) for col in raw_input.columns]
        st.markdown(render_table(["Kolom", "Nilai"], summary_rows), unsafe_allow_html=True)

    # ── Penjelasan non-monoton (khusus model tree-based) ────────────────────
    model_family = info.get("model_family", "")
    if model_family in ("RandomForestClassifier", "GradientBoostingClassifier", "DecisionTreeClassifier"):
        with st.expander("🔍 Mengapa probabilitas bisa turun meski pendapatan naik?"):
            st.markdown(f"""
            <div style="color:{P['tx']};font-size:.88em;line-height:1.7">
            <p><b style="color:{P['gold']}">Ini perilaku normal model berbasis pohon keputusan</b>,
            bukan error.</p>
            <b>Penjelasan teknis:</b>
            <ul style="color:{P['tx2']}">
              <li>Model ini mempelajari aturan-aturan bercabang dari data training, bukan garis lurus
                  seperti regresi.</li>
              <li>Model <b>tidak dijamin monoton</b> — pendapatan naik sedikit TIDAK selalu berarti
                  probabilitas naik.</li>
              <li>Perubahan kecil bisa menggeser titik data ke cabang berbeda dan mengubah hasil.</li>
            </ul>
            <b>Kapan prediksi paling bisa dipercaya?</b>
            <ul style="color:{P['tx2']}">
              <li>Probabilitas <b>&lt; 30% atau &gt; 70%</b> → sinyal kuat, bisa dipercaya.</li>
              <li>Probabilitas <b>30–70%</b> → borderline, fokus pada perbaikan faktor utama.</li>
              <li>Selalu perhatikan <b>Riwayat Kredit</b> dan <b>Rasio Beban Pinjaman</b> sebagai
                  indikator utama, bukan hanya angka probabilitas.</li>
            </ul>
            </div>""", unsafe_allow_html=True)

with st.expander("ℹ️ Tentang model ini"):
    st.write(f"**Algoritma**: {pretty_model_name(info.get('model_family', '-'))}")
    st.write(f"**OOF Accuracy (train CV)**: {info.get('oof_accuracy', '-')}")
    st.write(f"**OOF Macro-F1 (train CV)**: {info.get('oof_macro_f1', '-')}")
    st.write(f"**OOF ROC-AUC (train CV)**: {info.get('oof_roc_auc', '-')}")
    st.caption(
        "Catatan penting: dataset dasar berasal dari dataset publik *Loan Prediction* "
        "(Analytics Vidhya/Kaggle) yang diterjemahkan & dikonversi secara ilustratif ke "
        "skema Indonesia/IDR untuk keperluan pembelajaran. Ini BUKAN data pinjaman riil "
        "Indonesia, dan model ini TIDAK boleh dipakai untuk keputusan kredit sungguhan."
    )

st.markdown("<hr style='margin:24px 0 12px'>", unsafe_allow_html=True)
st.markdown(f"""
<div style="text-align:center;color:{P['tx2']};font-size:.78em;line-height:2">
  🤖 {pretty_model_name(info.get('model_family', 'Model'))} &nbsp;·&nbsp;
  📊 Dataset Loan Prediction — Kaggle &nbsp;·&nbsp;
  🔗 Kurs: {info.get('kurs_info', '-')}<br>
  ⚠️ Hasil bersifat indikatif — bukan keputusan resmi lembaga keuangan
</div>""", unsafe_allow_html=True)
