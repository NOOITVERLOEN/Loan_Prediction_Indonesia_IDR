"""
loan_utils.py
=============
Modul bersama untuk proyek Prediksi Kelayakan Pinjaman (Indonesia).

PENTING: File ini dipakai BAIK oleh notebook training MAUPUN oleh app.py Streamlit.
Jangan disalin/diduplikasi jadi dua versi berbeda — cukup 1 sumber kebenaran (single
source of truth). Ini WAJIB karena `joblib.dump()` menyimpan pipeline yang berisi
objek `RawCleaner` custom; saat `joblib.load()` dipanggil di app.py, Python butuh
definisi class `RawCleaner` yang PERSIS SAMA (sama modul, sama isi) supaya bisa
di-unpickle. Kalau app.py mendefinisikan ulang class ini secara manual dan sedikit
berbeda, hasil prediksi bisa salah tanpa ada error yang terlihat.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# ──────────────────────────────────────────────────────────────────────────
# 1) KONVERSI DATASET ASLI (INR, kolom Inggris) -> SKEMA INDONESIA (IDR)
#    Hanya dipakai saat MELATIH ULANG model dari loan_train.csv yang asli.
#    Streamlit app TIDAK memakai bagian ini -- input form sudah dalam
#    skema Indonesia/IDR secara langsung.
# ──────────────────────────────────────────────────────────────────────────
KURS = 187.22  # ilustrasi 1 INR = Rp 187,22 (nilai tukar akan berubah dari waktu ke waktu;
               # tidak memengaruhi validitas model karena hanya skala linear yang
               # dinormalisasi ulang oleh StandardScaler)

PETA_KOLOM = {
    'Loan_ID': 'ID_Pinjaman', 'Gender': 'Jenis_Kelamin', 'Married': 'Status_Pernikahan',
    'Dependents': 'Jumlah_Tanggungan', 'Education': 'Pendidikan', 'Self_Employed': 'Wiraswasta',
    'ApplicantIncome': 'Pendapatan_Pemohon', 'CoapplicantIncome': 'Pendapatan_Pendamping',
    'LoanAmount': 'Jumlah_Pinjaman', 'Loan_Amount_Term': 'Tenor_Pinjaman',
    'Credit_History': 'Riwayat_Kredit', 'Property_Area': 'Wilayah_Properti',
    'Loan_Status': 'Status_Pinjaman',
}
PETA_NILAI = {
    'Jenis_Kelamin': {'Male': 'Laki-laki', 'Female': 'Perempuan'},
    'Status_Pernikahan': {'Yes': 'Menikah', 'No': 'Belum Menikah'},
    'Pendidikan': {'Graduate': 'Sarjana', 'Not Graduate': 'Bukan Sarjana'},
    'Wiraswasta': {'Yes': 'Ya', 'No': 'Tidak'},
    'Riwayat_Kredit': {1.0: 'Baik', 0.0: 'Buruk'},
    'Wilayah_Properti': {'Urban': 'Perkotaan', 'Rural': 'Pedesaan', 'Semiurban': 'Semi_perkotaan'},
    'Status_Pinjaman': {'Y': 'Disetujui', 'N': 'Ditolak'},
}


def load_and_convert(path: str) -> pd.DataFrame:
    """Baca loan_train.csv asli (INR/Inggris) -> DataFrame skema Indonesia (IDR)."""
    df_raw = pd.read_csv(path)
    df_id = df_raw.copy()
    df_id['Loan_ID'] = df_id['Loan_ID'].str.replace('LP', 'KP', regex=False)
    df_id['ApplicantIncome'] = (df_id['ApplicantIncome'] * KURS).round(0)
    df_id['CoapplicantIncome'] = (df_id['CoapplicantIncome'] * KURS).round(0)
    df_id['LoanAmount'] = (df_id['LoanAmount'] * 1000 * KURS).round(0)  # LoanAmount asli dalam ribuan INR
    df_id.rename(columns=PETA_KOLOM, inplace=True)
    for kolom, peta in PETA_NILAI.items():
        if kolom in df_id.columns:
            df_id[kolom] = df_id[kolom].map(peta)
    return df_id.drop(columns=['ID_Pinjaman']).copy()


# ──────────────────────────────────────────────────────────────────────────
# 2) OPSI KATEGORI -- dipakai untuk (a) encoding eksplisit & (b) membangun
#    dropdown di form Streamlit, supaya keduanya SELALU konsisten.
# ──────────────────────────────────────────────────────────────────────────
BINARY_CATS = {
    'Jenis_Kelamin':     ['Laki-laki', 'Perempuan'],
    'Status_Pernikahan': ['Belum Menikah', 'Menikah'],
    'Pendidikan':        ['Bukan Sarjana', 'Sarjana'],
    'Wiraswasta':        ['Tidak', 'Ya'],
    'Riwayat_Kredit':    ['Buruk', 'Baik'],
}
AREA_CATS = ['Pedesaan', 'Perkotaan', 'Semi_perkotaan']
DEPENDENTS_OPTIONS = ['0', '1', '2', '3+']
DEPENDENTS_MAP = {'0': 0, '1': 1, '2': 2, '3+': 3}

CATEGORY_OPTIONS = {
    **{k: v for k, v in BINARY_CATS.items()},
    'Wilayah_Properti': AREA_CATS,
    'Jumlah_Tanggungan': DEPENDENTS_OPTIONS,
}

NUMERIC_RAW = ['Pendapatan_Pemohon', 'Pendapatan_Pendamping', 'Jumlah_Pinjaman', 'Tenor_Pinjaman']
CATEGORICAL_RAW = ['Jenis_Kelamin', 'Status_Pernikahan', 'Pendidikan', 'Wiraswasta',
                    'Riwayat_Kredit', 'Wilayah_Properti']
RAW_FEATURE_ORDER = (NUMERIC_RAW + CATEGORICAL_RAW + ['Jumlah_Tanggungan'])


# ──────────────────────────────────────────────────────────────────────────
# 3) TRANSFORMER CUSTOM: imputasi + feature engineering + log-transform.
#    Semua statistik (median/modus) HANYA dipelajari dari data yang dilihat
#    .fit() -> di dalam sklearn Pipeline itu artinya HANYA training fold,
#    tidak pernah data validasi/test. Ini yang mencegah data leakage.
# ──────────────────────────────────────────────────────────────────────────
class RawCleaner(BaseEstimator, TransformerMixin):
    """Imputasi (median/modus dari TRAIN saja) + feature engineering + log1p."""

    def fit(self, X, y=None):
        X = X.copy()
        self.numeric_medians_ = {c: X[c].median() for c in NUMERIC_RAW}
        self.categorical_modes_ = {c: X[c].mode(dropna=True).iloc[0] for c in CATEGORICAL_RAW}
        self.dependents_mode_ = X['Jumlah_Tanggungan'].mode(dropna=True).iloc[0]

        X_num = X.copy()
        for c in NUMERIC_RAW:
            X_num[c] = X_num[c].fillna(self.numeric_medians_[c])
        term_safe = X_num['Tenor_Pinjaman'].replace(0, np.nan)
        emi_tmp = X_num['Jumlah_Pinjaman'] / term_safe
        self.emi_median_ = emi_tmp.median()
        return self

    def transform(self, X):
        X = X.copy()

        for c in NUMERIC_RAW:
            X[c] = X[c].fillna(self.numeric_medians_[c])
        for c in CATEGORICAL_RAW:
            X[c] = X[c].fillna(self.categorical_modes_[c])
        X['Jumlah_Tanggungan'] = X['Jumlah_Tanggungan'].fillna(self.dependents_mode_)
        X['Jumlah_Tanggungan'] = (
            X['Jumlah_Tanggungan'].astype(str).map(DEPENDENTS_MAP).fillna(0).astype(int)
        )

        X['TotalIncome'] = X['Pendapatan_Pemohon'] + X['Pendapatan_Pendamping']
        term_safe = X['Tenor_Pinjaman'].replace(0, np.nan)
        X['EMI'] = X['Jumlah_Pinjaman'] / term_safe
        X['EMI'] = X['EMI'].fillna(self.emi_median_)
        X['LoanIncomeRatio'] = X['Jumlah_Pinjaman'] / (X['TotalIncome'] + 1)

        X['Pendapatan_Pemohon_Log'] = np.log1p(X['Pendapatan_Pemohon'])
        X['Pendapatan_Pendamping_Log'] = np.log1p(X['Pendapatan_Pendamping'])
        X['Jumlah_Pinjaman_Log'] = np.log1p(X['Jumlah_Pinjaman'])
        X['TotalIncome_Log'] = np.log1p(X['TotalIncome'])
        X['EMI_Log'] = np.log1p(X['EMI'])

        keep = ['Jenis_Kelamin', 'Status_Pernikahan', 'Jumlah_Tanggungan', 'Pendidikan',
                'Wiraswasta', 'Riwayat_Kredit', 'Wilayah_Properti', 'Tenor_Pinjaman',
                'Pendapatan_Pemohon_Log', 'Pendapatan_Pendamping_Log', 'Jumlah_Pinjaman_Log',
                'TotalIncome_Log', 'EMI_Log', 'LoanIncomeRatio']
        return X[keep]

    def get_feature_names_out(self, input_features=None):
        return np.array(['Jenis_Kelamin', 'Status_Pernikahan', 'Jumlah_Tanggungan', 'Pendidikan',
                          'Wiraswasta', 'Riwayat_Kredit', 'Wilayah_Properti', 'Tenor_Pinjaman',
                          'Pendapatan_Pemohon_Log', 'Pendapatan_Pendamping_Log',
                          'Jumlah_Pinjaman_Log', 'TotalIncome_Log', 'EMI_Log', 'LoanIncomeRatio'])


PASSTHROUGH_COLS = ['Jumlah_Tanggungan', 'Tenor_Pinjaman', 'Pendapatan_Pemohon_Log',
                     'Pendapatan_Pendamping_Log', 'Jumlah_Pinjaman_Log',
                     'TotalIncome_Log', 'EMI_Log', 'LoanIncomeRatio']


def build_encoder() -> ColumnTransformer:
    """Encoding eksplisit & deterministik (bukan LabelEncoder acak) supaya reproducible."""
    ordinal = OrdinalEncoder(
        categories=[BINARY_CATS[c] for c in BINARY_CATS],
        handle_unknown='use_encoded_value', unknown_value=-1,
    )
    onehot = OneHotEncoder(categories=[AREA_CATS], handle_unknown='ignore', sparse_output=False)
    return ColumnTransformer(
        transformers=[
            ('ord', ordinal, list(BINARY_CATS.keys())),
            ('onehot', onehot, ['Wilayah_Properti']),
            ('num', 'passthrough', PASSTHROUGH_COLS),
        ],
        verbose_feature_names_out=False,
    )


def safe_k_neighbors(y, default: int = 5) -> int:
    """Jaga SMOTE k_neighbors tetap valid meski kelas minoritas kecil."""
    n_minority = pd.Series(y).value_counts().min()
    return max(1, min(default, n_minority - 1))


def build_full_pipeline(clf, k_neighbors: int = 5) -> ImbPipeline:
    """
    SATU pipeline utuh: imputasi -> feature engineering -> encoding -> scaling
    -> SMOTE (hanya aktif saat .fit) -> model.

    Karena semuanya jadi SATU objek, cukup simpan 1 file (.joblib) dan tidak
    mungkin ada preprocessing yang "ketinggalan" saat dipakai untuk prediksi
    di tempat lain (mis. Streamlit) -- inilah perbaikan utama dari versi lama
    yang menyimpan model & scaler terpisah lalu mengandalkan app.py menulis
    ulang logika encoding secara manual.
    """
    return ImbPipeline([
        ('clean_fe', RawCleaner()),
        ('encode', build_encoder()),
        ('scale', StandardScaler()),
        ('smote', SMOTE(random_state=42, k_neighbors=k_neighbors)),
        ('clf', clf),
    ])


def build_raw_input(jenis_kelamin, status_pernikahan, jumlah_tanggungan, pendidikan,
                     wiraswasta, pendapatan_pemohon, pendapatan_pendamping,
                     jumlah_pinjaman, tenor_pinjaman, riwayat_kredit, wilayah_properti):
    """Bangun 1-baris DataFrame mentah dari input form -- urutan kolom tidak penting
    karena pipeline men-select kolom berdasarkan NAMA, bukan posisi."""
    return pd.DataFrame([{
        'Jenis_Kelamin': jenis_kelamin,
        'Status_Pernikahan': status_pernikahan,
        'Jumlah_Tanggungan': jumlah_tanggungan,
        'Pendidikan': pendidikan,
        'Wiraswasta': wiraswasta,
        'Pendapatan_Pemohon': float(pendapatan_pemohon),
        'Pendapatan_Pendamping': float(pendapatan_pendamping),
        'Jumlah_Pinjaman': float(jumlah_pinjaman),
        'Tenor_Pinjaman': float(tenor_pinjaman),
        'Riwayat_Kredit': riwayat_kredit,
        'Wilayah_Properti': wilayah_properti,
    }])


def predict_loan(pipeline, raw_input_df):
    """Kembalikan (label, probabilitas_disetujui) dari 1 baris raw_input_df."""
    pred = int(pipeline.predict(raw_input_df)[0])
    proba = float(pipeline.predict_proba(raw_input_df)[0, 1])
    label = 'Disetujui' if pred == 1 else 'Ditolak'
    return label, proba
