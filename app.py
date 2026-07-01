import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
from io import BytesIO
from gensim.models import FastText

# ─────────────────────────────────────────────
# Konfigurasi halaman
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Analisis Sentimen",
    page_icon="💬",
    layout="centered",
)

# ─────────────────────────────────────────────
# Load model (di-cache agar tidak reload terus)
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    rf_path  = "Model/rf_model.pkl"
    ft_path  = "Model/fasttext_model.model"

    if not os.path.exists(rf_path) or not os.path.exists(ft_path):
        return None, None

    rf_model       = joblib.load(rf_path)
    fasttext_model = FastText.load(ft_path)
    return rf_model, fasttext_model

# ─────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────
def preprocess_text(text: str) -> list:
    """Lowercase + tokenisasi sederhana (split whitespace)."""
    return str(text).lower().split()


def get_word_embeddings(tokens: list, model: FastText) -> np.ndarray:
    """Rata-rata vektor kata dari FastText."""
    vectors = []
    for word in tokens:
        if word in model.wv:
            vectors.append(model.wv[word])
        else:
            vectors.append(np.zeros(model.vector_size))
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)


def predict_single(text: str, rf_model, fasttext_model):
    """Prediksi satu teks, return (label_int, tokens, probabilities)."""
    tokens = preprocess_text(text)
    vector = get_word_embeddings(tokens, fasttext_model).reshape(1, -1)
    prediction    = rf_model.predict(vector)[0]
    probabilities = rf_model.predict_proba(vector)[0]
    return prediction, tokens, probabilities


def predict_batch(texts: pd.Series, rf_model, fasttext_model):
    """Prediksi banyak teks sekaligus, return DataFrame hasil."""
    labels = []
    conf_scores = []
    neg_scores = []
    pos_scores = []

    progress_bar = st.progress(0, text="Memproses data...")
    total = len(texts)

    for i, text in enumerate(texts):
        pred, _, probs = predict_single(text, rf_model, fasttext_model)
        labels.append(LABEL_MAP[pred])
        conf_scores.append(probs[pred] * 100)
        neg_scores.append(probs[0] * 100)
        pos_scores.append(probs[1] * 100)
        progress_bar.progress((i + 1) / total, text=f"Memproses {i + 1}/{total}")

    progress_bar.empty()

    return pd.DataFrame({
        "Sentimen": labels,
        "Confidence (%)": [round(c, 1) for c in conf_scores],
        "Negatif (%)": [round(c, 1) for c in neg_scores],
        "Positif (%)": [round(c, 1) for c in pos_scores],
    })


def convert_df_to_excel(df: pd.DataFrame) -> bytes:
    """Convert DataFrame ke excel bytes untuk didownload."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Hasil Prediksi")
    return output.getvalue()


LABEL_MAP = {0: "Negatif 😠", 1: "Positif 😊"}
COLOR_MAP  = {0: "#FF4B4B",   1: "#21C55D"}

# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.title("💬 Analisis Sentimen Review Aplikasi")
st.caption("Model: FastText Embedding + Random Forest (SMOTE | K-Fold)")

st.divider()

# Load model
rf_model, fasttext_model = load_models()

if rf_model is None:
    st.error(
        "⚠️ **Model tidak ditemukan.**\n\n"
        "Pastikan file berikut sudah tersedia:\n"
        "- `Model/Fasttext/SMOTE/rf_model.pkl`\n"
        "- `Model/Fasttext/SMOTE/fasttext_model.model`"
    )
    st.stop()

# ── Tabs: Single Text vs Batch Upload ──────────────────────────────
tab_single, tab_batch = st.tabs(["🔍 Single Text", "📂 Upload File (Batch)"])

# =====================================================================
# TAB 1: SINGLE TEXT
# =====================================================================
with tab_single:
    st.subheader("🔍 Prediksi Sentimen")
    user_input = st.text_area(
        label="Masukkan teks review:",
        placeholder="Contoh: aplikasinya bagus banget, koleksi musik lengkap dan tidak ada iklan...",
        height=140,
        key="single_input",
    )

    predict_btn = st.button("Analisis Sentimen", type="primary", use_container_width=True, key="single_btn")

    if predict_btn:
        text = user_input.strip()
        if not text:
            st.warning("Teks review tidak boleh kosong.")
        else:
            prediction, tokens, probabilities = predict_single(text, rf_model, fasttext_model)

            label = LABEL_MAP[prediction]
            color = COLOR_MAP[prediction]
            conf  = probabilities[prediction] * 100

            st.divider()
            st.subheader("📊 Hasil Analisis")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                    f"<div style='text-align:center; padding:20px; border-radius:12px; "
                    f"background-color:{color}22; border:2px solid {color};'>"
                    f"<h2 style='color:{color}; margin:0'>{label}</h2>"
                    f"<p style='color:gray; margin:4px 0 0'>Sentimen Terdeteksi</p>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with col2:
                st.metric("Confidence", f"{conf:.1f}%")
                neg_conf = probabilities[0] * 100
                pos_conf = probabilities[1] * 100
                st.caption(f"Negatif: {neg_conf:.1f}% | Positif: {pos_conf:.1f}%")

            # Progress bar probabilitas
            st.markdown("**Distribusi Probabilitas:**")
            st.progress(int(pos_conf), text=f"Positif {pos_conf:.1f}%")

            # Tampilkan tokens
            with st.expander("🔎 Detail Tokenisasi"):
                st.write(f"**Jumlah token:** {len(tokens)}")
                st.write(tokens)

# =====================================================================
# TAB 2: BATCH UPLOAD (CSV / EXCEL)
# =====================================================================
with tab_batch:
    st.subheader("📂 Prediksi Sentimen — Batch")
    st.caption("Upload file CSV atau Excel yang berisi kolom teks review.")

    uploaded_file = st.file_uploader(
        "Pilih file (.csv, .xlsx, .xls)",
        type=["csv", "xlsx", "xls"],
        key="batch_uploader",
    )

    if uploaded_file is not None:
        # Baca file sesuai ekstensi
        try:
            if uploaded_file.name.endswith(".csv"):
                df_batch = pd.read_csv(uploaded_file)
            else:
                df_batch = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")
            st.stop()

        if df_batch.empty:
            st.warning("File yang diupload kosong.")
        else:
            st.success(f"File berhasil dibaca — {len(df_batch)} baris data.")
            st.dataframe(df_batch.head(), use_container_width=True)

            # Pilih kolom mana yang berisi teks review
            text_column = st.selectbox(
                "Pilih kolom yang berisi teks review:",
                options=df_batch.columns.tolist(),
            )

            run_batch_btn = st.button(
                "Analisis Semua Data", type="primary", use_container_width=True, key="batch_btn"
            )

            if run_batch_btn:
                # Buang baris dengan teks kosong/NaN pada kolom terpilih
                valid_mask = df_batch[text_column].notna() & (df_batch[text_column].astype(str).str.strip() != "")
                if valid_mask.sum() == 0:
                    st.warning("Tidak ada teks valid pada kolom yang dipilih.")
                else:
                    df_valid = df_batch[valid_mask].reset_index(drop=True)
                    skipped = len(df_batch) - len(df_valid)

                    with st.spinner("Sedang memproses seluruh data..."):
                        hasil_df = predict_batch(df_valid[text_column], rf_model, fasttext_model)

                    df_result = pd.concat([df_valid.reset_index(drop=True), hasil_df], axis=1)

                    st.divider()
                    st.subheader("📊 Hasil Analisis Batch")

                    if skipped > 0:
                        st.info(f"{skipped} baris dilewati karena teks kosong.")

                    # Ringkasan distribusi sentimen
                    col1, col2, col3 = st.columns(3)
                    total_data = len(df_result)
                    total_pos  = (df_result["Sentimen"] == LABEL_MAP[1]).sum()
                    total_neg  = (df_result["Sentimen"] == LABEL_MAP[0]).sum()

                    col1.metric("Total Data", total_data)
                    col2.metric("Positif 😊", f"{total_pos} ({total_pos/total_data*100:.1f}%)")
                    col3.metric("Negatif 😠", f"{total_neg} ({total_neg/total_data*100:.1f}%)")

                    st.bar_chart(df_result["Sentimen"].value_counts())

                    st.markdown("**Tabel Hasil Prediksi:**")
                    st.dataframe(df_result, use_container_width=True)

                    # Tombol download hasil
                    col_dl1, col_dl2 = st.columns(2)
                    with col_dl1:
                        csv_data = df_result.to_csv(index=False).encode("utf-8-sig")
                        st.download_button(
                            "⬇️ Download CSV",
                            data=csv_data,
                            file_name="hasil_prediksi_sentimen.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )
                    with col_dl2:
                        excel_data = convert_df_to_excel(df_result)
                        st.download_button(
                            "⬇️ Download Excel",
                            data=excel_data,
                            file_name="hasil_prediksi_sentimen.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True,
                        )

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.divider()
st.caption("Model: FastText (sg=1, vector_size=100, window=3) + RandomForest (n_estimators=300) + SMOTE + 10-Fold CV")