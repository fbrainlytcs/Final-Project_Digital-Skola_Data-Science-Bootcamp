import os
import sys
import types
import numpy as np
import pandas as pd
import streamlit as st

from custom_transformers import DropColumns
import joblib

attribute_info = """
**Input Features for Porter Delivery Processing Time Predictor**
- Restaurant Category (`store_primary_category`) : 74 kategori restoran (dropdown dengan search).
- Total Items (`total_items`) : minimal 1.
- Subtotal Price (`subtotal`) : minimal 1.
- Distinct Items (`num_distinct_items`) : minimal 1.
- Protocol Delivery (`order_protocol_num`) : ['1.0' ... '7.0'] → disimpan numerik.
- Total Onshift Partners (`total_onshift_partners`) : minimal 1.
- Total Busy Partners (`total_busy_partners`) : minimal 1.
- Total Outstanding Orders (`total_outstanding_orders`) : minimal 1.
- Min Item Price (`min_item_price`) : minimal 1.
- Max Item Price (`max_item_price`) : minimal 1.

**Fitur turunan otomatis**
- avg_item_price = subtotal / total_items
- price_spread  = max_item_price - min_item_price
- busy_ratio    = total_busy_partners / total_onshift_partners (dibatasi 0..1)
"""

#  LIST kategori untuk selectbox 
STORE_CATEGORIES = [
    'afghan','african','alcohol','alcohol-plus-food','american','argentine','asian','barbecue','belgian','brazilian',
    'breakfast','british','bubble-tea','burger','burmese','cafe','cajun','caribbean','catering','cheese','chinese',
    'chocolate','comfort-food','convenience-store','dessert','dim-sum','ethiopian','european','fast','filipino',
    'french','gastropub','german','gluten-free','greek','hawaiian','indian','indonesian','irish','italian','japanese',
    'korean','kosher','latin-american','lebanese','malaysian','mediterranean','mexican','middle-eastern','moroccan',
    'nepalese','other','pakistani','pasta','persian','peruvian','pizza','russian','salad','sandwich','seafood',
    'singaporean','smoothie','soup','southern','spanish','steak','sushi','tapas','thai','turkish','vegan',
    'vegetarian','vietnamese'
]
PROTOCOLS = ['1.0','2.0','3.0','4.0','5.0','6.0','7.0']

#  OPTIONAL: mapping kategori -> angka (kalau model TIDAK punya preprocessor) 
CAT_TO_ID = {cat: i+1 for i, cat in enumerate(STORE_CATEGORIES)}

def load_model(model_file: str):
    try:
        return joblib.load(model_file)
    except Exception:
        return None

def build_feature_row(
    category: str, 
    total_items: int, 
    subtotal: float, 
    distinct_items: int, 
    protocol: str,
    total_onshift_partners: int,
    total_busy_partners: int,
    total_outstanding_orders: int,
    min_item_price: float,
    max_item_price: float,
    ) -> pd.DataFrame:

     # turunan aman
    avg_item_price = float(subtotal) / max(int(total_items), 1)
    price_spread = float(max_item_price) - float(min_item_price)
    # busy ratio dibatasi 0..1
    busy_ratio = float(total_busy_partners) / float(max(total_onshift_partners, 1))
    busy_ratio = max(0.0, min(busy_ratio, 1.0))

    return pd.DataFrame([{
        "store_primary_category": category,
        "total_items": int(total_items),
        "subtotal": float(subtotal),
        "num_distinct_items": int(distinct_items),
        "order_protocol_num": float(protocol),
        "total_onshift_partners": int(total_onshift_partners),
        "total_busy_partners": int(total_busy_partners),
        "total_outstanding_orders": int(total_outstanding_orders),
        "min_item_price": float(min_item_price),
        "max_item_price": float(max_item_price),
        # contoh fitur turunan yang mungkin berguna
        "avg_item_price": avg_item_price,
        "price_spread": price_spread,
        "busy_ratio": busy_ratio,
    }])

def run_prediction_app():
    st.subheader("Prediction Section")
    with st.expander("Attribute Info"):
        st.markdown(attribute_info)

    # --- Input Form ---
    st.subheader("Input Your Data")
    with st.form("porter_form", clear_on_submit=False):
        category = st.selectbox("Restaurant Category", STORE_CATEGORIES, index=STORE_CATEGORIES.index("indonesian"))
        
        col1, col2 = st.columns(2)
        with col1:
            total_items = st.number_input("Total Items", min_value=1, value=3, step=1)
            distinct_items = st.number_input("Distinct Items", min_value=1, value=2, step=1)
            total_onshift_partners = st.number_input("Total Onshift Partners", min_value=1, value=10, step=1)
            total_busy_partners = st.number_input("Total Busy Partners", min_value=1, value=4, step=1)
            total_outstanding_orders = st.number_input("Total Outstanding Orders", min_value=1, value=12, step=1)

        with col2:
            subtotal = st.number_input("Subtotal Price", min_value=1.0, value=5000.0, step=100.0)
            min_item_price = st.number_input("Min Item Price", min_value=1.0, value=1000.0, step=100.0)
            max_item_price = st.number_input("Max Item Price", min_value=1.0, value=20000.0, step=100.0)
            protocol = st.selectbox("Protocol Delivery", PROTOCOLS, index=0)
            sla = st.number_input("SLA (minutes) – for messaging", min_value=1, max_value=240, value=30, step=1)

        submitted = st.form_submit_button("Predict")
        
    if not submitted:
        return  # skip prediksi kalau belum submit
        
    # Validasi ringan
    if total_busy_partners > total_onshift_partners:
        st.warning("Busy partners melebihi onshift partners. Periksa kembali input.")
        
    if max_item_price < min_item_price:
        st.warning("Max Item Price lebih kecil dari Min Item Price. Periksa kembali input.")

    with st.expander("Your Selected Options"):
        st.write({
            "store_primary_category": category,
            "total_items": total_items,
            "subtotal": subtotal,
            "num_distinct_items": distinct_items,
            "order_protocol": protocol,
            "order_protocol": protocol,
            "total_onshift_partners": total_onshift_partners,
            "total_busy_partners": total_busy_partners,
            "total_outstanding_orders": total_outstanding_orders,
            "min_item_price": min_item_price,
            "max_item_price": max_item_price
        })

    st.subheader("Prediction Result")

    # siapkan fitur untuk model
    feats_df = build_feature_row(
        category=category,
        total_items=total_items,
        subtotal=subtotal,
        distinct_items=distinct_items,
        protocol=protocol,
        total_onshift_partners=total_onshift_partners,
        total_busy_partners=total_busy_partners,
        total_outstanding_orders=total_outstanding_orders,
        min_item_price=min_item_price,
        max_item_price=max_item_price,
    )

    # --- Load model (boleh pipeline yang sudah punya encoder di dalam) ---
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "processing_time_model.pkl")
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model tidak ditemukan di: {MODEL_PATH}")
        st.stop()
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"❌ Gagal load model: {e}")
        st.stop()   

    # --- Prediksi ---
    try:
        pred_time = float(model.predict(feats_df)[0])         # coba langsung (pipeline lengkap)
    except Exception as e1:
        st.warning(f"Model gagal menerima input raw: {e1}")
    try:
        feats_df_enc = feats_df.copy()
        feats_df_enc["store_primary_category"] = (
            feats_df_enc["store_primary_category"].map(CAT_TO_ID).fillna(0).astype(int)
        )
        pred_time = float(model.predict(feats_df_enc)[0]) # coba versi encoded numerik
    except Exception as e2:
        st.error("❌ Model tetap gagal dipakai walau sudah di-encode.")
        with st.expander("Detail error"):
            st.code(f"raw predict error: {e1}")
            st.code(f"encoded predict error: {e2}")
        with st.expander("feats_df preview"):
            st.dataframe(feats_df)
            st.write("dtypes:", feats_df.dtypes.astype(str).to_dict())
        st.stop()

    pred_time = round(pred_time, 1)
    st.caption("✅ Prediksi menggunakan model .pkl.")

    # --- Messaging berbasis SLA ---
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Predicted Processing Time", f"{pred_time} min")
    with c2:
        st.metric("SLA Threshold", f"{sla} min")
    with c3:
        st.metric("Gap vs SLA", f"{(sla - pred_time):+0.1f} min")

    if pred_time <= sla:
        st.success(f"Akan sampai cepat dalam waktu sekitar **{pred_time} menit** 🚀")
        st.caption("Prediksi berada **dalam** batas SLA.")
    else:
        st.warning(f"Butuh waktu lebih lama, sekitar **{pred_time} menit** ⏱️")
        st.caption("Prediksi **melampaui** batas SLA—pertimbangkan penyesuaian kapasitas / prioritas.")

    with st.expander("Features Sent to Model"):
        st.dataframe(feats_df)
