from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.h5"
GENDER_ENCODER_PATH = BASE_DIR / "label_encoder_gender.pkl"
GEO_ENCODER_PATH = BASE_DIR / "onehot_encoder_geo.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"


@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(MODEL_PATH)

    with open(GENDER_ENCODER_PATH, "rb") as file:
        gender_encoder = pickle.load(file)
    with open(GEO_ENCODER_PATH, "rb") as file:
        geo_encoder = pickle.load(file)
    with open(SCALER_PATH, "rb") as file:
        scaler = pickle.load(file)

    return model, gender_encoder, geo_encoder, scaler


def get_model_feature_order(scaler, geography_columns):
    if hasattr(scaler, "feature_names_in_"):
        return list(scaler.feature_names_in_)

    return [
        "CreditScore",
        "Gender",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
        *list(geography_columns),
    ]


def transform_input(raw_df, gender_encoder, geo_encoder, scaler):
    df = raw_df.copy()
    df["Gender"] = gender_encoder.transform(df["Gender"])

    geo_encoded = geo_encoder.transform(df[["Geography"]]).toarray()
    geo_columns = geo_encoder.get_feature_names_out(["Geography"])
    geo_df = pd.DataFrame(geo_encoded, columns=geo_columns, index=df.index)

    df = pd.concat([df.drop(columns=["Geography"]), geo_df], axis=1)
    df = df.reindex(columns=get_model_feature_order(scaler, geo_columns), fill_value=0)

    scaled = scaler.transform(df)
    return np.asarray(scaled, dtype=np.float32)


st.set_page_config(page_title="Churn App", page_icon="📉", layout="wide")

# Native Streamlit title so it stays visible even if custom CSS/HTML is blocked.
st.title("Customer Churn Prediction")
st.caption("Enter customer details and get an instant churn estimate.")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
        color: #f3f4f6;
    }

    .stApp {
        background: radial-gradient(circle at 15% 20%, #1b2430 0%, #111827 45%, #0b1020 100%);
    }

    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #ffffff;
        padding: 20px 24px;
        border-radius: 14px;
        margin-bottom: 16px;
        border: 1px solid #334155;
    }

    .hero h1 {
        margin: 0;
        font-size: 28px;
        font-weight: 700;
    }

    .hero p {
        margin: 6px 0 0;
        opacity: 0.9;
    }

    .result-card {
        background: #111827;
        border: 1px solid #374151;
        border-radius: 14px;
        padding: 14px 16px;
        box-shadow: 0 8px 18px rgba(2, 6, 23, 0.45);
    }

    .result-label {
        color: #9ca3af;
        font-size: 13px;
        margin-bottom: 4px;
    }

    .result-value {
        color: #f8fafc;
        font-size: 24px;
        font-weight: 700;
    }

    /* Inputs and containers in dark palette */
    .stNumberInput label, .stSelectbox label {
        color: #d1d5db !important;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background-color: #0f172a !important;
        border-color: #334155 !important;
        color: #f3f4f6 !important;
    }

    .stTextInput input, .stNumberInput input {
        color: #f3f4f6 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>Churn Dashboard</h1>
        <p>Simple prediction form with instant results.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    model, gender_encoder, geo_encoder, scaler = load_artifacts()
except Exception as exc:
    st.error(f"Could not load files: {exc}")
    st.stop()

geography_options = list(geo_encoder.categories_[0])
gender_options = list(gender_encoder.classes_)

left_col, right_col = st.columns(2)

with left_col:
    credit_score = st.number_input("Credit Score", value=650)
    geography = st.selectbox("Geography", geography_options)
    gender = st.selectbox("Gender", gender_options)
    age = st.number_input("Age", value=35)
    tenure = st.number_input("Tenure", value=5)

with right_col:
    balance = st.number_input("Balance", value=60000.0)
    products = st.number_input("Number of Products", value=2)
    has_card = st.number_input("Has Credit Card (0 or 1)", value=1)
    active_member = st.number_input("Is Active Member (0 or 1)", value=1)
    salary = st.number_input("Estimated Salary", value=50000.0)

if st.button("Predict", use_container_width=True):
    try:
        input_df = pd.DataFrame(
            {
                "CreditScore": [credit_score],
                "Geography": [geography],
                "Gender": [gender],
                "Age": [age],
                "Tenure": [tenure],
                "Balance": [balance],
                "NumOfProducts": [products],
                "HasCrCard": [has_card],
                "IsActiveMember": [active_member],
                "EstimatedSalary": [salary],
            }
        )

        model_input = transform_input(input_df, gender_encoder, geo_encoder, scaler)
        churn_probability = float(model.predict(model_input, verbose=0)[0][0])
        decision = "Churn" if churn_probability >= 0.5 else "No Churn"

        metric_1, metric_2 = st.columns(2)
        with metric_1:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">Churn Probability</div>
                    <div class="result-value">{churn_probability:.2%}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with metric_2:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">Prediction</div>
                    <div class="result-value">{decision}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.progress(churn_probability)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
