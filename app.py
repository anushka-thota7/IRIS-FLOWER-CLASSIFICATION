# 🌸 Iris Flower Classification Streamlit App
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- Streamlit Page Setup ---
st.set_page_config(page_title="🌸 Iris Flower Classifier", page_icon="🌺", layout="centered")

# --- Custom CSS Styling ---
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(135deg, #fff0f5 0%, #f3f9ff 100%);
            font-family: 'Poppins', sans-serif;
        }
        h1 {
            text-align: center;
            color: #6a0dad;
            font-weight: 800;
            margin-bottom: 10px;
        }
        .result-box {
            background-color: #ffffffcc;
            padding: 25px;
            border-radius: 20px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            color: #2E8B57;
        }
        .flower-img {
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 280px;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        [data-testid="stSidebar"] {
            background-color: #f3e6ff;
        }
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- Load Saved Model, Scaler, Encoder ---
model = joblib.load("iris_best_model.pkl")
scaler = joblib.load("iris_scaler.pkl")
le = joblib.load("iris_label_encoder.pkl")

# --- App Header ---
st.title("🌸 Iris Flower Classification App")
st.write("Adjust the sliders below and click **Predict** to identify the Iris flower species 🌺")

# --- Sidebar Inputs ---
st.sidebar.header("📏 Input Flower Measurements")
sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.8)
sepal_width  = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
petal_width  = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 1.2)

# --- Create Input DataFrame ---
input_data = pd.DataFrame({
    "sepal_length": [sepal_length],
    "sepal_width": [sepal_width],
    "petal_length": [petal_length],
    "petal_width": [petal_width]
})

# --- Prediction Button ---
if st.button("🔮 Predict Flower"):
    # Scale and predict
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)
    predicted_species = le.inverse_transform(prediction)[0]

    # Get class probabilities
    probs = model.predict_proba(scaled_input)[0]
    prob_df = pd.DataFrame({
        "Species": le.inverse_transform(np.arange(len(probs))),
        "Probability": probs
    }).sort_values(by="Probability", ascending=False)

    # 🌸 Emoji & Image Mapping
    flower_emojis = {
        "Iris-setosa": "🪷",
        "Iris-versicolor": "🌹",
        "Iris-virginica": "🌻"
    }
    flower_images = {
        "Iris-setosa": "https://upload.wikimedia.org/wikipedia/commons/a/a7/Irissetosa1.jpg",
        "Iris-versicolor": "https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg",
        "Iris-virginica": "https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg"
    }

    emoji = flower_emojis.get(predicted_species, "🌸")
    img_url = flower_images.get(predicted_species, "")

    # --- Display Results ---
    st.subheader("🌿 Your Input")
    st.dataframe(input_data, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<div class='result-box'>{emoji} Predicted Species:<br>{predicted_species}</div>", unsafe_allow_html=True)

    if img_url:
        st.markdown(f"<br><img src='{img_url}' class='flower-img'>", unsafe_allow_html=True)

    # --- Confidence Bar Chart ---
    st.subheader("📊 Model Confidence")
    fig, ax = plt.subplots(figsize=(6,3))
    sns.barplot(x="Probability", y="Species", data=prob_df, palette="crest", ax=ax)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Confidence Level")
    ax.set_ylabel("")
    st.pyplot(fig)

# --- Footer ---
st.markdown("<br><hr>", unsafe_allow_html=True)
st.caption("Made with 💖 by Anushka • Powered by Streamlit & Scikit-Learn")