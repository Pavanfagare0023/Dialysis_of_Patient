# ckd_rf_mode.py
# .....................import Lib........................

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import google.generativeai as genai

# --------------------------- Gemini Flash 1.5 Setup ---------------------------
GEMINI_API_KEY = "AIzaSyCAY2rHPfIpkKRiLKY9czqSuuJ3gc0zUo4"  #Replace with your actual key

genai.configure(api_key=GEMINI_API_KEY)
model_gen = genai.GenerativeModel("gemini-1.5-flash")

# --------------------------- Load Trained Model ---------------------------
model = joblib.load("ckd_rf_model.pkl")

st.set_page_config(page_title="CKD Dialysis Prediction", layout="centered")
st.title("Chronic Kidney Disease Dialysis Prediction")
st.write("Provide your test values below to predict dialysis requirement.")

# --------------------------- User Input ---------------------------
age = st.slider("Age", 2, 90, 45)
bp = st.slider("Blood Pressure", 50, 180, 120)
sg = st.selectbox("Specific Gravity", [1.005, 1.010, 1.015, 1.020, 1.025])
al = st.slider("Albumin", 0, 5, 0)
su = st.slider("Sugar", 0, 5, 0)
bgr = st.slider("Blood Glucose Random", 22, 490, 150)
bu = st.slider("Blood Urea", 1, 400, 50)
sc = st.slider("Serum Creatinine", 0.4, 76.0, 1.2)
sod = st.slider("Sodium", 111, 163, 140)
pot = st.slider("Potassium", 2.5, 47.0, 4.5)
hemo = st.slider("Hemoglobin", 3.1, 17.8, 12.0)

input_data = pd.DataFrame([{
    'age': age, 'bp': bp, 'sg': sg, 'al': al, 'su': su,
    'bgr': bgr, 'bu': bu, 'sc': sc, 'sod': sod, 'pot': pot, 'hemo': hemo
}])

# --------------------------- Predict Button ---------------------------
if st.button("Predict Dialysis Need"):
    prediction = model.predict(input_data)[0]
    result_text = "Dialysis Required" if prediction == 1 else "Dialysis Not Required"
    st.subheader(f"Prediction Result: {result_text}")

    # --------------------------- Gemini Explanation ---------------------------
    prompt = f"""
    Based on the following input data:
    {input_data.to_dict(orient='records')[0]}

    Explain in simple medical terms why this patient may or may not need dialysis.
    """
    try:
        gemini_response = model_gen.generate_content(prompt)
        st.markdown("###Explanation from Gemini:")
        st.write(gemini_response.text)
    except Exception as e:
        st.error("Error calling Gemini API. Please check your key or quota.")
