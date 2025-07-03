# --------- Prediction ----------

import streamlit as st
import pandas as pd
import joblib
import google.generativeai as genai
import os
from dotenv import load_dotenv

# --------------------------- Gemini API Setup ---------------------------
key =os.getenv("AIzaSyCAY2rHPfIpkKRiLKY9czqSuuJ3gc0zUo4"); 
genai.configure(api_key=key)
model_gen = genai.GenerativeModel("gemini-1.5-flash")

# --------------------------- Load Trained Model ---------------------------
model = joblib.load("ckd_rf_model.pkl")

st.set_page_config(page_title="CKD Dialysis Prediction", layout="centered")
st.title("Chronic Kidney Disease Dialysis Prediction")
st.write("Enter the patient details below to predict if dialysis is needed.")

# --------------------------- User Input ---------------------------

age = st.slider("Age", 2, 90, 45)
bp = st.slider("Blood Pressure", 50, 180, 120)
sg = st.selectbox("Specific Gravity", [1.005, 1.010, 1.015, 1.020, 1.025])
al = st.slider("Albumin", 0, 5, 0)
su = st.slider("Sugar", 0, 5, 0)
rbc = st.selectbox("Red Blood Cells", ['normal', 'abnormal'])
pc = st.selectbox("Pus Cell", ['normal', 'abnormal'])
bgr = st.slider("Blood Glucose Random", 22, 490, 150)
bu = st.slider("Blood Urea", 1, 400, 50)
sc = st.slider("Serum Creatinine", 0.4, 76.0, 1.2)
sod = st.slider("Sodium", 111, 163, 140)
pot = st.slider("Potassium", 2.5, 47.0, 4.5)
hemo = st.slider("Hemoglobin", 3.1, 17.8, 12.0)

# Categorical to numeric encoding
rbc_val = 0 if rbc == 'normal' else 1
pc_val = 0 if pc == 'normal' else 1

input_data = pd.DataFrame([{
    'age': age, 'bp': bp, 'sg': sg, 'al': al, 'su': su,
    'rbc': rbc_val, 'pc': pc_val, 'bgr': bgr, 'bu': bu,
    'sc': sc, 'sod': sod, 'pot': pot, 'hemo': hemo
}])

# --------------------------- Prediction ---------------------------
if st.button("Predict Dialysis Need"):
    prediction = model.predict(input_data)[0]
    result_text = "Dialysis Required" if prediction == 1 else "Dialysis Not Required"
    st.subheader(f"Prediction: {result_text}")

    # --------------------------- Gemini Explanation ---------------------------
    prompt = f"""
    Input data:
    {input_data.to_dict(orient='records')[0]}

    Explain in simple terms whether this patient needs dialysis.
    """
    try:
        gemini_response = model_gen.generate_content(prompt)
        st.markdown("### Gemini Explanation:")
        st.write(gemini_response.text)
    except Exception as e:
        st.error(f"Gemini API Error: {e}")
