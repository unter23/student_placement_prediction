import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("model/model.pkl", "rb"))

st.set_page_config(page_title="Placement Predictor", layout="centered")

st.title("🎓 Student Placement Predictor")

st.write("Enter student details:")

# Inputs
cgpa = st.slider("CGPA", 0.0, 10.0, 7.0)
tenth = st.slider("10th Percentage", 0, 100, 80)
twelfth = st.slider("12th Percentage", 0, 100, 75)
internships = st.slider("Internships", 0, 5, 1)
projects = st.slider("Projects", 0, 5, 2)

# Prediction
if st.button("Predict"):
    features = np.array([[cgpa, tenth, twelfth, internships, projects]])
    result = model.predict(features)

    if result[0] == 1:
        st.success("✅ High chances of getting placed!")
    else:
        st.error("❌ Low chances of placement.")
