import streamlit as st
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Forensic Disability AI", layout="wide")
st.title("⚖️ Forensic Medicine Disability Period Estimator")
st.write("Select the patient's clinical findings to estimate temporary disability days and view the medico-legal SHAP justification.")

# 1. LOAD YOUR SAVED AI MODELS
@st.cache_resource
def load_models():
    reg = xgb.XGBRegressor()
    reg.load_model("days_model.json")
    clf = xgb.XGBClassifier()
    clf.load_model("bracket_model.json")
    return reg, clf

days_model, bracket_model = load_models()

# 2. CREATE DROPDOWN MENUS FOR YOUR 10 CLINICAL COLUMNS
col1, col2 = st.columns(2)

with col1:
    st.subheader("Patient & Trauma Profile")
    age = st.number_input("Age (Years)", min_value=1, max_value=100, value=32)
    sex = st.selectbox("Sex", options=[(0, "Female"), (1, "Male")], format_func=lambda x: x[1])[0]
    occ = st.selectbox("Occupational Load", options=[(1, "1 - Sedentary / Office"), (2, "2 - Light / Standing"), (3, "3 - Heavy / Manual")], index=2, format_func=lambda x: x[1])[0]
    dm = st.selectbox("Diabetes Mellitus (Comorbidity_DM)", options=[(0, "0 - No"), (1, "1 - Yes")], format_func=lambda x: x[1])[0]
    mech = st.selectbox("Mechanism of Injury", options=[(1, "1 - Blunt"), (2, "2 - Sharp"), (3, "3 - Traffic (RTA)"), (4, "4 - Fall"), (5, "5 - Burn")], format_func=lambda x: x[1])[0]

with col2:
    st.subheader("Anatomical & Clinical Severity")
    region = st.selectbox("Primary Anatomical Region", options=[(1, "1 - Head / Face"), (2, "2 - Neck / Spine"), (3, "3 - Chest"), (4, "4 - Abdomen"), (5, "5 - Upper Limb"), (6, "6 - Lower Limb")], format_func=lambda x: x[1])[0]
    tissue = st.selectbox("Soft Tissue Depth", options=[(0, "0 - None"), (1, "1 - Abrasion / Contusion"), (2, "2 - Superficial Laceration"), (3, "3 - Deep / Tendon / Nerve")], index=2, format_func=lambda x: x[1])[0]
    fracture = st.selectbox("Fracture Severity", options=[(0, "0 - None"), (1, "1 - Simple / Non-displaced"), (2, "2 - Closed Displaced"), (3, "3 - Open / Comminuted")], index=2, format_func=lambda x: x[1])[0]
    interv = st.selectbox("Highest Medical Intervention", options=[(0, "0 - Observation / Dressing"), (1, "1 - Suturing / Splint"), (2, "2 - Closed Reduction"), (3, "3 - Major Surgery (ORIF)")], index=2, format_func=lambda x: x[1])[0]
    comp = st.selectbox("Documented Complication", options=[(0, "0 - None"), (1, "1 - Infection / Delayed Healing")], format_func=lambda x: x[1])[0]

# 3. RUN PREDICTION & EXPLANATION WHEN BUTTON IS CLICKED
if st.button("Calculate Disability Period & Generate Justification", type="primary"):
    patient_df = pd.DataFrame([{
        "Age": age, "Sex": sex, "Occupational_Load": occ, "Comorbidity_DM": dm,
        "Mechanism": mech, "Primary_Region": region, "Soft_Tissue_Depth": tissue,
        "Fracture_Severity": fracture, "Highest_Intervention": interv, "Complication": comp
    }])

    pred_days = days_model.predict(patient_df)[0]
    pred_bracket = bracket_model.predict(patient_df)[0] + 1
    bracket_names = {1: "Bracket 1 (<10 Days)", 2: "Bracket 2 (10–20 Days)", 3: "Bracket 3 (>20 Days)"}

    st.divider()
    r1, r2 = st.columns(2)
    r1.metric("Estimated Temporary Disability", f"{pred_days:.1f} Days")
    r2.metric("Predicted Penal Code Bracket", bracket_names.get(pred_bracket, f"Bracket {pred_bracket}"))

    # 4. DRAW THE SHAP EXPLANATION ON THE WEBPAGE
    st.subheader("📊 Medico-Legal Justification (Why the AI gave this estimate)")
    explainer = shap.TreeExplainer(days_model)
    shap_values = explainer(patient_df)

    fig, ax = plt.subplots(figsize=(9, 5))
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    st.pyplot(fig)