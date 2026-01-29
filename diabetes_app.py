import pickle
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Load the trained model
loaded_model = pickle.load(open("project_model.sav", "rb"))

# Predict function
def diabetes_prediction(input_data):
    input_array = np.asarray(input_data, dtype=float).reshape(1, -1)
    prediction = loaded_model.predict(input_array)
    
    # If your model supports probability (recommended)
    if hasattr(loaded_model, "predict_proba"):
        proba = loaded_model.predict_proba(input_array)[0][1]  # probability of Diabetes
    else:
        proba = 1.0 if prediction[0] == 1 else 0.0
    return prediction[0], proba

# App main
def main():
    st.set_page_config(page_title="Diabetes Predictor Dashboard", page_icon="💉", layout="wide")

    # Custom CSS for styling
    st.markdown(
        """
        <style>
        .stApp {background-color: #f0f8ff;}
        h1 {color: #4CAF50; text-align:center;}
        h2 {color: #2196F3;}
        </style>
        """, unsafe_allow_html=True
    )

    # Title
    st.title("💉 Diabetes Health Indicator Dashboard")
    st.markdown("Fill in your health data to see your diabetes risk probability and metrics.")

    # Sidebar for info
    st.sidebar.image("logo.png", width=200)
    st.sidebar.title("About")
    st.sidebar.info("""
    This dashboard predicts diabetes risk using your health indicators.
    Powered by a Machine Learning model.
    """)
    
    # Input columns
    col1, col2, col3 = st.columns(3)
    with col1:
        HighBP = st.number_input("HighBP (0 or 1)", min_value=0, max_value=1)
        Stroke = st.number_input("Stroke (0 or 1)", min_value=0, max_value=1)
        HvyAlcoholConsump = st.number_input("HvyAlcoholConsump (0 or 1)", min_value=0, max_value=1)
        DiffWalk = st.number_input("DiffWalk (0 or 1)", min_value=0, max_value=1)
    with col2:
        HighChol = st.number_input("HighChol (0 or 1)", min_value=0, max_value=1)
        HeartDiseaseorAttack = st.number_input("HeartDiseaseorAttack (0 or 1)", min_value=0, max_value=1)
        AnyHealthcare = st.number_input("AnyHealthcare (0 or 1)", min_value=0, max_value=1)
        PhysActivity = st.number_input("PhysActivity (0 or 1)", min_value=0, max_value=1)
    with col3:
        BMI = st.number_input("BMI", min_value=0.0, format="%.1f")
        GenHlth = st.number_input("GenHlth (1-5)", min_value=1, max_value=5)
        Sex = st.number_input("Sex (0 = Female, 1 = Male)", min_value=0, max_value=1)
        Age = st.number_input("Age (1-13)", min_value=1, max_value=13)

    # Prediction button
    if st.button("🔎 Check Diabetes Risk"):
        pred, risk_proba = diabetes_prediction([
            HighBP, HighChol, BMI, Stroke, HeartDiseaseorAttack,
            PhysActivity, HvyAlcoholConsump, AnyHealthcare,
            GenHlth, DiffWalk, Sex, Age
        ])

        # Show big result panel
        st.markdown("---")
        st.markdown(f"<h2 style='text-align: center;'>Prediction Result</h2>", unsafe_allow_html=True)
        if pred == 1:
            st.error(f"⚠️ High Risk: You may have diabetes!")
        else:
            st.success(f"✅ Low Risk: You are likely Non-Diabetic!")

        # Progress bar for risk
        st.markdown("### Diabetes Risk Probability")
        st.progress(int(risk_proba*100))

        # Metrics dashboard
        st.markdown("### Key Health Metrics")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("BMI", BMI)
        col_b.metric("Age", Age)
        col_c.metric("GenHlth", GenHlth)

        # Optional: Visual chart of key metrics
        st.markdown("### Health Metrics Overview")
        fig, ax = plt.subplots()
        metrics = ["BMI", "GenHlth", "Age"]
        values = [BMI, GenHlth, Age]
        colors = ['#4CAF50', '#2196F3', '#FF9800']
        ax.bar(metrics, values, color=colors)
        ax.set_ylim(0, max(30, max(values)+5))
        st.pyplot(fig)

if __name__ == "__main__":
    main()
