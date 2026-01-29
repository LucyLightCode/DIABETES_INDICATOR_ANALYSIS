import pickle
import numpy as np
import streamlit as st

# Load the saved model
loaded_model = pickle.load(open("project_model.sav", "rb"))

# Prediction function
def diabetes_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data, dtype=float)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)

    return "Non-Diabetes" if prediction[0] == 0 else "Diabetes"


def main():
    st.title("Diabetes Health Indicators App")

    # Numeric inputs (SAFE)
    HighBP = st.number_input("HighBP (0 or 1)", min_value=0, max_value=1)
    HighChol = st.number_input("HighChol (0 or 1)", min_value=0, max_value=1)
    BMI = st.number_input("BMI", min_value=0.0)
    Stroke = st.number_input("Stroke (0 or 1)", min_value=0, max_value=1)
    HeartDiseaseorAttack = st.number_input("HeartDiseaseorAttack (0 or 1)", min_value=0, max_value=1)
    PhysActivity = st.number_input("PhysActivity (0 or 1)", min_value=0, max_value=1)
    HvyAlcoholConsump = st.number_input("HvyAlcoholConsump (0 or 1)", min_value=0, max_value=1)
    AnyHealthcare = st.number_input("AnyHealthcare (0 or 1)", min_value=0, max_value=1)
    GenHlth = st.number_input("GenHlth (1-5)", min_value=1, max_value=5)
    DiffWalk = st.number_input("DiffWalk (0 or 1)", min_value=0, max_value=1)
    Sex = st.number_input("Sex (0 = Female, 1 = Male)", min_value=0, max_value=1)
    Age = st.number_input("Age (1-13)", min_value=1, max_value=13)

    if st.button("Diabetes Result"):
        result = diabetes_prediction([
            HighBP, HighChol, BMI, Stroke, HeartDiseaseorAttack,
            PhysActivity, HvyAlcoholConsump, AnyHealthcare,
            GenHlth, DiffWalk, Sex, Age
        ])
        st.success(result)


if __name__ == "__main__":
    main()
