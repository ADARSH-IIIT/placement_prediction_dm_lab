import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="Campus Placement Predictor",
    page_icon="🎓",
    layout="wide"
)

# --- Model Loading ---
@st.cache_resource
def load_model():
    """Loads the pre-trained model from the .joblib file."""
    try:
        model = joblib.load('placement_xgb_model.joblib')
        return model
    except FileNotFoundError:
        st.error("Model file 'placement_rf_model.joblib' not found. Please place it in the same directory as app.py.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# --- Page Title and Description ---
st.title("🎓 Campus Placement Predictor")
st.markdown("""
This app predicts whether a student will be placed in a campus recruitment drive.
Fill in the student's details in the sidebar to get a prediction.
""")

# --- Sidebar for User Inputs ---
st.sidebar.header("Enter Student Details:")

# Get the unique values from the dataset for dropdowns
# (These are hardcoded from the dataset for simplicity)
gender_options = ['M', 'F']
ssc_b_options = ['Others', 'Central']
hsc_b_options = ['Others', 'Central']
hsc_s_options = ['Commerce', 'Science', 'Arts']
degree_t_options = ['Sci&Tech', 'Comm&Mgmt', 'Others']
workex_options = ['No', 'Yes']
spec_options = ['Mkt&HR', 'Mkt&Fin']

# --- Input Fields ---
gender = st.sidebar.selectbox("Gender", gender_options)
ssc_p = st.sidebar.slider("Secondary Education Percentage (SSC %)", 30.0, 100.0, 65.0, 0.1)
ssc_b = st.sidebar.selectbox("Board of Secondary Education", ssc_b_options)

hsc_p = st.sidebar.slider("Higher Secondary Education Percentage (HSC %)", 30.0, 100.0, 65.0, 0.1)
hsc_b = st.sidebar.selectbox("Board of Higher Secondary Education", hsc_b_options)
hsc_s = st.sidebar.selectbox("Stream of Higher Secondary Education", hsc_s_options)

degree_p = st.sidebar.slider("Degree Percentage", 30.0, 100.0, 60.0, 0.1)
degree_t = st.sidebar.selectbox("Degree Type", degree_t_options)

workex = st.sidebar.selectbox("Work Experience", workex_options)
etest_p = st.sidebar.slider("Employability Test Percentage (etest_p)", 30.0, 100.0, 70.0, 0.1)

specialisation = st.sidebar.selectbox("MBA Specialisation", spec_options)
mba_p = st.sidebar.slider("MBA Percentage", 30.0, 100.0, 60.0, 0.1)


# --- Prediction Logic ---
if st.sidebar.button("Predict Placement") and model is not None:
    # 1. Create a DataFrame from the inputs
    # The column names MUST match the 'features' list from train.py
    input_data = {
        'gender': [gender],
        'ssc_p': [ssc_p],
        'ssc_b': [ssc_b],
        'hsc_p': [hsc_p],
        'hsc_b': [hsc_b],
        'hsc_s': [hsc_s],
        'degree_p': [degree_p],
        'degree_t': [degree_t],
        'workex': [workex],
        'etest_p': [etest_p],
        'specialisation': [specialisation],
        'mba_p': [mba_p]
    }
    input_df = pd.DataFrame(input_data)
    
    # 2. Make prediction
    try:
        prediction = model.predict(input_df) # Predicts 0 or 1
        prediction_proba = model.predict_proba(input_df) # Gets probabilities
        
        # 3. Display the results
        st.header("Prediction Result")
        
        # Probability of 'Placed' (which is class 1)
        prob_placed = prediction_proba[0][1] * 100
        
        if prediction[0] == 1:
            st.success(f"**Prediction: PLACED** (Probability: {prob_placed:.2f}%)")
            st.image("https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExMXNnd21jaXJ6bWhyZzF1cXZsdXR1cnNtdjRpbDhzcDRtZGRlZDR0YSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/kyLYXonQYYfwYDIeZl/giphy.gif", caption="Congratulations!")
        else:
            st.error(f"**Prediction: NOT PLACED** (Probability of Placement: {prob_placed:.2f}%)")
            st.image("https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExM3ZmemF1ZXV2MmR1bTZ2b3dwYW1wYjVwOWo3dHpweG5pemRodG40ZCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3o7TKSxdQJIoiRXHl6/giphy.gif", caption="Keep working hard!")

        with st.expander("See Prediction Details"):
            st.write(f"**Model's Prediction:** {'Placed' if prediction[0] == 1 else 'Not Placed'}")
            st.write(f"**Confidence (Placed):** `{prob_placed:.2f}%`")
            st.write(f"**Confidence (Not Placed):** `{prediction_proba[0][0] * 100:.2f}%`")
            st.subheader("Input Data Provided:")
            st.dataframe(input_df.style.format({
                "ssc_p": "{:.1f}%",
                "hsc_p": "{:.1f}%",
                "degree_p": "{:.1f}%",
                "etest_p": "{:.1f}%",
                "mba_p": "{:.1f}%"
            }))

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.error("Please check your input values.")

elif model is None:
    st.info("The application is ready. Please add the 'placement_rf_model.joblib' file to the app's directory to enable predictions.")
