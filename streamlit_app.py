# streamlit_app.py (FINAL, GUARANTEED FIX)

import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# --- Page Configuration ---
st.set_page_config(
    page_title="Cancer Prediction Project",
    page_icon="♋",
    layout="wide"
)

# --- Data and Model Loading (Cached for performance) ---
@st.cache_resource
def load_model_and_template():
    """Loads the data, trains the model, and prepares a prediction template."""
    df = pd.read_csv('synthetic_cancer_dataset.csv')
    
    # Define features (X) and target (y)
    X = df.drop('Cancer_Type', axis=1)
    y = df['Cancer_Type']
    
    # Define the preprocessing steps
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)])
    
    # Define the model pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])
    
    # Train the model
    model_pipeline.fit(X, y)
    
    # Return the trained model AND the feature dataframe (X) to be used as a template
    return model_pipeline, X, df

# Load the model, the feature template, and the full dataframe
model, X_template, full_df = load_model_and_template()

st.success("Model is ready!")
st.markdown("---")

# --- App Layout ---
st.title("🔬 BDA Project: Cancer Type Prediction")
st.markdown("""
This application predicts cancer types using a Random Forest model trained on synthetic data. 
Please provide the patient's details in the sidebar to get a prediction.
""")

# --- Sidebar for User Input ---
st.sidebar.header("Patient Information")

def user_input_features():
    """Create sidebar widgets and return a dictionary of user inputs."""
    # NOTE: The label for the slider is user-friendly, but the key in the dictionary
    # must exactly match the column name in the CSV file.
    
    age = st.sidebar.slider("Age", int(X_template['Age'].min()), int(X_template['Age'].max()), 45)
    gender = st.sidebar.selectbox("Gender", X_template['Gender'].unique())
    smoking = st.sidebar.selectbox("Smoking Status", X_template['Smoking'].unique())
    genetic_risk = st.sidebar.selectbox("Genetic Risk", X_template['Genetic_Risk'].unique())
    physical_activity = st.sidebar.slider("Physical Activity (hours/week)", 0.0, 10.0, 3.0, 0.5)
    
    # This slider will update BOTH 'Alcohol' and 'Alcohol_Intake' columns
    alcohol_intake = st.sidebar.slider("Alcohol Intake (drinks/week)", 0.0, 20.0, 5.0, 1.0)
    
    bmi = st.sidebar.slider("BMI", 15.0, 40.0, 25.0, 0.5)

    data = {
        'Age': age,
        'Gender': gender,
        'Smoking': smoking,
        'Genetic_Risk': genetic_risk,
        'Physical_Activity': physical_activity,
        'Alcohol': alcohol_intake,
        'Alcohol_Intake': alcohol_intake, # Ensuring both columns get the value
        'BMI': bmi
    }
    return data

user_input_dict = user_input_features()

# --- Prediction Display ---
st.subheader("Prediction Result")

if st.sidebar.button("Predict"):
    
    # <<< THE FINAL, SIMPLE FIX >>>
    # 1. Create a copy of the first row of the original features.
    #    This is our perfect template with all columns and correct data types.
    prediction_input = X_template.iloc[[0]].copy()

    # 2. Update the template with the user's input values from the sidebar.
    for key, value in user_input_dict.items():
        prediction_input[key] = value
    # <<< END FIX >>>

    try:
        # Use the fully prepared DataFrame for prediction
        prediction = model.predict(prediction_input)
        prediction_proba = model.predict_proba(prediction_input)
        
        st.success(f"**Predicted Cancer Type:** `{prediction[0]}`")
        
        st.write("### Prediction Confidence")
        proba_df = pd.DataFrame(prediction_proba, columns=model.classes_)
        st.write(proba_df.T.style.format("{:.2%}"))

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.error("There might be a mismatch in library versions. Please ensure your `requirements.txt` file matches the one provided.")

st.sidebar.markdown("---")
st.sidebar.info("This is a demonstration application. The data is synthetic and not for medical use.")

# --- Data Visualizations ---
st.markdown("---")
with st.expander("📊 Dataset Visualizations"):
    
    st.subheader("Distribution of Patient Age")
    fig1, ax1 = plt.subplots()
    sns.histplot(full_df['Age'], kde=True, bins=30, ax=ax1)
    st.pyplot(fig1)

    st.subheader("Smoking Status Across Cancer Types")
    smoking_data = full_df.groupby(['Cancer_Type', 'Smoking']).size().reset_index(name='count')
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.barplot(y='Cancer_Type', x='count', hue='Smoking', data=smoking_data, palette='viridis', ax=ax2)
    st.pyplot(fig2)
