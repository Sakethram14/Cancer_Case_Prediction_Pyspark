# streamlit_app.py (Corrected Version)

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

# --- Data Loading ---
@st.cache_data
def load_data():
    """Load the synthetic cancer dataset."""
    df = pd.read_csv('synthetic_cancer_dataset.csv')
    return df

df = load_data()

# --- Model Training Function with Caching ---
@st.cache_resource
def train_model():
    """Trains and returns a scikit-learn pipeline model."""
    df_train = pd.read_csv('synthetic_cancer_dataset.csv')
    X = df_train.drop('Cancer_Type', axis=1)
    y = df_train['Cancer_Type']
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    model_pipeline.fit(X, y)
    return model_pipeline

with st.spinner("Training model on first run... This may take a moment."):
    model = train_model()

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
    """Create sidebar widgets for user input."""
    age = st.sidebar.slider("Age", 20, 80, 45)
    gender = st.sidebar.selectbox("Gender", ("Male", "Female"))
    smoking = st.sidebar.selectbox("Smoking Status", ("Non-smoker", "Smoker"))
    genetic_risk = st.sidebar.selectbox("Genetic Risk", ("Low", "Medium", "High"))
    physical_activity = st.sidebar.slider("Physical Activity (hours/week)", 0.0, 10.0, 3.0, 0.5)
    alcohol_intake = st.sidebar.slider("Alcohol Intake (drinks/week)", 0.0, 20.0, 5.0, 1.0)
    bmi = st.sidebar.slider("BMI", 15.0, 40.0, 25.0, 0.5)

    data = {
        'Age': age,
        'Gender': gender,
        'Smoking': smoking,
        'Genetic_Risk': genetic_risk,
        'Physical_Activity': physical_activity,
        'Alcohol': alcohol_intake, # <<< CHANGE: Renamed 'Alcohol_Intake' to 'Alcohol'
        'BMI': bmi
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()


# --- Prediction Display ---
st.subheader("Prediction Result")

if st.sidebar.button("Predict"):

    # <<< CHANGE START: Add missing columns with default values >>>
    # Get the list of columns the model was trained on
    training_columns = model.named_steps['preprocessor'].transformers_[1][2].tolist() + model.named_steps['preprocessor'].transformers_[0][2].tolist()

    # Create a new DataFrame with default values for all training columns
    prediction_df = pd.DataFrame(columns=training_columns)
    prediction_df = pd.concat([prediction_df, input_df], ignore_index=True)

    # Fill missing columns with default neutral values ('No' for symptoms, 0 for others)
    # This ensures the model receives all expected features
    for col in training_columns:
        if col not in prediction_df.columns:
            # Assuming symptom-like columns are categorical and can be defaulted to 'No'
            if prediction_df[col].dtype == 'object' or col in ['Family_History', 'Abnormal_Bleeding', 'Cough', 'Shortness_of_Breath', 'Mouth_Pain', 'Fever', 'Weight_Loss', 'Ulcers', 'Fatigue', 'Lump_in_Breast', 'Loss_of_Appetite', 'Chest_Pain', 'Easy_Bruising', 'Night_Sweats']:
                 prediction_df[col] = 'No'
            else:
                 prediction_df[col] = 0

    # Reorder columns to match the training order
    prediction_df = prediction_df[training_columns]
    # <<< CHANGE END >>>

    try:
        prediction = model.predict(prediction_df)
        prediction_proba = model.predict_proba(prediction_df)
        
        st.success(f"**Predicted Cancer Type:** `{prediction[0]}`")
        
        st.write("### Prediction Confidence")
        proba_df = pd.DataFrame(prediction_proba, columns=model.classes_)
        st.write(proba_df.T.style.format("{:.2%}"))

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

st.sidebar.markdown("---")
st.sidebar.info("This is a demonstration application. The data is synthetic and not for medical use.")

# --- Data Visualizations ---
st.markdown("---")
with st.expander("📊 Dataset Visualizations"):
    
    st.subheader("Distribution of Patient Age")
    fig1, ax1 = plt.subplots()
    sns.histplot(df['Age'], kde=True, bins=30, ax=ax1)
    st.pyplot(fig1)

    st.subheader("Smoking Status Across Cancer Types")
    smoking_data = df.groupby(['Cancer_Type', 'Smoking']).size().reset_index(name='count')
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.barplot(y='Cancer_Type', x='count', hue='Smoking', data=smoking_data, palette='viridis', ax=ax2)
    st.pyplot(fig2)
