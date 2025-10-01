# streamlit_app.py (Final Corrected Version)

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
    """
    Trains a scikit-learn pipeline model and returns both the
    trained model and the list of feature columns used for training.
    """
    df_train = pd.read_csv('synthetic_cancer_dataset.csv')

    X = df_train.drop('Cancer_Type', axis=1)
    y = df_train['Cancer_Type']
    
    # <<< CHANGE START: Store column names for later use >>>
    training_columns = X.columns
    # <<< CHANGE END >>>
    
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
    
    # <<< CHANGE START: Return the column names along with the model >>>
    return model_pipeline, training_columns
    # <<< CHANGE END >>>

with st.spinner("Preparing model... This may take a moment on first load."):
    # Unpack the model and the training columns
    model, training_cols = train_model()

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
        'Alcohol': alcohol_intake,
        'BMI': bmi
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# --- Prediction Display ---
st.subheader("Prediction Result")

if st.sidebar.button("Predict"):
    
    # <<< FINAL FIX START: This block robustly creates the full feature set >>>
    # Get user input from the sidebar
    user_input_dict = input_df.iloc[0].to_dict()

    # Create a full DataFrame with all training columns and default values
    # This ensures all columns are present and in the correct order
    full_prediction_df = pd.DataFrame(columns=training_cols)
    full_prediction_df.loc[0] = 0 # Start with a row of zeros
    
    # Set default for object/categorical columns to 'No'
    for col in full_prediction_df.select_dtypes(include=['object']).columns:
        full_prediction_df[col] = 'No'

    # Update the DataFrame with the user's actual input
    for key, value in user_input_dict.items():
        if key in full_prediction_df.columns:
            full_prediction_df[key] = value
    # <<< FINAL FIX END >>>

    try:
        # Use the fully prepared DataFrame for prediction
        prediction = model.predict(full_prediction_df)
        prediction_proba = model.predict_proba(full_prediction_df)
        
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
