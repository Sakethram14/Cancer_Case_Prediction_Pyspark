# streamlit_app.py (FINAL, SIMPLIFIED, AND ROBUST VERSION)

import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Cancer Prediction Project", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv('synthetic_cancer_dataset.csv')
    return df

df = load_data()
# This is the template with all the original feature columns and their data types
X_template = df.drop('Cancer_Type', axis=1)

@st.cache_resource
def train_model():
    y = df['Cancer_Type']
    X = df.drop('Cancer_Type', axis=1)
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)])
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])
    model_pipeline.fit(X, y)
    return model_pipeline

with st.spinner("Preparing model..."):
    model = train_model()
st.success("Model is ready!")
st.markdown("---")

st.title("🔬 BDA Project: Cancer Type Prediction")
st.sidebar.header("Patient Information")

def user_input_features():
    age = st.sidebar.slider("Age", 20, 80, 45)
    gender = st.sidebar.selectbox("Gender", ("Male", "Female"))
    smoking = st.sidebar.selectbox("Smoking Status", ("Non-smoker", "Smoker"))
    genetic_risk = st.sidebar.selectbox("Genetic Risk", ("Low", "Medium", "High"))
    physical_activity = st.sidebar.slider("Physical Activity (hours/week)", 0.0, 10.0, 3.0, 0.5)
    alcohol_intake = st.sidebar.slider("Alcohol Intake (drinks/week)", 0.0, 20.0, 5.0, 1.0)
    bmi = st.sidebar.slider("BMI", 15.0, 40.0, 25.0, 0.5)
    data = {'Age': age, 'Gender': gender, 'Smoking': smoking, 'Genetic_Risk': genetic_risk,
            'Physical_Activity': physical_activity, 'Alcohol': alcohol_intake, 'BMI': bmi}
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

st.subheader("Prediction Result")

if st.sidebar.button("Predict"):
    
    # <<< FINAL, SIMPLIFIED, AND ROBUST FIX >>>
    # 1. Take the first row of the original data as a perfect dictionary template
    template_dict = X_template.iloc[0].to_dict()

    # 2. Get the user's input as a dictionary
    user_input_dict = input_df.iloc[0].to_dict()
    
    # 3. Update the template with the user's values
    template_dict.update(user_input_dict)
    
    # 4. Convert the final, complete dictionary back to a DataFrame
    # This guarantees all columns, data types, and order are correct.
    prediction_input = pd.DataFrame([template_dict])
    # <<< END FIX >>>

    try:
        prediction = model.predict(prediction_input)
        prediction_proba = model.predict_proba(prediction_input)
        
        st.success(f"**Predicted Cancer Type:** `{prediction[0]}`")
        
        st.write("### Prediction Confidence")
        proba_df = pd.DataFrame(prediction_proba, columns=model.classes_)
        st.write(proba_df.T.style.format("{:.2%}"))

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# (The expander code for visualizations is omitted for brevity but should be kept in your file)
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
