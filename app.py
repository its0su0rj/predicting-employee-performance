import pandas as pd
import numpy as np
import streamlit as st
import joblib

# Load the saved model, encoder, and column names
model = joblib.load('trained_model.pkl')
encoder = joblib.load('encoder.pkl')
saved_columns = joblib.load('columns.pkl')

# Define user input features
st.sidebar.header('Input Features')

def user_input_features():
    department = st.sidebar.selectbox('Department', ['Sales', 'Operations', 'Technology', 'Analytics', 'R&D', 'Procurement', 'Finance', 'HR', 'Legal'])
    region = st.sidebar.selectbox('Region', ['region_1', 'region_2', 'region_3', 'region_4', 'region_5', 'region_6', 'region_7', 'region_8', 'region_9', 'region_10', 'region_11', 'region_12', 'region_13', 'region_14', 'region_15', 'region_16', 'region_17', 'region_18', 'region_19', 'region_20', 'region_21', 'region_22', 'region_23', 'region_24', 'region_25', 'region_26', 'region_27', 'region_28', 'region_29', 'region_30', 'region_31', 'region_32', 'region_33', 'region_34'])
    education = st.sidebar.selectbox('Education', ['Bachelor\'s', 'Master\'s & above', 'Below Secondary'])
    gender = st.sidebar.selectbox('Gender', ['male', 'female'])
    recruitment_channel = st.sidebar.selectbox('Recruitment Channel', ['sourcing', 'other', 'referred'])
    no_of_trainings = st.sidebar.slider('Number of Trainings', 1, 10, 1)
    age = st.sidebar.slider('Age', 20, 60, 30)
    length_of_service = st.sidebar.slider('Length of Service', 1, 30, 5)
    KPIs_met_more_than_80 = st.sidebar.slider('KPIs Met More Than 80%', 0, 1, 0)
    awards_won = st.sidebar.slider('Awards Won', 0, 1, 0)
    avg_training_score = st.sidebar.slider('Average Training Score', 0, 100, 50)
    
    features = pd.DataFrame({
        'department': [department],
        'region': [region],
        'education': [education],
        'gender': [gender],
        'recruitment_channel': [recruitment_channel],
        'no_of_trainings': [no_of_trainings],
        'age': [age],
        'length_of_service': [length_of_service],
        'KPIs_met_more_than_80': [KPIs_met_more_than_80],
        'awards_won': [awards_won],
        'avg_training_score': [avg_training_score]
    })
    
    return features

input_df = user_input_features()

# Encode input features
categorical_features = ['department', 'region', 'education', 'gender', 'recruitment_channel']
encoded_features = pd.DataFrame(encoder.transform(input_df[categorical_features]), columns=encoder.get_feature_names_out(categorical_features))

# Replace the categorical features with the encoded features
input_df = input_df.drop(categorical_features, axis=1)
input_df = pd.concat([input_df, encoded_features], axis=1)

# Ensure all column names are strings
input_df.columns = input_df.columns.astype(str)

# Align input_df with the saved columns
input_df = input_df.reindex(columns=saved_columns, fill_value=0)

# Make predictions
predictions = model.predict(input_df)

# Display results
st.subheader('Prediction')
st.write(predictions)

# Show user input
st.subheader('User Input features')
st.write(input_df)
