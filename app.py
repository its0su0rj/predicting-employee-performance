import pandas as pd
import numpy as np
import streamlit as st
import joblib
import plotly.express as px
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt

# Load the saved model, encoder, and column names
model = joblib.load('trained_model.pkl')
encoder = joblib.load('encoder.pkl')
saved_columns = joblib.load('columns.pkl')

# Define user input features
st.sidebar.header('Input Features')
st.sidebar.markdown("""
    Please enter the details of the employee:
""")

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

# Streamlit app layout
st.title("Employee Performance Prediction")
st.markdown("""
    This application predicts the performance of employees based on various input features.
    Use the sidebar to input employee details and see the prediction results below.
""")

# Display user input
st.subheader('User Input Features')
st.write(input_df)

# Display prediction results
st.subheader('Prediction Result')
st.write(f"Predicted Performance Rating: {predictions[0]:.2f}")

# Plotting the results using Plotly
st.subheader('Performance Prediction Visualization')
fig = px.bar(x=['Predicted Rating'], y=[predictions[0]], labels={'x': 'Rating Type', 'y': 'Rating'}, title="Employee Performance Rating")
st.plotly_chart(fig)

# Additional Visualizations
st.subheader('Additional Visualizations')

# Heatmap of input features
st.write("Correlation Heatmap of Input Features:")
corr = input_df.corr()
fig_corr, ax_corr = plt.subplots()
sns.heatmap(corr, annot=True, ax=ax_corr)
st.pyplot(fig_corr)

# Distribution plot of predicted rating
st.write("Distribution Plot of Predicted Rating:")
fig_dist = ff.create_distplot([predictions], group_labels=['Predicted Rating'])
st.plotly_chart(fig_dist)

# Box plot of input features
st.write("Box Plot of Input Features:")
fig_box = px.box(input_df)
st.plotly_chart(fig_box)
