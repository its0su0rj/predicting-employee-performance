import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import OneHotEncoder
import joblib
import plotly.express as px
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt

# Load the saved model and encoder
model = joblib.load('trained_model.pkl')
encoder = joblib.load('encoder.pkl')

# Sidebar for user input features
st.sidebar.header('Input Features')

def user_input_features():
    department = st.sidebar.selectbox('Department', ['Sales & Marketing', 'Operations', 'Technology', 'Analytics', 'R&D', 'Procurement', 'Finance', 'HR', 'Legal'])
    region = st.sidebar.selectbox('Region', ['region_1', 'region_2', 'region_3', 'region_4', 'region_5', 'region_6', 'region_7', 'region_8', 'region_9', 'region_10', 'region_11', 'region_12', 'region_13', 'region_14', 'region_15', 'region_16', 'region_17', 'region_18', 'region_19', 'region_20', 'region_21', 'region_22', 'region_23', 'region_24', 'region_25', 'region_26', 'region_27', 'region_28', 'region_29', 'region_30', 'region_31', 'region_32', 'region_33', 'region_34'])
    education = st.sidebar.selectbox('Education', ['Bachelor\'s', 'Master\'s & above', 'Below Secondary'])
    gender = st.sidebar.selectbox('Gender', ['male', 'female'])
    recruitment_channel = st.sidebar.selectbox('Recruitment Channel', ['sourcing', 'referred', 'other'])
    no_of_trainings = st.sidebar.slider('Number of Trainings', 1, 10, 1)
    age = st.sidebar.slider('Age', 20, 60, 30)
    previous_year_rating = st.sidebar.slider('Previous Year Rating', 1.0, 5.0, 3.0)
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
        'previous_year_rating': [previous_year_rating],
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

# Make predictions
predictions = model.predict(input_df)

# Display results
st.subheader('Prediction')
st.write(predictions)

# Display user input
st.subheader('User Input Features')
st.write(input_df)

# Display distribution plot of predicted rating if there are enough predictions
if len(predictions) > 1:
    fig_dist = ff.create_distplot([predictions], group_labels=['Predicted Rating'])
    st.plotly_chart(fig_dist, use_container_width=True)
else:
    st.write("Not enough data to create a distribution plot.")

# Display a bar chart of predicted ratings
fig_bar = px.bar(x=input_df.index, y=predictions, labels={'x': 'Index', 'y': 'Predicted Rating'}, title="Bar Chart of Predicted Ratings")
st.plotly_chart(fig_bar, use_container_width=True)

# Display a heatmap of input features
fig_heatmap = px.imshow(input_df, aspect="auto", title="Heatmap of Input Features")
st.plotly_chart(fig_heatmap, use_container_width=True)

# Display pairplot using seaborn
st.subheader('Pairplot of Input Features')
pairplot_fig = sns.pairplot(input_df)
st.pyplot(pairplot_fig)

# Display correlation heatmap using seaborn
st.subheader('Correlation Heatmap of Input Features')
plt.figure(figsize=(10, 8))
corr_heatmap = sns.heatmap(input_df.corr(), annot=True, cmap='coolwarm')
st.pyplot(plt)
