import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the dataset
file_path = 'employees_performance_dataset.csv'
data = pd.read_csv(file_path)

# Data preprocessing
data['education'].fillna(data['education'].mode()[0], inplace=True)
data['previous_year_rating'].fillna(data['previous_year_rating'].median(), inplace=True)

# Feature selection
features = data.drop(['employee_id', 'previous_year_rating'], axis=1)
target = data['previous_year_rating']

# Encoding categorical features
categorical_features = features.select_dtypes(include=['object']).columns
encoder = OneHotEncoder(sparse=False, drop='first')
encoded_features = pd.DataFrame(encoder.fit_transform(features[categorical_features]))

# Replacing categorical features with encoded features
features = features.drop(categorical_features, axis=1)
features = pd.concat([features, encoded_features], axis=1)

# Ensure all column names are strings
features.columns = features.columns.astype(str)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Model training and evaluation
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

# Redefining target variable for classification
target_classification = data['KPIs_met_more_than_80']
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(features, target_classification, test_size=0.2, random_state=42)

# Logistic Regression
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train_class, y_train_class)
predictions_class = logistic_model.predict(X_test_class)
accuracy = accuracy_score(y_test_class, predictions_class)

# Grid Search for best parameter
param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5)
grid_search.fit(X_train_class, y_train_class)
best_param = grid_search.best_params_
best_score = grid_search.best_score_
test_score = grid_search.score(X_test_class, y_test_class)

# Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_class, y_train_class)
rf_predictions = rf_model.predict(X_test_class)
rf_accuracy = accuracy_score(y_test_class, rf_predictions)

# Save the model
joblib.dump(model, 'trained_model.pkl')

# Streamlit web app
st.title("Employee Performance Prediction")

# Input features for prediction
st.sidebar.header('Input Features')

def user_input_features(data):
    department = st.sidebar.selectbox('Department', data['department'].unique())
    region = st.sidebar.selectbox('Region', data['region'].unique())
    education = st.sidebar.selectbox('Education', data['education'].unique())
    gender = st.sidebar.selectbox('Gender', data['gender'].unique())
    recruitment_channel = st.sidebar.selectbox('Recruitment Channel', data['recruitment_channel'].unique())
    no_of_trainings = st.sidebar.slider('Number of Trainings', 1, 10, 1)
    age = st.sidebar.slider('Age', 20, 60, 30)
    length_of_service = st.sidebar.slider('Length of Service', 1, 35, 5)
    KPIs_met_more_than_80 = st.sidebar.slider('KPIs Met More Than 80%', 0, 1, 0)
    awards_won = st.sidebar.slider('Awards Won', 0, 1, 0)
    avg_training_score = st.sidebar.slider('Average Training Score', 0, 100, 50)
    
    data = {
        'department': department,
        'region': region,
        'education': education,
        'gender': gender,
        'recruitment_channel': recruitment_channel,
        'no_of_trainings': no_of_trainings,
        'age': age,
        'length_of_service': length_of_service,
        'KPIs_met_more_than_80': KPIs_met_more_than_80,
        'awards_won': awards_won,
        'avg_training_score': avg_training_score
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features(data)

# Encode input features
encoded_input_features = pd.DataFrame(encoder.transform(input_df[categorical_features]))
input_df = input_df.drop(categorical_features, axis=1)
input_df = pd.concat([input_df, encoded_input_features], axis=1)

# Ensure all column names are strings
input_df.columns = input_df.columns.astype(str)

# Prediction
prediction = model.predict(input_df)

st.subheader('Prediction')
st.write(f'Predicted Previous Year Rating: {prediction[0]}')
