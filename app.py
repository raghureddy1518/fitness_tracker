import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import time
import warnings

warnings.filterwarnings('ignore')

# ---- App Title ----
st.write("## 🏋️‍♂️ Personal Fitness Tracker")
st.write("""
    In this WebApp, you can predict the **calories burned** based on your parameters 
    such as `Age`, `Gender`, `BMI`, `Duration`, `Heart Rate`, and `Body Temperature`.
""")

# ---- Sidebar for User Input ----
st.sidebar.header("⚙️ **User Input Parameters:**")

def user_input_features():
    age = st.sidebar.slider("🎂 Age", 10, 100, 30)
    bmi = st.sidebar.slider("⚖️ BMI", 15, 40, 22)
    duration = st.sidebar.slider("⏱️ Duration (min)", 0, 35, 15)
    heart_rate = st.sidebar.slider("❤️ Heart Rate", 60, 130, 80)
    body_temp = st.sidebar.slider("🌡️ Body Temperature (C)", 36, 42, 38)
    gender_button = st.sidebar.radio("🚻 Gender", ("Male", "Female"))

    gender = 1 if gender_button == "Male" else 0

    # Map the input to model columns
    data_model = {
        "Age": age,
        "BMI": bmi,
        "Duration": duration,
        "Heart_Rate": heart_rate,
        "Body_Temp": body_temp,
        "Gender_male": gender  # 1 for Male, 0 for Female
    }

    features = pd.DataFrame(data_model, index=[0])
    return features

# Load user inputs
df = user_input_features()

# ---- Display User Parameters ----
st.write("---")
st.header("🔍 **Your Parameters:**")
st.write(df)

# ---- Load and Preprocess Data ----
@st.cache_data
def load_data():
    calories = pd.read_csv("calories.csv")
    exercise = pd.read_csv("exercise.csv")

    # Merge data and preprocess
    exercise_df = exercise.merge(calories, on="User_ID")
    exercise_df.drop(columns="User_ID", inplace=True)

    # Add BMI column
    for data in [exercise_df]:
        data["BMI"] = data["Weight"] / ((data["Height"] / 100) ** 2)
        data["BMI"] = round(data["BMI"], 2)

    return exercise_df

# Load data
exercise_df = load_data()

# ---- Train-Test Split ----
train_data, test_data = train_test_split(exercise_df, test_size=0.2, random_state=1)

# Preprocess data
train_data = train_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
test_data = test_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]

# One-hot encoding for Gender
train_data = pd.get_dummies(train_data, drop_first=True)
test_data = pd.get_dummies(test_data, drop_first=True)

# Split features and target
X_train = train_data.drop("Calories", axis=1)
y_train = train_data["Calories"]

X_test = test_data.drop("Calories", axis=1)
y_test = test_data["Calories"]

# ---- Model Training ----
st.write("---")
st.header("🚀 **Training the Model...**")

# Progress Bar for Training
progress_bar = st.progress(0)
for i in range(100):
    progress_bar.progress(i + 1)
    time.sleep(0.01)

# Train the model
model = RandomForestRegressor(n_estimators=1000, max_features=3, max_depth=6)
model.fit(X_train, y_train)

# Align columns with training data
df = df.reindex(columns=X_train.columns, fill_value=0)

# ---- Make Prediction ----
prediction = model.predict(df)

# ---- Display Prediction ----
st.write("---")
st.header("🔥 **Calories Burned Prediction:**")
st.success(f"{round(prediction[0], 2)} **kilocalories**")

# ---- Similar Results ----
st.write("---")
st.header("📊 **Similar Results:**")

# Find similar results based on predicted calories
calorie_range = [prediction[0] - 10, prediction[0] + 10]
similar_data = exercise_df[
    (exercise_df["Calories"] >= calorie_range[0]) & 
    (exercise_df["Calories"] <= calorie_range[1])
]

st.write(similar_data.sample(5))

# ---- General Information ----
st.write("---")
st.header("📈 **General Information:**")

# Comparison with others
age_comparison = (exercise_df["Age"] < df["Age"].values[0]).mean() * 100
duration_comparison = (exercise_df["Duration"] < df["Duration"].values[0]).mean() * 100
heart_rate_comparison = (exercise_df["Heart_Rate"] < df["Heart_Rate"].values[0]).mean() * 100
temp_comparison = (exercise_df["Body_Temp"] < df["Body_Temp"].values[0]).mean() * 100

st.write(f"🎯 You are older than **{round(age_comparison, 2)}%** of other users.")
st.write(f"⏱️ Your exercise duration is longer than **{round(duration_comparison, 2)}%** of others.")
st.write(f"❤️ Your heart rate is higher than **{round(heart_rate_comparison, 2)}%** of others.")
st.write(f"🌡️ Your body temperature is higher than **{round(temp_comparison, 2)}%** of others.")

