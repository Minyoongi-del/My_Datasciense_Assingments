import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the model and scaler
model = joblib.load(r"D:\Logistic Regression\titanic_logistic_regression_model.pkl")
scaler = joblib.load(r"D:\Logistic Regression\titanic_scaler.pkl")

# Load the data for visualization
train_data = pd.read_csv('D:\\Logistic Regression\\Titanic_train.csv')

# Function to preprocess input data
def preprocess_input(data):
    data_scaled = scaler.transform(data)
    return data_scaled

# Streamlit UI
st.set_page_config(page_title="Titanic Survival Prediction", page_icon="🚢")
st.title("Titanic Survival Prediction 🚢")
st.markdown("This app predicts whether a passenger survived the Titanic disaster based on input features.")

# Input fields with colorful styles
st.sidebar.header("Input Features")
pclass = st.sidebar.selectbox("Passenger Class", [1, 2, 3])
age = st.sidebar.number_input("Age", min_value=0, max_value=100, value=30)
sibsp = st.sidebar.number_input("Number of Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
parch = st.sidebar.number_input("Number of Parents/Children Aboard", min_value=0, max_value=10, value=0)
fare = st.sidebar.number_input("Fare", min_value=0.0, max_value=500.0, value=10.0)
sex = st.sidebar.selectbox("Gender", ["male", "female"])
embarked = st.sidebar.selectbox("Embarked", ["Q", "S", "C"])

# Preprocessing input
sex_male = 1 if sex == "male" else 0
embarked_Q = 1 if embarked == "Q" else 0
embarked_S = 1 if embarked == "S" else 0

input_data = pd.DataFrame({
    "Pclass": [pclass],
    "Age": [age],
    "SibSp": [sibsp],
    "Parch": [parch],
    "Fare": [fare],
    "Sex_male": [sex_male],
    "Embarked_Q": [embarked_Q],
    "Embarked_S": [embarked_S]
})

# Preprocess and make prediction
if st.sidebar.button("Predict"):
    input_data_scaled = preprocess_input(input_data)
    prediction = model.predict(input_data_scaled)[0]
    
    # Show the prediction result
    if prediction == 1:
        st.success("The passenger is likely to **survive**!", icon="✅")
    else:
        st.error("The passenger is likely to **not survive**.", icon="❌")
    
    # Visualization of the prediction
    st.subheader("Prediction Visualization")
    plt.figure(figsize=(10, 5))
    plt.bar(['Survived', 'Not Survived'], [prediction, 1 - prediction], color=['#28a745', '#dc3545'])
    plt.ylabel("Probability")
    plt.title("Prediction Outcome", fontsize=16, color='#343a40')
    plt.xticks(rotation=45)
    st.pyplot(plt)

    # Additional Visualizations Based on Survival Prediction
    st.subheader("Visualizations Based on Survival Prediction")

    # 1. Survival by Gender
    plt.figure(figsize=(10, 5))
    sns.countplot(data=train_data, x='Sex', hue='Survived', palette='Set2')
    plt.title("Survival Count by Gender", fontsize=16)
    plt.xlabel("Gender")
    plt.ylabel("Count")
    st.pyplot(plt)

    # 2. Survival by Passenger Class
    plt.figure(figsize=(10, 5))
    sns.countplot(data=train_data, x='Pclass', hue='Survived', palette='Set2')
    plt.title("Survival Count by Passenger Class", fontsize=16)
    plt.xlabel("Passenger Class")
    plt.ylabel("Count")
    st.pyplot(plt)

    # 3. Age vs Survival
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=train_data, x='Survived', y='Age', palette='Set2')
    plt.title("Age Distribution by Survival", fontsize=16)
    plt.xlabel("Survived (0 = No, 1 = Yes)")
    plt.ylabel("Age")
    st.pyplot(plt)

    # 4. Fare vs Survival
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=train_data, x='Survived', y='Fare', palette='Set2')
    plt.title("Fare Distribution by Survival", fontsize=16)
    plt.xlabel("Survived (0 = No, 1 = Yes)")
    plt.ylabel("Fare")
    st.pyplot(plt)

# Custom styling for a multicolor theme
st.markdown(
    """
    <style>
    .reportview-container {
        background: linear-gradient(to right, #ff7e5f, #feb47b); /* Gradient background */
    }
    .title {
        font-size: 2.5em;
        color: #fff; /* White color for the title */
        text-align: center;
    }
    .stButton {
        background-color: #007BFF; /* Blue color for buttons */
        color: white;
    }
    .stSidebar {
        background-color: #ffe0b2; /* Light orange for the sidebar */
    }
    .stMarkdown {
        color: #2c3e50; /* Darker color for text */
    }
    </style>
    """,
    unsafe_allow_html=True
)
