import streamlit as st
import numpy as np
import pandas as pd
import statistics
from sklearn.model_selection import train_test_split

# ------------------------------
# KNN Classifier from Scratch
# ------------------------------
class KNN_Classifier():

    def __init__(self, distance_metric):
        self.distance_metric = distance_metric

    def get_distance_metric(self, training_data_point, test_data_point):

        if self.distance_metric == 'euclidean':
            dist = 0
            for i in range(len(training_data_point) - 1):
                dist += (training_data_point[i] - test_data_point[i])**2

            return np.sqrt(dist)

        elif self.distance_metric == 'manhattan':
            dist = 0
            for i in range(len(training_data_point) - 1):
                dist += abs(training_data_point[i] - test_data_point[i])

            return dist

    def nearest_neighbors(self, X_train, test_data, k):

        distance_list = []

        for training_data in X_train:
            distance = self.get_distance_metric(training_data, test_data)
            distance_list.append((training_data, distance))

        distance_list.sort(key=lambda x: x[1])

        neighbors_list = []

        for j in range(k):
            neighbors_list.append(distance_list[j][0])

        return neighbors_list

    def predict(self, X_train, test_data, k):

        neighbors = self.nearest_neighbors(X_train, test_data, k)

        labels = []

        for data in neighbors:
            labels.append(data[-1])

        predicted_class = statistics.mode(labels)

        return predicted_class


# ------------------------------
# Load Dataset
# ------------------------------
df = pd.read_csv("diabetes.csv")

X = df.drop(columns='Outcome', axis=1)
Y = df['Outcome']

X = X.to_numpy()
Y = Y.to_numpy()

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    stratify=Y,
    random_state=2
)

# Add target column to training data
X_train = np.insert(X_train, 8, Y_train, axis=1)

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("🩺 Diabetes Prediction using KNN From Scratch")

st.write("Enter the patient details below:")

pregnancies = st.number_input("Pregnancies", min_value=0)
glucose = st.number_input("Glucose")
blood_pressure = st.number_input("Blood Pressure")
skin_thickness = st.number_input("Skin Thickness")
insulin = st.number_input("Insulin")
bmi = st.number_input("BMI")
diabetes_pedigree = st.number_input("Diabetes Pedigree Function")
age = st.number_input("Age")

distance_metric = st.selectbox(
    "Distance Metric",
    ["euclidean", "manhattan"]
)

k = st.slider(
    "Select K Value",
    min_value=1,
    max_value=15,
    value=5
)

if st.button("Predict"):

    input_data = np.array([
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree,
        age
    ])

    classifier = KNN_Classifier(distance_metric)

    prediction = classifier.predict(
        X_train,
        input_data,
        k
    )

    if prediction == 0:
        st.success("✅ The person is NOT diabetic.")
    else:
        st.error("⚠️ The person is diabetic.")
