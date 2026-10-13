import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓")

st.title("🎓 Student Performance Prediction")
st.write("An AI project supporting SDG 4: Quality Education")

@st.cache_resource
def train_model():
    df = pd.read_csv("student-por.csv", sep=None, engine="python")
    features = [
        "G1", "G2", "studytime",
        "failures", "absences", "health"
    ]
    X = df[features]
    y = df["G3"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200, random_state=42
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)

    return model, mae

model, mae = train_model()

st.success("AI model is ready!")

st.subheader("Enter Student Details")

g1 = st.slider("First period grade (G1)", 0, 20, 10)
g2 = st.slider("Second period grade (G2)", 0, 20, 10)
studytime = st.selectbox("Weekly study time", [1, 2, 3, 4], index=1)
failures = st.selectbox("Past class failures", [0, 1, 2, 3], index=0)
absences = st.number_input("School absences", min_value=0, max_value=100, value=0)
health = st.slider("Self-reported health (1–5)", 1, 5, 3)

if st.button("Predict Final Grade"):
    student = pd.DataFrame([{
        "G1": g1,
        "G2": g2,
        "studytime": studytime,
        "failures": failures,
        "absences": absences,
        "health": health
    }])

    prediction = model.predict(student)[0]
    st.metric("Estimated final grade (G3)", f"{prediction:.1f} / 20")
    st.caption("This is an estimate, not a guaranteed result.")

st.divider()
st.subheader("Model Information")
st.write(f"Mean Absolute Error on test data: {mae:.2f} grade points")
st.write("Dataset: Student Portuguese language performance")
st.write("Goal: Support SDG 4 — Quality Education")