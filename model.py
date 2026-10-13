import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load the dataset
df = pd.read_csv("student-por.csv", sep=None, engine="python")

# 2. Select input features
features = [
    "G1",
    "G2",
    "studytime",
    "failures",
    "absences",
    "health",
    "famsup",
    "schoolsup"
]

X = df[features].copy()
y = df["G3"]

# 3. Convert text columns into numeric columns
X = pd.get_dummies(X, drop_first=True)

# 4. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Train the model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)
model.fit(X_train, y_train)

# 6. Make predictions
predictions = model.predict(X_test)

# 7. Evaluate the model
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("STUDENT PERFORMANCE ML MODEL")
print("-" * 35)
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R-squared score: {r2:.3f}")

print("\nActual and predicted grades:")
for actual, predicted in zip(y_test.iloc[:10], predictions[:10]):
    print(f"Actual: {actual:2d} | Predicted: {predicted:.2f}")

print("\nModel training completed successfully!")