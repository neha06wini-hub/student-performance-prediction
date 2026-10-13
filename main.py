import pandas as pd

# Load the student dataset
df = pd.read_csv("student-por.csv", sep=None, engine="python")

# Display project title
print("=" * 45)
print("STUDENT PERFORMANCE ANALYSIS")
print("=" * 45)

# Show the first five records
print("\nFirst 5 student records:")
print(df.head())

# Show dataset information
print("\nDataset size:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

# Check missing values
print("\nMissing values:")
print(df.isnull().sum().sum())

# Analyze final grades
print("\nAverage final grade:", round(df["G3"].mean(), 2))
print("Highest final grade:", df["G3"].max())
print("Lowest final grade:", df["G3"].min())

# Analyze study time
print("\nAverage grade by study time:")
print(df.groupby("studytime")["G3"].mean().round(2))

print("\nAnalysis completed successfully!")