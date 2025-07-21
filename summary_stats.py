import kagglehub
import pandas as pd


path = kagglehub.dataset_download("ziya07/college-student-management-dataset")
csv_file = path + "/college_student_management_data.csv"


try:
    df = pd.read_csv(csv_file)
    print("Dataset loaded successfully.\n")
except Exception as e:
    print(f"Error loading CSV file: {e}")
    exit(1)

# --- Deep Summary Statistics ---
print("\n--- Overall Summary Statistics ---")
print(df.describe(include='all'))

print("\n--- Summary Statistics by Risk Level ---")
# Grouped statistics for numeric features
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
print(df.groupby('risk_level')[numeric_cols].describe().transpose())

print("\n--- Count of Students by Risk Level ---")
print(df['risk_level'].value_counts())

print("\n--- Average GPA by Risk Level ---")
print(df.groupby('risk_level')['GPA'].mean())

print("\n--- Average Attendance Rate by Risk Level ---")
print(df.groupby('risk_level')['attendance_rate'].mean())

print("\n--- Average Assignment Submission Rate by Risk Level ---")
print(df.groupby('risk_level')['assignment_submission_rate'].mean())

# You can add more grouped stats as needed for your analysis or interview 