import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


path = kagglehub.dataset_download("ziya07/college-student-management-dataset")

print("Path to dataset files:", path)

# Path to the CSV file
csv_file = path + "/college_student_management_data.csv"

# Load the CSV data using pandas
try:
    df = pd.read_csv(csv_file)
    # print("\nFirst 5 rows of the dataset:")
    # print(df)
    # print("\nMissing values in each column:")
    # print(df.isnull().sum())
    df.info()
    df.describe(include='all')
    df.nunique()
   
except Exception as e:
    print(f"Error loading CSV file: {e}")

# --- Probability Calculations ---
print("\n--- Probability Calculations ---")
# Probability of each risk level in the dataset
total = len(df)
risk_probs = df['risk_level'].value_counts(normalize=True)
print("Probability of each risk level:")
print(risk_probs)

# Probability of High risk given low attendance
low_attendance = df[df['attendance_rate'] < 0.75]
prob_high_given_low_attendance = (low_attendance['risk_level'] == 'High').mean()
print(f"\nP(High risk | attendance_rate < 0.75): {prob_high_given_low_attendance:.2f}")

# --- Data Visualization ---
print("\n--- Data Visualization ---")
sns.set(style="whitegrid")

# 1. Distribution of risk levels
plt.figure(figsize=(6,4))
sns.countplot(x='risk_level', data=df, palette='Set2')
plt.title('Distribution of Risk Levels')
plt.xlabel('Risk Level')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# 2. Attendance rate by risk level
plt.figure(figsize=(6,4))
sns.boxplot(x='risk_level', y='attendance_rate', data=df, palette='Set2')
plt.title('Attendance Rate by Risk Level')
plt.xlabel('Risk Level')
plt.ylabel('Attendance Rate')
plt.tight_layout()
plt.show()

# 3. Correlation heatmap
plt.figure(figsize=(10,8))
corr = df.select_dtypes(include=['float64', 'int64']).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.show()

# 4. Pairplot for selected features
selected_features = ['attendance_rate', 'GPA', 'forum_participation_count', 'video_completion_rate', 'risk_level']
sns.pairplot(df[selected_features], hue='risk_level', palette='Set2')
plt.suptitle('Pairplot of Selected Features by Risk Level', y=1.02)
plt.show()