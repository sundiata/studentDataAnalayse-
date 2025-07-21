import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


path = kagglehub.dataset_download("ziya07/college-student-management-dataset")
csv_file = path + "/college_student_management_data.csv"

df = pd.read_csv(csv_file)

sns.set(style="whitegrid", palette="Set2")

# --- Grouped Analysis: Gender vs Risk Level ---
print("\n--- Gender vs Risk Level ---")
gender_risk = df.groupby(['gender', 'risk_level']).size().unstack(fill_value=0)
print(gender_risk)

plt.figure(figsize=(8,5))
gender_risk_norm = gender_risk.div(gender_risk.sum(axis=1), axis=0)
gender_risk_norm.plot(kind='bar', stacked=True, colormap='Set2')
plt.title('Proportion of Risk Levels by Gender')
plt.xlabel('Gender')
plt.ylabel('Proportion')
plt.legend(title='Risk Level')
plt.tight_layout()
plt.show()

# --- Grouped Analysis: Major vs Risk Level ---
print("\n--- Major vs Risk Level (Top 10 Majors) ---")
major_counts = df['major'].value_counts().head(10).index
major_risk = df[df['major'].isin(major_counts)].groupby(['major', 'risk_level']).size().unstack(fill_value=0)
print(major_risk)

plt.figure(figsize=(12,6))
major_risk_norm = major_risk.div(major_risk.sum(axis=1), axis=0)
major_risk_norm.plot(kind='bar', stacked=True, colormap='Set2')
plt.title('Proportion of Risk Levels by Major (Top 10)')
plt.xlabel('Major')
plt.ylabel('Proportion')
plt.legend(title='Risk Level')
plt.tight_layout()
plt.show()

# --- Grouped Analysis: Enrollment Status vs Risk Level ---
print("\n--- Enrollment Status vs Risk Level ---")
enroll_risk = df.groupby(['enrollment_status', 'risk_level']).size().unstack(fill_value=0)
print(enroll_risk)

plt.figure(figsize=(8,5))
enroll_risk_norm = enroll_risk.div(enroll_risk.sum(axis=1), axis=0)
enroll_risk_norm.plot(kind='bar', stacked=True, colormap='Set2')
plt.title('Proportion of Risk Levels by Enrollment Status')
plt.xlabel('Enrollment Status')
plt.ylabel('Proportion')
plt.legend(title='Risk Level')
plt.tight_layout()
plt.show()

# --- Line Graphs: Risk Level Trends ---

# 1. Line graph for Gender vs Risk Level
plt.figure(figsize=(8,5))
gender_risk_norm.plot(kind='line', marker='o')
plt.title('Trend of Risk Levels by Gender')
plt.xlabel('Gender')
plt.ylabel('Proportion')
plt.legend(title='Risk Level')
plt.tight_layout()
plt.show()

# 2. Line graph for Major vs Risk Level (Top 10)
plt.figure(figsize=(12,6))
major_risk_norm.plot(kind='line', marker='o')
plt.title('Trend of Risk Levels by Major (Top 10)')
plt.xlabel('Major')
plt.ylabel('Proportion')
plt.legend(title='Risk Level')
plt.tight_layout()
plt.show()

# 3. Line graph for Enrollment Status vs Risk Level
plt.figure(figsize=(8,5))
enroll_risk_norm.plot(kind='line', marker='o')
plt.title('Trend of Risk Levels by Enrollment Status')
plt.xlabel('Enrollment Status')
plt.ylabel('Proportion')
plt.legend(title='Risk Level')
plt.tight_layout()
plt.show() 