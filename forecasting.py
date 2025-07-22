import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

path = kagglehub.dataset_download("ziya07/college-student-management-dataset")
csv_file = path + "/college_student_management_data.csv"

df = pd.read_csv(csv_file)

# --- Forecasting Risk Level Proportions for the Next Five Years ---
print("\n--- Forecasting Risk Level Proportions for the Next Five Years ---")

# Assume current proportions as baseline
risk_probs = df['risk_level'].value_counts(normalize=True).sort_index()
print("Current risk level proportions:")
print(risk_probs)

# Simulate a simple linear trend 
# For demonstration, let's assume a 1% annual decrease in High risk, 0.5% increase in Low, rest to Medium
years = np.arange(2024, 2029)
forecast = pd.DataFrame(index=years, columns=risk_probs.index)


forecast.loc[2024] = risk_probs.values

for i, year in enumerate(years[1:], start=1):
    prev = forecast.loc[years[i-1]].astype(float)
    high = max(prev['High'] - 0.01, 0)
    low = min(prev['Low'] + 0.005, 1)
   
    medium = 1 - high - low
    forecast.loc[year] = [high, low, medium]

print("\nForecasted proportions:")
print(forecast)

plt.figure(figsize=(10,6))
colors = {'High': '#e74c3c', 'Medium': '#f1c40f', 'Low': '#2ecc71'}
for col in forecast.columns:
    plt.plot(forecast.index, forecast[col], marker='o', label=col, color=colors.get(col, None), linewidth=2)

    for x, y in zip(forecast.index, forecast[col]):
        plt.text(x, y + 0.01, f"{y:.2f}", ha='center', va='bottom', fontsize=10)
plt.title('Forecasted Proportion of Student Risk Levels (2024-2028)', fontsize=16)
plt.xlabel('Year', fontsize=13)
plt.ylabel('Proportion', fontsize=13)
plt.ylim(0, 1.05)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Risk Level', fontsize=12, title_fontsize=13, loc='upper right')
plt.tight_layout()
plt.show()

