import pandas as pd  
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression  
from sklearn.metrics import mean_absolute_error, r2_score  

# -------------------------------  
# Step 1 – Load Dataset  
# -------------------------------  
df = pd.read_csv("C:\Users\shree\Downloads\Smart_Traffic_Flow_Prediction_Dataset (2).csv")  

# -------------------------------  
# Step 2 – Data Cleaning  
# -------------------------------  
df.columns = df.columns.str.strip()  
df = df.dropna()  

print("First 5 Rows:\n", df.head())  

# -------------------------------  
# Step 3 – Exploratory Data Analysis (EDA)  
# -------------------------------  
print("\nDataset Info:\n")  
print(df.info())  

print("\nStatistical Summary:\n")  
print(df.describe())  

# -------------------------------  
# Graph 1: Traffic Count vs Time  
# -------------------------------  
plt.figure()  
plt.plot(df['Hour_of_Day'], df['Vehicle_Count'], 'o-')  
plt.title("Traffic Count vs Time")  
plt.xlabel("Hour of Day")  
plt.ylabel("Vehicle Count")  
plt.show()  

# -------------------------------  
# Graph 2: Peak Hour Chart  
# -------------------------------  
plt.figure()  
df.groupby('Hour_of_Day')['Vehicle_Count'].mean().plot(kind='bar')  
plt.title("Peak Hour Traffic")  
plt.xlabel("Hour of Day")  
plt.ylabel("Average Vehicle Count")  
plt.show()  

# -------------------------------  
# Step 4 – Feature Engineering  
# -------------------------------  
df['Prev_Count'] = df['Vehicle_Count'].shift(1)  
df = df.dropna()  

X = df[['Hour_of_Day', 'Day_of_Week', 'Temperature_C', 'Rainfall_mm', 'Is_Holiday', 'Prev_Count']]  
y = df['Vehicle_Count']  

# -------------------------------  
# Step 5 – Train Model  
# -------------------------------  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  

model = LinearRegression()  
model.fit(X_train, y_train)  

# -------------------------------  
# Step 6 – Prediction  
# -------------------------------  
y_pred = model.predict(X_test)  

# -------------------------------  
# Step 7 – Results & Insights  
# -------------------------------  
print("\nModel Evaluation:")  
print("MAE:", mean_absolute_error(y_test, y_pred))  
print("R2 Score:", r2_score(y_test, y_pred))  

# Peak & Low traffic  
peak_hour = df.groupby('Hour_of_Day')['Vehicle_Count'].mean().idxmax()  
low_hour = df.groupby('Hour_of_Day')['Vehicle_Count'].mean().idxmin()  

print("\nMost Congested Time (Peak Hour):", peak_hour)  
print("Least Busy Hour:", low_hour)  

# -------------------------------  
# Step 8 – Custom Prediction  
# -------------------------------  
sample = [[10, 2, 30, 0, 0, 50]]  
pred = model.predict(sample)  

print("\nPredicted Vehicle Count:", pred[0])  
