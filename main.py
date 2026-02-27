import pandas as pd 
import numpy as np 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# load dataset 
data = sns.load_dataset("penguins")
print("first 5 rows of dataset:\n")
print(data.head())
data = data.dropna()

#define features and targets
X = data[[
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g"
]]

y = data["species"]

#train - test split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=42)

#feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# make predictions
y_pred = model.predict(X_test)

# evaluate model
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# text with custom inputs
print("\n CUSTOM PREDICTION")

bill_length = float(input("Enter bill length (mm): "))
bill_depth = float(input("Enter bill depth (mm): "))
flipper_length = float(input("Enter flipper length (mm): "))
body_mass = float(input("Enter body mass (g): "))

new_data = np.array([[bill_length, bill_depth, flipper_length, body_mass]])

# Scale using same scaler
new_data_scaled = scaler.transform(new_data)

prediction = model.predict(new_data_scaled)

print("Predicted Penguin Species:", prediction[0])