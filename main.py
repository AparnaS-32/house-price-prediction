import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
data = pd.read_csv("house_data.csv")

# Separate features and target
X = data[["area", "bedrooms", "age"]]
y = data["price"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Test model
predictions = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("HOUSE PRICE PREDICTION")
print("----------------------")
print("Model training completed!")

print("\nMODEL EVALUATION")
print("----------------")
print("Mean Absolute Error:", round(mae, 2))
print("R2 Score:", round(r2, 4))

# Predict a new house
new_house = pd.DataFrame({
    "area": [1600],
    "bedrooms": [3],
    "age": [4]
})

predicted_price = model.predict(new_house)

print("\nPredicted House Price:")
print("Rs.", round(predicted_price[0], 2))