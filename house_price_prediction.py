import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Create a sample dataset (rooms, size_sqft, location_score, price)
np.random.seed(42)
n = 200
rooms = np.random.randint(1, 6, n)
size = np.random.randint(500, 3500, n)
location_score = np.random.uniform(1, 10, n)
# price = 50000*rooms + 150*size + 30000*location_score + noise
noise = np.random.normal(0, 25000, n)
price = 50000*rooms + 150*size + 30000*location_score + noise

df = pd.DataFrame({
    'rooms': rooms,
    'size_sqft': size,
    'location_score': location_score,
    'price': price
})

# Save to CSV (so you can also load your own)
df.to_csv('house_data.csv', index=False)
print("Sample dataset saved as 'house_data.csv'")

# Load data (you can replace with any CSV having same columns)
data = pd.read_csv('house_data.csv')
print(data.head())

# Features and target
X = data[['rooms', 'size_sqft', 'location_score']]
y = data['price']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Performance ---")
print(f"Mean Absolute Error: ${mae:,.2f}")
print(f"R² Score: {r2:.4f}")
print("Coefficients:")
for feat, coef in zip(X.columns, model.coef_):
    print(f"  {feat}: ${coef:,.2f}")
print(f"Intercept: ${model.intercept_:,.2f}")

# Predict a new house
print("\n--- Predict a new house ---")
rooms_new = int(input("Number of rooms: "))
size_new = float(input("Size (sq ft): "))
loc_new = float(input("Location score (1-10): "))
new_price = model.predict([[rooms_new, size_new, loc_new]])[0]
print(f"Estimated price: ${new_price:,.2f}")

# Optional: plot actual vs predicted
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()