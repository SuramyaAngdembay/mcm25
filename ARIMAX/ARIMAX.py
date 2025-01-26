# this was even before i considered optimization. Here, I 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data
df = pd.read_csv('data_with_WR.csv')

us_data = df[df['NOC'] == 'United States'] 

data = pd.DataFrame({
    'Year': us_data['Year'],
    'Country': us_data['NOC'],
    'Medals': us_data['Total_Medal_Count'],
    'Athletes': us_data['Total_Athletes'],
    'Host': us_data['host_status'],
    'Winning_Ratio': us_data['Winning_Ratio']
})

data = data.sort_values(by='Year')

data['Medals_lag'] = data['Medals'].shift(1)
data['Medals_lag2'] = data['Medals'].shift(2)
data['Medals_lag3'] = data['Medals'].shift(3)

data['Winning_Ratio_lag2'] = data['Winning_Ratio'].shift(2)

# data = data.fillna(0)
data = data.dropna()

# Define dependent and exogenous variables
y = data['Medals']
# X = data[['Athletes', 'Host', 'Medals_lag', 'Medals_lag3', 'Winning_Ratio']]
# X = data[['Athletes', 'Host', 'Medals_lag', 'Medals_lag2', 'Medals_lag3', 'Winning_Ratio']]
# X = data[['Athletes', 'Host', 'Medals_lag', 'Medals_lag2', 'Winning_Ratio']]
# X = data[['Athletes', 'Host', 'Medals_lag3', 'Winning_Ratio']]
X = data[['Athletes', 'Host', 'Medals_lag2', 'Winning_Ratio']]
# X = data[['Athletes', 'Host', 'Medals_lag2', 'Winning_Ratio_lag2']]


# X = data[['Athletes', 'Host', 'Medals_lag', 'Winning_Ratio']]
# X = data[['Athletes', 'Host', 'Winning_Ratio']]

# Split data into training (80%) and testing (20%)
split_index = int(len(data) * 0.8)
y_train, y_test = y[:split_index], y[split_index:]
X_train, X_test = X[:split_index], X[split_index:]

# Fit an ARIMAX model
model = SARIMAX(y_train, exog=X_train, order=(1, 1, 1))  # Adjust p, d, q as needed
# model = SARIMAX(y_train, exog=X_train, order=(1, 1, 1))  # Adjust p, d, q as needed
# model = SARIMAX(y_train, exog=X_train, order=(10, 0, 3))  # Adjust p, d, q as needed
result = model.fit(disp=False)

# Predict on test set
y_pred = result.forecast(steps=len(y_test), exog=X_test)

# Replace negative predictions with 0
y_pred = np.where(y_pred < 0, 0, y_pred)

# Round predictions to the nearest integer
# y_pred = np.round(y_pred).astype(int)

# Calculate errors
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared: {r2:.2f}")


# Plot actual vs predicted
plt.figure(figsize=(10, 6))
plt.plot(data['Year'], y, label="Actual Medals", marker="o")
plt.plot(data['Year'][split_index:], y_pred, label="Predicted Medals", marker="o", linestyle="--")
plt.xlabel("Year")
plt.ylabel("Medals")
plt.title("ARIMAX Prediction of Medals Won")
plt.legend()
plt.grid()
plt.show()