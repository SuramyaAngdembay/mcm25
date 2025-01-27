# it fell into the trap of overfitting as we used the test set to tune the hyperparameters.
# but, I have now used it to tune the hyperparameters on the training set and then used the test set to evaluate the model.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import optuna

# Load data
df = pd.read_csv('data_with_WR.csv')
us_data = df[df['NOC'] == 'Canada']

data = pd.DataFrame({
    'Year': us_data['Year'],
    'Country': us_data['NOC'],
    'Medals': us_data['Total_Medal_Count'],
    'Athletes': us_data['Total_Athletes'],
    'Host': us_data['host_status'],
    'Winning_Ratio': us_data['Winning_Ratio']
})

# Convert 'Year' to datetime and set as index
data['Year'] = pd.to_datetime(data['Year'], format='%Y')
data = data.set_index('Year')

# Lag features
data['Medals_lag1'] = data['Medals'].shift(1)
data['Medals_lag2'] = data['Medals'].shift(2)
data['Medals_lag3'] = data['Medals'].shift(3)

data = data.dropna()

# Dependent and independent variables
y = data['Medals']
X = data[['Athletes', 'Host', 'Medals_lag2', 'Winning_Ratio']]

# Train-test split
split_index = int(len(data) * 0.8)
y_train, y_test = y[:split_index], y[split_index:]
X_train, X_test = X[:split_index], X[split_index:]

# Bayesian Optimization with Optuna
def objective(trial):
    p = trial.suggest_int('p', 0, 20)
    d = trial.suggest_int('d', 0, 3)
    q = trial.suggest_int('q', 0, 20)
    try:
        model = SARIMAX(y_train, exog=X_train, order=(p, d, q),
                        enforce_stationarity=False, enforce_invertibility=False)
        result = model.fit(disp=False, maxiter=1000)
        y_pred = result.forecast(steps=len(y_train), exog=X_train)
        rmse = np.sqrt(mean_squared_error(y_train, y_pred))
        return rmse
    except:
        return float('inf')

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

# Best parameters
best_params = study.best_params
print(f"Best parameters: {best_params}")
print(f"Best RMSE: {study.best_value}")

# Final model with best parameters
# final_model = SARIMAX(y_train, exog=X_train, order=(1, 1, 1),
#                       enforce_stationarity=False, enforce_invertibility=False)
final_model = SARIMAX(y_train, exog=X_train, order=(best_params['p'], best_params['d'], best_params['q']),
                      enforce_stationarity=False, enforce_invertibility=False)
final_result = final_model.fit(disp=False, maxiter=1000)

# Prediction
y_pred = final_result.forecast(steps=len(y_test), exog=X_test)

# Calculate errors
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R-squared: {r2:.2f}")

# Plot actual vs predicted
plt.figure(figsize=(10, 6))
plt.plot(data.index, y, label="Actual Medals", marker="o")
plt.plot(data.index[split_index:], y_pred, label="Predicted Medals", marker="o", linestyle="--")
plt.xlabel("Year")
plt.ylabel("Medals")
plt.title("Bayesian Optimization with SARIMAX")
plt.legend()
plt.grid()
plt.show()