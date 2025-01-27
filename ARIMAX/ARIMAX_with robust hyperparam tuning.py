import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import optuna

# Data Preparation
def prepare_data():
    # Load data
    df = pd.read_csv('data_with_WR.csv')
    us_data = df[df['NOC'] == 'Italy']

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

    return data

# Robust Objective Function for Optuna
def robust_objective(trial, y_train, X_train):
    # Suggest hyperparameters with constrained search space
    p = trial.suggest_int('p', 0, 10)
    d = trial.suggest_int('d', 0, 3)
    q = trial.suggest_int('q', 0, 10)
    
    try:
        # Create and fit the model
        model = SARIMAX(y_train, exog=X_train, order=(p, d, q),
                        enforce_stationarity=False, enforce_invertibility=False)
        result = model.fit(disp=False, maxiter=1000)
        
        # Predict on training data
        y_pred = result.get_forecast(steps=len(y_train), exog=X_train).predicted_mean
        
        # Calculate performance metrics
        mse = mean_squared_error(y_train, y_pred)
        r2 = r2_score(y_train, y_pred)
        
        # Robust objective function
        if r2 < 0:
            return float('inf')
        
        # Minimize MSE while favoring higher R-squared
        objective_value = mse / (r2 + 1e-10)
        
        return objective_value
    
    except Exception as e:
        return float('inf')

# Tune SARIMAX Hyperparameters
def tune_sarimax(y_train, X_train):
    study = optuna.create_study(direction='minimize')
    
    def objective(trial):
        return robust_objective(trial, y_train, X_train)
    
    study.optimize(objective, n_trials=200)
    
    return study.best_params, study.best_value

# Main Execution
def main():
    # Prepare data
    data = prepare_data()

    # Dependent and independent variables
    y = data['Medals']
    X = data[['Athletes', 'Host', 'Medals_lag2', 'Winning_Ratio']]

    # Train-test split
    split_index = int(len(data) * 0.8)
    y_train, y_test = y[:split_index], y[split_index:]
    X_train, X_test = X[:split_index], X[split_index:]

    # Tune hyperparameters
    best_params, best_score = tune_sarimax(y_train, X_train)
    print(f"Best parameters: {best_params}")
    print(f"Best Score: {best_score}")

    # Final model with best parameters
    final_model = SARIMAX(y_train, exog=X_train, 
                          order=(best_params['p'], best_params['d'], best_params['q']),
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
    plt.plot(y, label="Actual Medals", marker="o")
    plt.plot(data.index[split_index:], y_pred, label="Predicted Medals", marker="o", linestyle="--")
    plt.xlabel("Year")
    plt.ylabel("Medals")
    plt.title("SARIMAX Model: Actual vs Predicted Medals")
    plt.legend()
    plt.grid()
    plt.show()

# Run the main function
if __name__ == "__main__":
    main()