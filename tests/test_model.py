import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the trained Random Forest model
rf_model = joblib.load('rf_model.pkl')

# Load the dataset (you can replace this with any test dataset)
# For simplicity, we’ll use EW-MAX.csv (you can change this if you have a separate test dataset)
data = pd.read_csv('EW-MAX.csv')

# Prepare the features (X) and target (y)
# Assuming that the target variable is the "Close" price, and we use previous day's data for prediction
X = data[['Close', 'High', 'Low', 'Volume', 'Open']].values  # Use actual column names
y_true = data['Close'].values  # True target values

# Make predictions using the model
y_pred = rf_model.predict(X)

# Calculate Evaluation Metrics
def test_model_metrics():
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R-Squared (R2): {r2}")

    # Assert to ensure that values are within an acceptable range
    assert mse < 1.0, f"MSE is too high: {mse}"
    assert mae < 1.0, f"MAE is too high: {mae}"
    assert r2 > 0.95, f"R2 is too low: {r2}"

# Call the test function
test_model_metrics()
