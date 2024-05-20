import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
import numpy as np

# Load data from CSV file
def load_data(csv_file):
    data = pd.read_csv(csv_file)
    return data

# Preprocess data
def preprocess_data(data):
    data = data.dropna()  # Drop missing values
    X = data[['speed']].values  # Feature
    y = data['power'].values  # Target
    return train_test_split(X, y, test_size=0.2, random_state=42)  # Split data

# Train polynomial regression model
def train_model(X_train, y_train, degree=2):
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X_train, y_train)
    return model

# Evaluate model performance
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse, y_pred

# Predict power output for given wind speeds
def predict_power_output(model, wind_speed, num_turbines, power_per_turbine):
    wind_speed = np.array([[wind_speed]])
    predicted_power_per_turbine = model.predict(wind_speed)[0]
    total_predicted_power = num_turbines * min(predicted_power_per_turbine, power_per_turbine)
    return total_predicted_power

# Main function
def main(csv_file):
    data = load_data(csv_file)
    X_train, X_test, y_train, y_test = preprocess_data(data)
    
    # Train the polynomial regression model
    model = train_model(X_train, y_train, degree=2)  # degree=2 for quadratic relationship
    mse, y_pred = evaluate_model(model, X_test, y_test)
    print(f"Mean Squared Error: {mse:.2f}")
    
    # Predict power output for user-provided inputs
    while True:
        try:
            con = 13
            wind_speed_input = input("Enter wind speed for prediction (or 'exit' to quit): ")
            if wind_speed_input.lower() == 'exit':
                break
            wind_speed = float(wind_speed_input)
            num_turbines = int(input("Enter number of turbines: "))
            power_per_turbine = float(input("Enter power per turbine (kW): "))
            
            total_predicted_power = predict_power_output(model, wind_speed, num_turbines, power_per_turbine)*(wind_speed/(0.95 * con))
            print(f"Predicted total power output: {total_predicted_power:.2f} kW")
        except ValueError:
            print("Invalid input. Please enter numeric values.")

# Example usage
if __name__ == "__main__":
    csv_file = 'data.csv'  # Replace with your CSV file path
    main(csv_file)
