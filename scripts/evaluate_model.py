import pandas as pd
from sklearn.metrics import mean_squared_error
import joblib
import yaml

def evaluate_model(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    model = joblib.load(config['model_path'])
    data = pd.read_csv(config['processed_data_path'])

    # Drop the 'Price' and 'Address' columns (ensure only numeric data is used)
    X = data.drop(['Price', 'Address'], axis=1)
    y = data['Price']

    # Convert all columns to numeric, forcing conversion where possible
    X = X.apply(pd.to_numeric, errors='coerce')

    # Drop any rows where conversion failed (i.e., where NaN exists)
    X.dropna(inplace=True)
    y = y[X.index]  # Ensure y is aligned with the filtered X

    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    
    print(f"Model Evaluation: Mean Squared Error = {mse}")

if __name__ == "__main__":
    evaluate_model('params.yaml')
