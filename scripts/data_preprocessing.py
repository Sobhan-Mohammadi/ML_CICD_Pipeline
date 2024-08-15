import pandas as pd
import yaml

def load_data(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Manually define the column names
    column_names = [
        'Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
        'Avg. Area Number of Bedrooms', 'Area Population', 'Price', 'Address'
    ]
    
    # Load the data with specified column names
    data = pd.read_csv(config['data_path'], names=column_names)
    return data

def preprocess_data(data):
    # Handle numeric columns only for missing value imputation
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
    return data

if __name__ == "__main__":
    config_path = 'params.yaml'
    data = load_data(config_path)
    processed_data = preprocess_data(data)
    processed_data.to_csv('data/processed/processed_housing.csv', index=False)
