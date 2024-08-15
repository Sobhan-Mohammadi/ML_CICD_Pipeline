import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import yaml

def train_model(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Load the processed data, skipping the header row if necessary
    data = pd.read_csv(config['processed_data_path'])
    
    # Drop the 'Price' column to get features and set 'Price' as target
    X = data.drop(['Price', 'Address'], axis=1)
    y = data['Price']

    # Convert the data to numeric if it's not already
    X = X.apply(pd.to_numeric, errors='coerce')
    y = pd.to_numeric(y, errors='coerce')

    # Remove any rows with NaN values after conversion
    X.dropna(inplace=True)
    y.dropna(inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config['train']['test_size'], random_state=config['train']['random_state']
    )

    model = RandomForestRegressor(
        n_estimators=config['train']['n_estimators'], max_depth=config['train']['max_depth']
    )
    model.fit(X_train, y_train)

    joblib.dump(model, config['model_path'])

if __name__ == "__main__":
    train_model('params.yaml')
