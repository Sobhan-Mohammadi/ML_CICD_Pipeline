import pytest
import pandas as pd
from scripts.data_preprocessing import preprocess_data

def test_preprocess_data():
    data = pd.DataFrame({
        'feature1': [1, 2, 3, None],
        'feature2': [4, None, 6, 7]
    })
    processed_data = preprocess_data(data)
    
    assert processed_data.isnull().sum().sum() == 0
test_train_model.py:

import pytest
import os
from scripts.train_model import train_model

def test_train_model():
    train_model('params.yaml')
    assert os.path.exists('models/house_price_model.pkl')
