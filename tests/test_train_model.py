import pytest
import os
from scripts.train_model import train_model

def test_train_model():
    train_model('params.yaml')
    assert os.path.exists('models/house_price_model.pkl')
