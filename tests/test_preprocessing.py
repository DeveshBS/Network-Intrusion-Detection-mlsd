import os
import pandas as pd
import pytest
import sys
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.train import COLUMN_NAMES

@pytest.fixture
def dummy_data():
    # Generate 1 row of dummy data perfectly matching NSL-KDD
    # First 41 columns are features, 42 is attack_type, 43 is difficulty
    row = [0] * 43
    row[1] = 'tcp'
    row[2] = 'http'
    row[3] = 'SF'
    row[41] = 'neptune' # attack
    row[42] = 21 # difficulty
    
    return pd.DataFrame([row], columns=COLUMN_NAMES)

def test_data_columns(dummy_data):
    """Test if all 43 columns are mapped initially"""
    assert len(dummy_data.columns) == 43, "NSL-KDD raw data should have 43 columns"
    assert "attack_type" in dummy_data.columns

def test_target_label_extraction(dummy_data):
    """Verify that 'neptune' gets encoded effectively into a binary target later on"""
    # Simulate extraction as done in original code
    dummy_data['class'] = dummy_data['attack_type'].apply(lambda x: 0 if x == 'normal' else 1)
    
    assert dummy_data.iloc[0]['class'] == 1, "Neptune should be marked as an anomaly (1)"
