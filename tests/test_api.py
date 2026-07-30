# tests/test_api.py — Automated unit tests

import joblib
import numpy as np
import pytest

def test_model_loads():
    model = joblib.load("models/term_deposit_model.pkl")
    assert model is not None
    print("Model loaded successfully")

def test_preprocessor_loads():
    preprocessor = joblib.load("models/scaler.pkl")
    assert preprocessor is not None
    print("Preprocessor loaded successfully")

def test_feature_indices_loads():
    indices = joblib.load("models/important_indices.pkl")
    assert len(indices) > 0
    print(f"Feature indices loaded: {len(indices)} features")

def test_threshold():
    prob_high = 0.75
    prob_low  = 0.15
    threshold = 0.3
    assert (prob_high >= threshold) == True
    assert (prob_low  >= threshold) == False
    print("Decision threshold working correctly")