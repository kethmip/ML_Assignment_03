
import numpy as np
import cloudpickle

with open("best_model.pkl", "rb") as f: model = cloudpickle.load(f)

def test_model_input():
    X_sample = np.array([[1.0, 0.0, 1.0, 0.0]])
    prediction = model.predict(X_sample)
    assert prediction is not None

def test_model_output_shape():
    X_sample = np.array([[1.0, 0.0, 1.0, 0.0]])
    prediction = model.predict(X_sample)
    assert prediction.shape == (1,)
