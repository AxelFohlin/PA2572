import numpy as np

def prepare_features(embedding_model, amenities: list[str], numerical_data: list[float]):
    embedded = embedding_model.encode(amenities)
    embedded = np.mean(embedded, axis=0).reshape(1, -1)
    numerical = np.array(numerical_data).reshape(1, -1)
    return np.hstack((embedded, numerical))

def predict_price(model, features):
    return model.predict(features)[0]
