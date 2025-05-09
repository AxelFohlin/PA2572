import numpy as np

def prepare_features(vectorizer, amenities: list[str], numerical_data: list[float]):
    amenities_str = " ".join(amenities)

    tfidf_matrix = vectorizer.transform([amenities_str])
    numerical = np.array(numerical_data).reshape(1, -1)

    return np.hstack((tfidf_matrix.toarray(), numerical))


def predict_price(model, features):
    return model.predict(features)[0]
