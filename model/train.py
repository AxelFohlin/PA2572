import numpy as np
from sklearn.ensemble import RandomForestRegressor

def train_model(df, vectorizer):
    tfidf_matrix = vectorizer.fit_transform(df['amenities_clean'])
    numerical_features = df[['bedrooms', 'bathrooms', 'accommodates', 'minimum_nights', 'maximum_nights', 'longitude', 'latitude']].values

    print("NUMERICAL: ", numerical_features[9])

    X = np.hstack((tfidf_matrix.toarray(), numerical_features))
    y = df['price']

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model
