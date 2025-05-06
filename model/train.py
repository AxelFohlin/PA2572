import numpy as np
from sklearn.ensemble import RandomForestRegressor

def train_model(df):
    numerical_features = df[['bedrooms', 'bathrooms', 'accommodates', 'minimum_nights']].values
    embedded_features = np.array(df['embedded_amenities'].tolist())
    X = np.hstack((embedded_features, numerical_features))
    y = df['price'].values

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model
